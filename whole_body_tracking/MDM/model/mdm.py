import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from model.BERT.BERT_encoder import load_bert
from utils.misc import WeightedSum

from data_loaders.dataset import get_features_dims
import math


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def create_causal_attention_mask(sequence_type_array):
    """
    Actions have a causal mask
    States have full attention over states


    A0   [[False,  True,  True,  True,  True,  True],
    S0    [ True, False,  True, False,  True, False],
    A1    [False, False, False,  True,  True,  True],
    S1    [ True, False,  True, False,  True, False],
    A2    [False, False, False, False, False,  True],
    S2    [ True, False,  True, False,  True, False]]
            A0    S0      A1    S1      A2    S2
    """
    seq_len = len(sequence_type_array)
    is_state = torch.tensor(sequence_type_array, dtype=torch.bool)
    
    # Start with a standard causal mask (tril) to prevent attending to future elements
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))



    # Apply the custom rules
    for i in range(seq_len):
        if is_state[i]:
            # If the current token is a state, it can attend to all other states.
            # This logic needs to override the causal mask for other states.
            # A state at index i can attend to any state j, regardless of j's position.
            state_indices = torch.where(is_state)[0]
            action_indices = torch.where(~is_state)[0]
            mask[i, state_indices] = True
            mask[i, action_indices] = False
        else:
            # If the current token is an action, it can only attend to
            # past actions and past/current states (this is handled by the tril mask).
            pass # The initial causal mask is sufficient here.

    return ~mask



import torch
import torch.nn as nn
from torch.nn import functional as F

class FilmTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    An encoder layer that inherits from the base PyTorch class and adds FiLM.
    
    Note: We are assuming the default `norm_first=False`. If you use `norm_first=True`,
    the logic inside the forward pass would need to be adjusted accordingly.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False, 
                 norm_first=False, device=None, dtype=None):
        
        # Call the parent class's __init__ to handle all the layer setup.
        # This is the main benefit of inheritance.
        super().__init__(
            d_model, nhead, dim_feedforward, dropout, activation,
            layer_norm_eps, batch_first, norm_first, device, dtype
        )
        # No need to define self.self_attn, self.linear1, etc. It's all done for us.

    def forward(self, src, gamma, beta, src_mask=None, src_key_padding_mask=None):
        """
        The forward pass, modified to accept and apply FiLM parameters.
        
        Args:
            src (Tensor): The input sequence. Shape: [seq_len, bs, features].
            gamma (Tensor): The FiLM scale parameter. Shape: [bs, features].
            beta (Tensor): The FiLM shift parameter. Shape: [bs, features].
            src_mask (Optional[Tensor]): The mask for the src sequence.
            src_key_padding_mask (Optional[Tensor]): The mask for the src keys per batch.
        """
        # Ensure we're not using norm_first, as this implementation is for the standard post-norm architecture
        if self.norm_first:
            raise NotImplementedError("This custom FiLM layer is implemented for norm_first=False.")
        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
    
        gamma_reshaped = gamma.unsqueeze(0)
        beta_reshaped = beta.unsqueeze(0)
        
        x_modulated = gamma_reshaped * x + beta_reshaped
        x = self.norm2(x + self._ff_block(x_modulated))

        return x
    

# The FilmGenerator and PositionalEncoding classes remain exactly the same as before.
class FilmGenerator(nn.Module):
    def __init__(self, cond_dim, features):
        super().__init__()
        self.generator = nn.Linear(cond_dim, features * 2)

    def forward(self, conditioning_vector):
        gamma_beta = self.generator(conditioning_vector)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)
        return gamma, beta


class FilmTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, cond_dim, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.film_generator = FilmGenerator(cond_dim, d_model)

        # The ONLY CHANGE is here: we instantiate our new inherited class.
        self.layers = nn.ModuleList([
            FilmTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, src, conditioning_vector, src_mask=None, src_key_padding_mask=None):
        src = self.pos_encoder(src * math.sqrt(self.d_model))
        
        gamma, beta = self.film_generator(conditioning_vector)

        output = src
        for layer in self.layers:
            # The call signature is the same.
            output = layer(output, gamma, beta, src_mask=src_mask, 
                             src_key_padding_mask=src_key_padding_mask)
            
        return output
    
# Component 3: The main encoder class (inherits from the base encoder)
class FilmTransformerEncoder(nn.TransformerEncoder):
    """
    An encoder that inherits from the base PyTorch class. It uses a custom
    FilmTransformerEncoderLayer and manages the FiLM parameter generation
    and distribution during the forward pass.
    """
    def __init__(self, encoder_layer, num_layers, d_model, cond_dim, norm=None):
        # This is crucial: call the parent __init__. It will handle cloning
        # the encoder_layer `num_layers` times to create self.layers.
        super().__init__(encoder_layer, num_layers, norm)

        # Add our custom FiLM generator
        self.film_generator = FilmGenerator(cond_dim, d_model)

    def forward(self, src, cond, mask=None, src_key_padding_mask=None):
        """
        The forward pass, modified to accept and apply FiLM parameters.
        
        Args:
            src (Tensor): The input sequence. Shape: [seq_len, bs, features].
            conditioning_vector (Tensor): The conditioning info. Shape: [bs, cond_dim].
            mask (Optional[Tensor]): The mask for the src sequence.
            src_key_padding_mask (Optional[Tensor]): The mask for src keys per batch.
        """
        output = src
        
        # 1. Generate all FiLM parameters at once
        # Shapes: [bs, num_layers, features]
        gamma, beta = self.film_generator(cond)

        # 2. Loop through the layers and apply FiLM
        # self.layers was created for us by the nn.TransformerEncoder __init__
        for i, mod in enumerate(self.layers):

            # The custom layer expects gamma and beta
            output = mod(output, gamma, beta, 
                         src_mask=mask, 
                         src_key_padding_mask=src_key_padding_mask)

        # 3. Apply the final optional normalization
        if self.norm is not None:
            output = self.norm(output)

        return output

class MDM(nn.Module):
    def __init__(self, cfg):
        """
        modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs
        """
        super().__init__()
        self.cfg = cfg
        self.mdm_cfg = cfg.model

        self.njoints = 1
        self.latent_dim = self.mdm_cfg.latent_dim
        self.ff_size = self.mdm_cfg.ff_size
        self.num_layers = self.mdm_cfg.num_layers
        self.num_heads = self.mdm_cfg.num_heads
        self.dropout = self.mdm_cfg.dropout
        #TODO i need to set the pe dropout lower then attn dropout
        self.pe_dropout = self.mdm_cfg.dropout
        self.activation = self.mdm_cfg.activation
        self.action_input_feats = get_features_dims(self.cfg.dataset.action_data_keys)
        self.robot_input_feats = get_features_dims(self.cfg.dataset.state_data_keys) if self.mdm_cfg.prediction_type == 'state' else get_features_dims(self.cfg.dataset.context_data_keys)

        self.cond_mode = self.mdm_cfg.cond_mode
        self.cond_mask_prob = self.mdm_cfg.cond_mask_prob 
        self.arch = self.mdm_cfg.arch
        self.emb_policy = self.mdm_cfg.emb_policy
        self.pred_len = self.mdm_cfg.pred_len*2
        self.context_len = self.mdm_cfg.context_len*2 #states and actions
        self.total_len = self.pred_len + self.context_len
        self.is_prefix_comp = self.mdm_cfg.prefix_comp
        self.prediction_type = self.mdm_cfg.prediction_type
        self.action_pred_len = self.mdm_cfg.action_pred_len

    
        self.cross_attn_conds = self.mdm_cfg.cross_attn_conds
        self.attn_type = self.mdm_cfg.attn_type
        if self.attn_type == 'causal':
            ordering = torch.zeros((self.pred_len + (self.context_len if self.is_prefix_comp else 0)),dtype=torch.bool)
            ordering[1::2] = 1
            m = create_causal_attention_mask(ordering)
            self.causal_mask = m.unsqueeze(0) # [1, slen, slen]

        #TODO we can get rid of this
        self.emb_trans_dec = self.mdm_cfg.emb_trans_dec

        if self.mdm_cfg.prediction_type == self.mdm_cfg.prefix_type:
            self.input_process = InputProcess(
                self.action_input_feats,             
                self.robot_input_feats, 
                self.latent_dim)
            
        else:
            self.prefix_input_feats = get_features_dims(self.cfg.dataset.context_data_keys)
            self.input_process = MixedInputProcess(
                self.action_input_feats,
                self.robot_input_feats,
                self.prefix_input_feats,
                self.latent_dim
            )

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.pe_dropout, max_len=cfg.get('pos_embed_max_len', 5000))


        if 'target_vel' in self.cond_mode:
            self.embed_target_vel = nn.Linear(3, self.latent_dim)

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            if 'target_vel' in self.cond_mode:
                seqTransEncoderLayer = FilmTransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.ff_size,
                    dropout=self.dropout,
                    activation=self.activation
                )
                self.seqTransEncoder = FilmTransformerEncoder(
                    seqTransEncoderLayer,
                    num_layers=self.num_layers,
                    d_model=self.latent_dim,
                    cond_dim=3
                )
            else:
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                            num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)

    
        if self.mdm_cfg.get('kerras_timestep',False):
            self.embed_timestep = KerrasTimestepEmbedder(self.latent_dim)
        else:
            self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if 'no_cond' not in self.cond_mode:
            if 'text' in self.cond_mode:
                # We support CLIP encoder and DistilBERT
                print('EMBED TEXT')
                
                self.text_encoder_type = cfg.get('text_encoder_type', 'bert')
                
                assert self.arch == 'trans_dec'
                # assert self.emb_trans_dec == False # passing just the time embed so it's fine
                print("Loading BERT...")
                # bert_model_path = 'model/BERT/distilbert-base-uncased'
                bert_model_path = 'distilbert/distilbert-base-uncased'
                self.clip_model = load_bert(bert_model_path)  # Sorry for that, the naming is for backward compatibility
                self.encode_text = self.bert_encode_text
                self.clip_dim = 768

                
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)

        # if self.cross_attn_conds is not None and 'context' in self.cross_attn_conds and 'actions' in self.cross_attn_conds:
        #     dim = get_features_dims(self.cfg.dataset.action_data_keys)
        #     self.emb_ca_actions = nn.Linear(dim, self.latent_dim)

        #     dim = get_features_dims(self.cfg.dataset.context_data_keys)
        #     self.emb_ca_context = nn.Linear(dim, self.latent_dim)
        # elif self.cross_attn_conds is not None and 'context' in self.cross_attn_conds and not 'actions' in self.cross_attn_conds or not 'context' in self.cross_attn_conds and 'actions' in self.cross_attn_conds:
        #     #TODO move to apply rules
        #     raise ValueError('Cross attention context requires both acitons and context')

            
                

        self.output_process = OutputProcess(
            self.action_input_feats, 
            self.robot_input_feats,
            self.latent_dim, 
            self.njoints)
        
        print('MDM created =======================')





    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[-2]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
    def mask_cond_return_mask(self, cond, mask_prob, force_mask=False):
        bs = cond.shape[-2]
        if force_mask:
            return torch.zeros_like(cond), torch.ones((1,bs,1), device=cond.device)
        elif self.training and mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask), mask
        else:
            return cond, torch.zeros((1,bs,1), device=cond.device)

    def bert_encode_text(self, raw_text):
        # enc_text = self.clip_model(raw_text)
        # enc_text = enc_text.permute(1, 0, 2)
        # return enc_text
        enc_text, mask = self.clip_model(raw_text)  # self.clip_model.get_last_hidden_state(raw_text, return_mask=True)  # mask: False means no token there
        enc_text = enc_text.permute(1, 0, 2)
        mask = ~mask  # mask: True means no token there, we invert since the meaning of mask for transformer is inverted  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        return enc_text, mask

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, _, _, _ = x.shape
        action_dim = y['action_dim']

        x_action = x[:, :, :action_dim, :]
        x_robot = x[:, :, action_dim:, :]


            
        if self.cfg.model.independant_noise:
            if not self.training:
                timesteps = timesteps.repeat(1,2)
            assert timesteps.dim() == 2
            time_emb_a = self.embed_timestep(timesteps[:, 0]).repeat(int(self.context_len/2)+ int(self.pred_len/2), 1, 1) # [seq_len, bs, d]
            time_emb_a[:int(self.context_len/2)] = 0. # no time noise embedding for context vars
            time_emb_s = self.embed_timestep(timesteps[:, 1]).repeat(int(self.context_len/2)+ int(self.pred_len/2), 1, 1) # [seq_len, bs, d]
            time_emb_s[:int(self.context_len/2)] = 0. # no time noise embedding for context vars
            time_emb = torch.stack([time_emb_a, time_emb_s], dim=1).reshape(2 * time_emb_a.shape[0], time_emb_a.shape[1], time_emb_a.shape[2])
        
        else:
            assert timesteps.dim() == 1
            time_emb = self.embed_timestep(timesteps)  # [1, bs, d]

        # if 'target_vel' in y.keys() and 'target_vel' in self.cond_mode:
        #     target_vel_emb = self.embed_target_vel(y['target_vel'])
        #     target_vel_emb, target_vel_mask = self.mask_cond_return_mask(target_vel_emb[None], mask_prob=self.cfg.model.target_vel_mask_prob, force_mask=y.get('target_vel_uncond', False))
        #     y['target_vel_mask'] = (target_vel_mask - 1)*-1 # convert to 0 is masked for losses
        #     time_emb += target_vel_emb

        # Build input for prefix completion
        if self.is_prefix_comp:
            x_prefix = None
            x_action = torch.cat([y['prefix_a'], x_action], dim=-1)

            if self.mdm_cfg.prediction_type == self.mdm_cfg.prefix_type:
                if self.prediction_type == 'state':
                    x_robot = torch.cat([y['prefix_s'], x_robot], dim=-1)
                elif self.prediction_type == 'context':
                    x_robot = torch.cat([y['prefix_c'], x_robot], dim=-1)
                else:
                    raise NotImplementedError
            else:
                assert self.mdm_cfg.prefix_type == 'context'
                x_prefix = y['prefix_c'].clone()

        # y['mask_actions'] = y['mask'].clone()           
        # if self.action_pred_len > 0:
        #     # mask out the actions 
        #     y['mask_actions'][:, :, :, self.action_pred_len:] = 0 
        
        
        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            raise NotImplementedError
            if 'text_embed' in y.keys():  # caching option
                enc_text = y['text_embed']
            else:
                enc_text = self.encode_text(y['text'])
            if type(enc_text) == tuple:
                enc_text, text_mask = enc_text
                if text_mask.shape[0] == 1 and bs > 1:  # casting mask for the single-prompt-for-all case
                    text_mask = torch.repeat_interleave(text_mask, bs, dim=0)
            text_emb = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))  # casting mask for the single-prompt-for-all case
            if self.emb_policy == 'add':
                emb = text_emb + time_emb
            else:
                emb = torch.cat([time_emb, text_emb], dim=0)
                text_mask = torch.cat([torch.zeros_like(text_mask[:, 0:1]), text_mask], dim=1)

        if self.cross_attn_conds is not None and len( self.cross_attn_conds)>0:
            ca_args = []
            assert self.emb_policy == 'cat'
            if 'emb' in self.cross_attn_conds:
                ca_args.append(time_emb)
            if 'context' in self.cross_attn_conds:
                
                # if self.training and np.random.rand() < 0.1:
                #     context_mask = torch.ones((bs, 1, 1, int(self.context_len/2)), device=x.device)
                #     masked_history = np.random.randint(1, self.pred_len-1)
                #     context_mask[:, :, :, :masked_history] = 0.
  
                #     y['prefix_c'] *= context_mask                    
                #     y['prefix_a'] *= context_mask

                emb_context = self.emb_ca_context(y['prefix_c'].squeeze(1).permute(2,0,1).to(x_action.device)) # [seq, bs, d]
                emb_actions = self.emb_ca_actions(y['prefix_a'].squeeze(1).permute(2,0,1).to(x_action.device)) # [seq, bs, d]
                
                emb_act_cont = torch.stack([emb_actions, emb_actions], dim=1).reshape(2 * emb_context.shape[0], emb_context.shape[1], emb_context.shape[2])
                emb_act_cont = self.sequence_pos_encoder(emb_act_cont) 
                ca_args.append(emb_act_cont)

            emb = torch.cat(ca_args, dim=0)
            ca_mask = torch.zeros((emb_context.shape[1], emb.shape[0]), device=x_action.device, dtype=torch.bool)

        if 'no_cond' in self.cond_mode or 'target_vel' in self.cond_mode: 
            # unconstrained
            emb = time_emb

        x = self.input_process(x_action, x_robot, x_prefix)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            # xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = x + emb
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]

            mask = None
            if self.attn_type == 'causal':
                mask = self.causal_mask.repeat(bs*self.num_heads, 1, 1).to(x.device)
                assert mask.dtype == torch.bool

            if 'target_vel' in self.cond_mode:
                output = self.seqTransEncoder(src=xseq, cond=y['target_vel'], mask=mask)
            else:
                output = self.seqTransEncoder(xseq, mask=mask)#[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'trans_dec':
  
            if self.emb_trans_dec:
                raise NotImplementedError
                xseq = torch.cat((time_emb, x), axis=0)
            else:
                xseq = x

            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]

            output = self.seqTransDecoder(tgt=xseq, memory=emb, memory_key_padding_mask=ca_mask)  # Rotem's bug fix
      
            if self.emb_trans_dec:
                raise NotImplementedError
                output = output[1:] # [seqlen, bs, d]

        # Extract completed suffix
        if self.is_prefix_comp:
            output = output[self.context_len:]
        
        output_action, output_robot = self.output_process(output)  # [bs, njoints, nfeats, nframes]

        output = torch.cat((output_action, output_robot), dim=-2)

        return output


    def _apply(self, fn):
        super()._apply(fn)



    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    

class KerrasTimestepEmbedder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(timestep_embedding(timesteps=timesteps, dim=self.latent_dim)).unsqueeze(0)



class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, action_input_feats, robot_input_feats, latent_dim):
        super().__init__()
   
        self.action_input_feats = action_input_feats        
        self.robot_input_feats = robot_input_feats

        self.latent_dim = latent_dim

        self.robotEmbedding = nn.Linear(self.robot_input_feats, self.latent_dim)
        self.actionsEmbedding = nn.Linear(self.action_input_feats, self.latent_dim)
        

    def forward(self, x_action, x_robot, x_prefix=None):
        bs, njoints, nfeats, nframes = x_action.shape
        x_action = x_action.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        
        bs, njoints, nfeats, nframes = x_robot.shape

        # frames, bs, features
        x_robot = x_robot.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        x_action = self.actionsEmbedding(x_action)
        x_robot = self.robotEmbedding(x_robot)

        x = torch.stack([x_action, x_robot], dim=1).reshape(2 * nframes, bs, x_robot.size(-1))
  
        return x
    
class MixedInputProcess(nn.Module):
    def __init__(self, 
                 action_input_feats, 
                 robot_input_feats, 
                 prefix_input_feats,
                 latent_dim):
        super().__init__()
   
        self.action_input_feats = action_input_feats        
        self.robot_input_feats = robot_input_feats        
        self.prefix_input_feats = prefix_input_feats

        self.latent_dim = latent_dim

        self.robotEmbedding = nn.Linear(self.robot_input_feats, self.latent_dim)        
        self.prefixEmbedding = nn.Linear(self.prefix_input_feats, self.latent_dim)
        self.actionsEmbedding = nn.Linear(self.action_input_feats, self.latent_dim)
        

    def forward(self, x_action, x_robot, x_prefix):

        # Assume action are already concatonated
        bs, njoints, nfeats, a_nframes = x_action.shape
        x_action = x_action.permute((3, 0, 1, 2)).reshape(a_nframes, bs, njoints*nfeats)
        
        # frames, bs, features
        bs, njoints, nfeats, nframes = x_robot.shape
        x_robot = x_robot.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        # frames, bs, features
        bs, njoints, nfeats, nframes = x_prefix.shape
        x_prefix = x_prefix.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)


        x_action = self.actionsEmbedding(x_action)
        x_robot = self.robotEmbedding(x_robot)        
        x_prefix = self.prefixEmbedding(x_prefix)

        x_mixed = torch.cat((x_prefix, x_robot))

        x = torch.stack([x_action, x_mixed], dim=1).reshape(2 * a_nframes, bs, x_robot.size(-1))
  
        return x


class OutputProcess(nn.Module):
    def __init__(self, action_input_feats, robot_input_feats, latent_dim, njoints):
        super().__init__()
    
        self.action_input_feats = action_input_feats        
        self.robot_input_feats = robot_input_feats

        self.latent_dim = latent_dim
        self.njoints = njoints
 
        self.actionFinal = nn.Linear(self.latent_dim, self.action_input_feats)        
        self.robotFinal = nn.Linear(self.latent_dim, self.robot_input_feats)

   

    def forward(self, output):
        nframes, bs, d = output.shape

        action_frames = output[0::2]
        robot_frames = output[1::2]


        output_action = self.actionFinal(action_frames)  # [seqlen, bs, 150]        
        output_robot = self.robotFinal(robot_frames)  # [seqlen, bs, 150]

        output_action = output_action.reshape(int(nframes/2), bs, self.njoints, self.action_input_feats)
        output_robot = output_robot.reshape(int(nframes/2), bs, self.njoints, self.robot_input_feats)     

        output_action = output_action.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        output_robot = output_robot.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]

        return output_action, output_robot


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output
    
class EmbedTargetLocSingle(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim, num_layers=1):
        super().__init__()
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.target_cond_dim = len(self.extended_goal_joint_names) * 4  # 4 => (x,y,z,is_valid)
        self.latent_dim = latent_dim
        _layers = [nn.Linear(self.target_cond_dim, self.latent_dim)]
        for _ in range(num_layers):
            _layers += [nn.SiLU(), nn.Linear(self.latent_dim, self.latent_dim)]
        self.mlp = nn.Sequential(*_layers)

    def forward(self, input, target_joint_names, target_heading):
        # TODO - generate validity from outside the model
        validity = torch.zeros_like(input)[..., :1]
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            for j in sample_joint_names_w_heading:
                validity[sample_idx, self.extended_goal_joint_names.index(j)] = 1.

        mlp_input = torch.cat([input, validity], dim=-1).view(input.shape[0], -1)
        return self.mlp(mlp_input)


class EmbedTargetLocSplit(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim, num_layers=1):
        super().__init__()
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.target_cond_dim = 4
        self.latent_dim = latent_dim
        self.splited_dim = self.latent_dim // len(self.extended_goal_joint_names)
        assert self.latent_dim % len(self.extended_goal_joint_names) == 0
        self.mini_mlps = nn.ModuleList()
        for _ in self.extended_goal_joint_names:
            _layers = [nn.Linear(self.target_cond_dim, self.splited_dim)]
            for _ in range(num_layers):
                _layers += [nn.SiLU(), nn.Linear(self.splited_dim, self.splited_dim)]
            self.mini_mlps.append(nn.Sequential(*_layers))

    def forward(self, input, target_joint_names, target_heading):
        # TODO - generate validity from outside the model
        validity = torch.zeros_like(input)[..., :1]
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            for j in sample_joint_names_w_heading:
                validity[sample_idx, self.extended_goal_joint_names.index(j)] = 1.

        mlp_input = torch.cat([input, validity], dim=-1)
        mlp_splits = [self.mini_mlps[i](mlp_input[:, i]) for i in range(mlp_input.shape[1])] 
        return torch.cat(mlp_splits, dim=-1)
  
class EmbedTargetLocMulti(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim):
        super().__init__()
        
        # todo: use a tensor of weight per joint, and another one for biases, then apply a selection in one go like we to for actions
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.extended_goal_joint_idx = {joint_name: idx for idx, joint_name in enumerate(self.extended_goal_joint_names)}
        self.n_extended_goal_joints = len(self.extended_goal_joint_names)
        self.target_loc_emb = nn.ParameterDict({joint_name: 
            nn.Sequential(
                nn.Linear(3, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim)) 
            for joint_name in self.extended_goal_joint_names})  # todo: check if 3 works for heading and traj
            # nn.Linear(3, latent_dim) for joint_name in self.extended_goal_joint_names})  # todo: check if 3 works for heading and traj
        self.target_all_loc_emb = WeightedSum(self.n_extended_goal_joints) # nn.Linear(self.n_extended_goal_joints, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, input, target_joint_names, target_heading):
        output = torch.zeros((input.shape[0], self.latent_dim), dtype=input.dtype, device=input.device)
        
        # Iterate over the batch and apply the appropriate filter for each joint
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            output_one_sample = torch.zeros((self.n_extended_goal_joints, self.latent_dim), dtype=input.dtype, device=input.device)
            for joint_name in sample_joint_names_w_heading:
                layer = self.target_loc_emb[joint_name]
                output_one_sample[self.extended_goal_joint_idx[joint_name]] = layer(input[sample_idx, self.extended_goal_joint_idx[joint_name]])  
            output[sample_idx] = self.target_all_loc_emb(output_one_sample)
            # print(torch.where(output_one_sample.sum(axis=1)!=0)[0].cpu().numpy())
               
        return output
