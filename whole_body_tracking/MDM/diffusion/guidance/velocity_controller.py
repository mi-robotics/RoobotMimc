import torch

class VelocityGuidance:
    """
    A callable class that implements the cond_fn for joystick steering guidance.
    It computes the gradient of the cost function with respect to the input trajectory.
    """

    def __init__(self, mean, std, guidance_scale=0, vel_indices=[0,1], action_dim=29, device='cuda'):
        """
        Initializes the guidance function.

        Args:
            guidance_scale (float): The strength of the guidance. A larger value
                will push the model's output more strongly towards the target velocity.
            vel_indices (list of int): A list of two  integers specifying the indices
                in the state vector corresponding to the x and y velocities.
                For example, if state is [pos_x, pos_y, vel_x, vel_y], this would be [2, 3].
        """
        self.guidance_scale = guidance_scale
        self.vel_indices = vel_indices
        self.action_dim = action_dim
        self.device = device
        self.mean = torch.from_numpy(mean).to(self.device)[None, None,:, None]
        self.std = torch.from_numpy(std).to(self.device)[None, None,:, None]

        if len(self.vel_indices) != 2:
            raise ValueError("vel_indices must contain exactly two indices for x and y velocity.")
        
    def __call__(self, x, t, **model_kwargs):
        """
        This method is the actual `cond_fn`. It will be called by `condition_mean`.

        Returns:
            torch.Tensor: The gradient to be used for conditioning the mean.
        """
        # If no joystick input is provided, no guidance is applied.
        if model_kwargs is None or "target_vel_guidance" not in model_kwargs['y']:
            raise ValueError('Missing guidance param: target_vel_guidance')
        
        # We use torch.enable_grad() to compute gradients.
        with torch.enable_grad():
            # Make a copy of x that requires a gradient.
            
            x_in = x.detach().clone().requires_grad_(True) # [bs, 1, features, frames]
            prefix_s = x_in[:, :, self.action_dim:, 0].unsqueeze(-1)
            predicted_state = x_in[:, :, self.action_dim:, : ]
            predicted_state = torch.cat((prefix_s, predicted_state), dim=-1)
            predicted_state = (predicted_state * self.std ) + self.mean

            def extract_velocities(pos):
            
                delta_pos = pos[:, :, :, 1:] - pos[:, :, :, :-1]
                delta_pos_global = torch.cumsum(delta_pos, dim=-1)
              
                vel = delta_pos_global*50/16 #fps
                return vel
            
            
            predicted_velocities = extract_velocities(predicted_state[:, :, self.vel_indices, :])
     
            target_velocities = model_kwargs['y']["target_vel_guidance"].to(x_in.device) # [bs, 3]
            if target_velocities.dim() == 2:
                target_velocities = target_velocities.unsqueeze(1).unsqueeze(-1) # [bs, 1, 3, 1]

     
            diff = predicted_velocities - target_velocities

            # Squared L2 norm: ||...||^2. We sum the squares over the velocity components (last dim).
            sq_l2_norm = (diff ** 2).sum(dim=-2)

            # Sum over the time horizon t' and apply the 1/2 factor.
            # We also sum over the batch dimension to get a single scalar loss.
            cost = 0.5 * sq_l2_norm.sum()

            # --- Compute the Gradient ---
            # Compute the gradient of the cost with respect to our input x_in.
            # This calculates ∇_x G_js(x)
            grad = torch.autograd.grad(cost, x_in)[0]  #/ (t+1)

        
        return -self.guidance_scale * grad

#     def __call__(self, x, t, **model_kwargs):
#         """
#         This method is the actual `cond_fn`. It will be called by `condition_mean`.

#         Returns:
#             torch.Tensor: The gradient to be used for conditioning the mean.
#         """
#         # If no joystick input is provided, no guidance is applied.
#         if model_kwargs is None or "target_vel_guidance" not in model_kwargs['y']:
#             raise ValueError('Missing guidance param: target_vel_guidance')

#         # We use torch.enable_grad() to compute gradients.
#         with torch.enable_grad():
#             # Make a copy of x that requires a gradient.
            
#             x_in = x.detach().clone().requires_grad_(True) # [bs, 1, features, frames]
#             predicted_state = x_in[:, :, self.action_dim:, : ]
#             predicted_state = (predicted_state * self.std ) + self.mean
#             predicted_velocities = predicted_state[:, :, self.vel_indices, : ] #[ bs, 1, 3. frames]

#             # 2. Get the target velocity (joystick input) g_v from model_kwargs.
#             # Expected shape: (batch_size, 2)
#             target_velocities = model_kwargs['y']["target_vel_guidance"].to(x_in.device) # [bs, 3]

#             # Reshape target_velocities from (B, 3) to (B, 1, 3, 1) to allow
#             # broadcasting across the time horizon dimension.
#             if target_velocities.dim() == 2:
#                 target_velocities = target_velocities.unsqueeze(1).unsqueeze(-1) # [bs, 1, 3, 1]

#             diff = predicted_velocities - target_velocities
#             # print(diff)

#             # Squared L2 norm: ||...||^2. We sum the squares over the velocity components (last dim).
#             sq_l2_norm = (diff ** 2).sum(dim=-2)

#             # Sum over the time horizon t' and apply the 1/2 factor.
#             # We also sum over the batch dimension to get a single scalar loss.
#             cost = 0.5 * sq_l2_norm.sum()

#             # --- Compute the Gradient ---
#             # Compute the gradient of the cost with respect to our input x_in.
#             # This calculates ∇_x G_js(x)
#             grad = torch.autograd.grad(cost, x_in)[0]  #/ (t+1)

#             # grad[:, :, self.action_dim:, : ] /= self.std

#             # print(grad[:, :, self.action_dim:, : ][:, :, self.vel_indices, : ])
# #


#         # 2. --- MANUAL PER-BATCH GRADIENT CLIPPING LOGIC ---
#         # max_norm = 0.5  # Set your desired maximum norm here
#         # grad_norms = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
#         # clip_coefs = max_norm / (grad_norms + 1e-6)
#         # clip_coefs.clamp_(max=1.0)
#         # grad.mul_(clip_coefs)
        
#         # print(f"[cond_fn] Per-batch norms BEFORE clip: {grad_norms.flatten().cpu().numpy()}")
#         # new_norms = torch.norm(grad, p=2, dim=(1, 2, 3)) # No keepdim needed for printing
#         # print(f"[cond_fn] Per-batch norms AFTER clip:  {new_norms.flatten().cpu().numpy()}")
#         # print(grad[:, :, self.action_dim:, : ][:, :, self.vel_indices, : ])
     
#         return -self.guidance_scale * grad