from adversarial_attacks.PSO import PSO
import torch
from torch import nn
from typing import Optional, Dict, List
import functools
import kornia


class ShadowAttack:
    def __init__(self, model: nn.Module, PSO_params: Optional[Dict] = None):
        """
        Args:
            model (nn.Module): Model that accepts a sequence of images of shape (seq_length, 3 or more, H, W), where the first three channels are RGB images.
        """
        self.model = model
        self.loss_fn = nn.L1Loss()

        if PSO_params is None:
            self.PSO_params = {
                "num_iters": 10,
                "n_particles": 10,
                "dim": 6,
                "c1": 1.5,
                "c2": 1.5,
                "w": 0.5,
                "pos_constraint": torch.zeros((2, 6)),  # Adjusted automatically by adjust method.
                "vel_constraint": torch.zeros((2, 6)),  # Adjusted automatically by adjust method.
            }
        else:
            self.PSO_params = PSO_params

    def _compute_particle_loss(self, shadow_polygon: torch.Tensor, X: torch.Tensor, y: torch.Tensor):
        """
        Compute with the input perturbed by shadow represented by a triangular `shadow_polygon`.
        Assumes X of shape (seq_length, 3 or more, H_i, W_i) and y of shape (*, H, W)
        """
        # Obtain a binary shadow mask based on the shadow polygon.
        img_h, img_w = X.shape[-2:]
        shadow_mask = self.get_shadow_mask(shadow_polygon, img_h, img_w)
        # Apply the same shadow mask on the sequence of RGB imgs.
        X_p = self.apply_shadow_mask(shadow_mask, X)
        # Compute model's output under adversarial input
        y_p = self.model(X_p.unsqueeze(0))
        # Compute L1 loss between unperturbed output and perturbed output
        loss = self.loss_fn(y, y_p)
        return loss

    def get_shadow_mask(self, shadow_polygon: torch.Tensor, img_h: int, img_w: int):
        """
        Get binary shadow mask, with 1 indicating shadow,
        corresponding to the shadow polygon to Mask in the given image sizes.
        Assumes shadow_polygon is a flattened shape (1,6)
        """
        img_mask = torch.zeros((img_h, img_w), dtype=torch.uint8)
        # Generate Cartesian product of pixel coordinates
        rows = torch.arange(img_h)
        cols = torch.arange(img_w)
        img_mask_points = torch.cartesian_prod(rows, cols)  # Shape: (img_h * img_w, 2)

        # Get points inside triangle
        points_inside = self._check_points_inside_triangle(shadow_polygon, img_mask_points)

        # Assign 1 to points inside the triangle
        img_mask[img_mask_points[points_inside, 0], img_mask_points[points_inside, 1]] = 1
        return img_mask

    def apply_shadow_mask(self, shadow_mask: torch.Tensor, X: torch.Tensor, shadow_coefficient: float = 0.43):
        """Apply shadow mask on top of model's input X.
        Assumes shadow_mask of shape (H_i, W_i) and
        X of shape (seq_length, 3 or more, H_i, W_i)
        """
        # Convert the RGB images to LAB color space
        X_lab = X.detach().clone().double()
        X_lab[:, :3] /= 255.0
        X_lab[:, :3] = kornia.color.rgb_to_lab(X_lab[:, :3])
        # Modify L value based on shadow_mask and shadow_coefficient
        X_lab[:, 0][shadow_mask.expand(size=(len(X_lab), -1, -1)) == 1] *= shadow_coefficient
        # Clip L value
        X_lab[:, 0] = torch.clip(X_lab[:, 0], min=0.0, max=100.0)
        # Convert back to RGB color space
        X_lab[:, :3] = kornia.color.lab_to_rgb(image=X_lab[:, :3], clip=True)
        X_lab[:, :3] *= 255.0
        return X_lab

    def _check_points_inside_triangle(self, vertices, points):
        """Check whether batch of points are inside the defined triangle"""
        vertices = vertices.view(3, 2)
        # Add the third dimension
        n = len(points)
        vertices_3d = torch.cat([vertices, torch.zeros(len(vertices), 1)], dim=1)
        points_3d = torch.cat([points, torch.zeros(n, 1)], dim=1)
        # Check by cross products
        a, b, c = vertices_3d
        cross1 = torch.cross(torch.broadcast_to(b - a, (n, 3)), points_3d - a, dim=1)[:, 2] <= 0
        cross2 = torch.cross(torch.broadcast_to(c - b, (n, 3)), points_3d - b, dim=1)[:, 2] <= 0
        cross3 = torch.cross(torch.broadcast_to(a - c, (n, 3)), points_3d - c, dim=1)[:, 2] <= 0
        return ~(cross1 ^ cross2) & ~(cross2 ^ cross3)

    def _adjust_PSO_constraints(self, img_h: int, img_w: int) -> Dict:
        vertex_pos_constraint = torch.tensor(
            [
                [0.0, 0.0],  # min for (x,y)
                [img_h, img_w],  # max for (x,y)
            ]
        )
        vertex_vel_constraint = torch.tensor(
            [
                [-img_h / 10, -img_w / 10],
                [img_h / 10, img_w / 10],
            ]
        )
        pos_constraint = vertex_pos_constraint.repeat(1, 3)
        vel_constraint = vertex_vel_constraint.repeat(1, 3)
        self.PSO_params["pos_constraint"] = pos_constraint
        self.PSO_params["vel_constraint"] = vel_constraint

    def generate_attack(self, X):
        """
        X (Tensor of shape (seq_length, 3 or more, Hi, Wi)): Sequence of RGB (first three channels) and Depth (last channel) images.
        Return Xp of same shape but perturbed
        """
        with torch.no_grad():
            # Compute model's unperturbed output y and store it as reference.
            n, c, h, w = X.shape
            y = self.model(X.unsqueeze(0))

            # Use Particle Swarm Optimization to search for perturbed images X_p that maximizes L2 loss
            self._adjust_PSO_constraints(h, w)
            pso_optimizer = PSO(**self.PSO_params)
            particle_loss_fn = functools.partial(self._compute_particle_loss, X=X, y=y)  # Need to add the negative.
            particle_neg_loss_fn = lambda p: -particle_loss_fn(p)
            best_particle_neg_loss, best_particle = pso_optimizer.optimize(particle_neg_loss_fn)

            # Get the perturbed images Xp
            shadow_mask = self.get_shadow_mask(best_particle, h, w)
            Xp = self.apply_shadow_mask(shadow_mask, X)
            return -best_particle_neg_loss, Xp
