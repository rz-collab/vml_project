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
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)

        if PSO_params is None:
            self.PSO_params = {
                "batch_size": 0,  # Adjusted automatically by adjust method.
                "num_iters": 5,
                "n_particles": 2,
                "dim": 6,
                "c1": 2.0,
                "c2": 2.0,
                "w": 1.0,
                "pos_constraint": torch.zeros((2, 6)),  # Adjusted automatically by adjust method.
                "vel_constraint": torch.zeros((2, 6)),  # Adjusted automatically by adjust method.
            }
        else:
            self.PSO_params = PSO_params

    def _compute_particle_loss(
        self, shadow_polygons: torch.Tensor, camera_imgs, other_model_inputs: List[torch.Tensor], y: torch.Tensor
    ):
        """
        Returns the negative L1 difference as loss between unperturbed output and perturbed output.
        shadow_polygons (particles) is a batch of partcles of size N * n_particles
        other_model_inputs is [pclouds, intrinsic, extrinsics], assumed to be batch size N * n_particles.
        camera_imgs is only batch size N
        y is assumed to already be duplicated to be batch size N * n_particles as well.
        Returns a (N * n_particles, ) loss tensor
        """
        # Assume camera images assumed already in LAB space of shape (N, T, 3, H, W)
        X_lab = camera_imgs
        N, T, _, H, W = X_lab.shape
        n_particles = shadow_polygons.shape[0] // N

        # Get binary masks (N * n_particles, H, W) from these polygons (N * n_particles, dim=6)
        shadow_masks = self.get_shadow_masks(shadow_polygons, H, W)

        # Apply shadow masks on a copy of camera images
        Xp = X_lab.clone().repeat(n_particles, 1, 1, 1, 1)
        Xp = self.apply_shadow_masks(shadow_masks, Xp)

        # Compute model's output under perturbed input Xp (N * n_particles, ...)
        y_p, _, _ = self.model(Xp, *other_model_inputs)

        # Compute negative L1 loss between unperturbed output and perturbed output
        loss = torch.abs(y - y_p)  # Element-wise L1 loss
        batch_neg_loss = -loss.mean(dim=(1, 2, 3))  # Per-batch loss
        return batch_neg_loss

    def apply_shadow_masks(self, shadow_masks: torch.Tensor, X_lab: torch.Tensor, L_factor: float = 0.43):
        """
        Apply shadow masks to a batch of LAB images by modifying their luminance values in the LAB color space.

        Args:
            shadow_masks (torch.Tensor): A tensor of shape (N, H_i, W_i), where N is the number of shadow masks.
                                         Each mask is a binary map with 1 indicating shadow regions.
            X_lab (torch.Tensor): A tensor of shape (N, T, 3, H_i, W_i) representing a batch of sequences of length T of LAB images.
            L_factor (float, optional): A scaling factor for the L (luminance) channel in the LAB color space
                                        for regions under shadow. Default is 0.43.
        Returns:
            Xp (torch.Tensor): A tensor of same shape as X_lab with LAB images perturbed by shadow.
        """
        X_lab = X_lab.clone()
        N, T, _, H, W = X_lab.shape

        # Expand shadow masks for sequences of RGB images
        shadow_masks = shadow_masks.unsqueeze(1).expand(-1, T, -1, -1)

        # Modify L value (first channel) based on shadow_mask and L_factor
        L_channel = X_lab[:, :, 0]
        L_channel = L_channel + shadow_masks * (L_factor * L_channel - L_channel)
        X_lab[:, :, 0] = L_channel

        # Convert back to RGB color space
        Xp = self._change_lab_to_rgb(X_lab)
        return Xp

    def _adjust_PSO_constraints(self, batch_size: int, img_h: int, img_w: int):
        self.PSO_params["batch_size"] = batch_size
        vertex_pos_constraint = torch.tensor(
            [
                [0.0, 0.0],  # min for (x,y)
                [img_h, img_w],  # max for (x,y).
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

    def _change_rgb_to_lab(self, X: torch.Tensor) -> torch.Tensor:
        N, T, C, H, W = X.shape
        X = X.view(-1, C, H, W)
        X = kornia.color.rgb_to_lab(X)
        X = X.view(N, T, C, H, W)
        return X

    def _change_lab_to_rgb(self, X: torch.Tensor) -> torch.Tensor:
        N, T, C, H, W = X.shape
        X = X.view(-1, C, H, W)
        X = kornia.color.lab_to_rgb(X, clip=True)
        X = X.view(N, T, C, H, W)
        return X

    def generate_attack(self, model_inputs: List[torch.Tensor]):
        """
        model_inputs should be a list of [color_img=X, pcloud, intrinsics, extrinsics]
        Generate adversarial attack on X of shape (batch_size, seq_length, 3, H, W)
        """

        # Transfer inputs to the same device
        for i in range(len(model_inputs)):
            model_inputs[i] = model_inputs[i].to(device=self.device)

        with torch.no_grad():
            # Compute model's unperturbed output y
            X = model_inputs[0]
            N, T, C, H, W = X.shape
            y, _, _ = self.model(*model_inputs)

            # Convert RGB images to LAB color space. Assuming RGB images are normalized to range [0,1]
            X_lab = self._change_rgb_to_lab(X.clone())

            # Create Particle Swarm Optimization optimizer
            self._adjust_PSO_constraints(len(X_lab), H, W)
            pso_optimizer = PSO(**self.PSO_params)
            n_particles = self.PSO_params["n_particles"]

            # The particle loss functions works with batch size N * n_particles.
            # Reshape `model_inputs`` and `y`` to match batch size
            other_model_inputs = model_inputs[1:]
            for i in range(len(other_model_inputs)):
                shape = other_model_inputs[i].shape
                other_model_inputs[i] = (
                    other_model_inputs[i].unsqueeze(0).expand(n_particles, *shape).reshape(N * n_particles, *shape[1:])
                )
            y = y.unsqueeze(0).expand(n_particles, *y.shape).reshape(N * n_particles, *y.shape[1:])

            # Create particle loss function
            particle_neg_loss_fn = functools.partial(
                self._compute_particle_loss, camera_imgs=X_lab, other_model_inputs=other_model_inputs, y=y
            )

            # Use PSO to search for best shadow placement that maximizes loss for the batch of inputs N
            batch_best_particle_neg_loss, batch_best_particle = pso_optimizer.optimize(particle_neg_loss_fn)

            # Get the batch of perturbed images Xp
            shadow_masks = self.get_shadow_masks(batch_best_particle, H, W)
            Xp = self.apply_shadow_masks(shadow_masks, X_lab)
            return -batch_best_particle_neg_loss, Xp

    def _check_points_inside_triangles(self, vertices_batch, points_batch):
        """
        Check whether a batch of points are inside the corresponding batch of triangles.

        Args:
            vertices_batch: Tensor of shape (N, 6), representing the vertices of N triangles.
                            Each row contains [x1, y1, x2, y2, x3, y3].
            points_batch: Tensor of shape (N, M, 2), representing M points for each of the N triangles.
                        Each row contains the points to be checked against the corresponding triangle.

        Returns:
            A boolean tensor of shape (N, M) where each entry indicates if the corresponding point
            is inside the triangle.
        """
        N = vertices_batch.shape[0]
        M = points_batch.shape[1]

        # Reshape vertices to (N, 3, 2) for easier manipulation
        vertices = vertices_batch.view(N, 3, 2)

        # Add the third dimension (z=0) for 3D cross product calculation
        vertices_3d = torch.cat([vertices, torch.zeros(N, 3, 1, device=self.device)], dim=2)  # Shape: (N, 3, 3)
        points_3d = torch.cat([points_batch, torch.zeros(N, M, 1, device=self.device)], dim=2)  # Shape: (N, M, 3)

        # Extract vertices a, b, c
        a, b, c = vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2]  # Each: (N, 3)

        # Calculate cross products for each point with respect to each edge
        cross1 = torch.cross(b.unsqueeze(1) - a.unsqueeze(1), points_3d - a.unsqueeze(1), dim=2)[:, :, 2] <= 0
        cross2 = torch.cross(c.unsqueeze(1) - b.unsqueeze(1), points_3d - b.unsqueeze(1), dim=2)[:, :, 2] <= 0
        cross3 = torch.cross(a.unsqueeze(1) - c.unsqueeze(1), points_3d - c.unsqueeze(1), dim=2)[:, :, 2] <= 0

        # Combine conditions to check if points are inside the triangles
        return ~(cross1 ^ cross2) & ~(cross2 ^ cross3)

    def get_shadow_masks(self, shadow_polygons: torch.Tensor, img_h: int, img_w: int):
        """
        Get binary shadow masks for a batch of shadow polygons (shadow_polygons).

        Args:
            shadow_polygons: Tensor of shape (N, 6), where each row contains the flattened
                            vertices of a triangle (x1, y1, x2, y2, x3, y3).
            img_h: Height of the image.
            img_w: Width of the image.

        Returns:
            A tensor of shape (N, img_h, img_w), where each (img_h, img_w) mask corresponds to a shadow polygon.
        """
        N = shadow_polygons.shape[0]  # Number of polygons

        # Generate Cartesian product of pixel coordinates
        rows = torch.arange(img_h)
        cols = torch.arange(img_w)
        img_mask_points = torch.cartesian_prod(rows, cols).float().to(self.device)  # Shape: (img_h * img_w, 2)

        # Reshape points for batch processing
        img_mask_points_batch = img_mask_points.unsqueeze(0).expand(N, -1, -1)  # Shape: (N, img_h * img_w, 2)

        # Check points inside each polygon using the batch function
        points_inside = self._check_points_inside_triangles(
            shadow_polygons, img_mask_points_batch
        )  # Shape: (N, img_h * img_w)

        # Create the masks using points_inside
        masks_flat = torch.zeros((N, img_h * img_w), dtype=torch.uint8).to(self.device)
        masks_flat[points_inside] = 1  # Assign 1 to points inside the polygons

        # Reshape masks to (N, img_h, img_w)
        masks = masks_flat.view(N, img_h, img_w)

        # Erase mask points in the top half of images, so shadow mask is only on the terrain.
        masks[:, 0 : img_h // 2, :] = 0

        return masks
