import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, Optional

from reconstruction_alg.hqs_alg import BatchParallel_HQS_X_Problem
from convex_solver_base.optim_base import BatchParallelUnconstrainedProblem, BatchParallelGNProblem
from autoencoder.autoencoder import Neurips2017_Encoder


class BatchEncoderMeasurementProxProblem(BatchParallelUnconstrainedProblem,
                                         BatchParallel_HQS_X_Problem,
                                         BatchParallelGNProblem):
    """
    X-step for HQS with a frozen CNN encoder as the measurement operator:

        p(y | x) ∝ exp(-½ ‖y - E(x)‖²_F)

    Minimizes per image in the batch:

        (½) ‖y - E(x)‖²_F  +  (ρ/2) ‖x - z‖²

    where E is a frozen Neurips2017_Encoder, y are the observed encoder outputs
    for each image (flattened), and z is the HQS auxiliary variable.

    Gradients w.r.t. the image are computed via autograd through the encoder.
    The encoder weights are never updated.
    """

    def __init__(self,
                 batch: int,
                 encoder: Neurips2017_Encoder,
                 image_shape: Tuple[int, int],
                 rho: float,
                 obs_scale: float = 1.0,
                 dtype: torch.dtype = torch.float32):
        """
        :param batch: batch size
        :param encoder: Neurips2017_Encoder already in eval() mode with frozen weights
        :param image_shape: (height, width)
        :param rho: initial HQS coupling parameter ρ
        """
        super().__init__()

        self.batch_size = batch
        self.rho = rho
        self.obs_scale = obs_scale
        self.height, self.width = image_shape
        self.n_pixels = self.height * self.width

        for param in encoder.parameters():
            param.requires_grad_(False)
        self.encoder = encoder

        # determine flattened encoder output size with a dummy forward pass
        with torch.no_grad():
            enc_device = next(encoder.parameters()).device
            dummy = torch.zeros((1, self.height, self.width), dtype=dtype, device=enc_device)
            self.enc_out_size = self.encoder(dummy).numel()

        # observed measurements y (flattened encoder outputs), shape (batch, enc_out_size)
        self.register_buffer('observations',
                             torch.zeros((batch, self.enc_out_size), dtype=dtype))

        # HQS auxiliary variable z, shape (batch, height, width)
        self.register_buffer('z_const_tensor',
                             torch.zeros((batch, self.height, self.width), dtype=dtype))

        # optimization variable: image, shape (batch, height, width)
        self.image = nn.Parameter(
            torch.empty((batch, self.height, self.width), dtype=dtype))
        nn.init.normal_(self.image, mean=0.0, std=1.0)

    def set_observations(self, observations: torch.Tensor) -> None:
        """Update the observed encoder outputs y for the current batch.

        :param observations: shape (batch, enc_out_size) — flattened E(x_gt) per image
        """
        self.observations.data[:] = observations.data[:]

    def assign_z(self, z: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = z.data[:]

    def reinitialize_variables(self,
                               initialized_z_const: Optional[torch.Tensor] = None) -> None:
        if initialized_z_const is None:
            nn.init.normal_(self.z_const_tensor, mean=0.0, std=1.0)
        else:
            self.z_const_tensor.data[:] = initialized_z_const.data[:]
        nn.init.normal_(self.image, mean=0.0, std=1.0)

    def get_reconstructed_image(self) -> torch.Tensor:
        return self.image.detach()

    @property
    def n_problems(self) -> int:
        return self.batch_size

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        # shape (batch, height, width)
        batched_image = args[0]

        # encoder forward; autograd tracks grad w.r.t. batched_image through here
        # shape (batch, enc_out_size)
        encoded = self.encoder(batched_image).reshape(self.batch_size, -1) / self.obs_scale

        # residual: (batch, enc_out_size)  (observations already normalized by obs_scale)
        residual = self.observations - encoded

        # data fidelity per image: (batch,)
        data_loss = 0.5 * torch.sum(residual * residual, dim=1)

        # HQS prox penalty: (batch,)
        prox_diff = batched_image - self.z_const_tensor
        prox_loss = 0.5 * self.rho * torch.sum(prox_diff * prox_diff, dim=(1, 2))

        return data_loss + prox_loss

    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        # prior is in image space, so A = I
        return args[0]

    def _gn_matvec(self, packed_vars: torch.Tensor, v: torch.Tensor, lambda_lm: float) -> torch.Tensor:
        """Compute (J_E(x)^T J_E(x) + (rho + lambda_lm) I) v for each image.

        Uses one forward-mode JVP pass to get J_E v, then one backward VJP pass
        to get J_E^T (J_E v). Never forms the full Jacobian matrix.

        :param packed_vars: shape (n, n_pixels) — current linearization points
        :param v: shape (n, n_pixels) — CG search direction vectors
        :param lambda_lm: LM damping; total diagonal = self.rho + lambda_lm
        :return: shape (n, n_pixels)
        """
        n = packed_vars.shape[0]
        H, W = self.height, self.width
        damping = self.rho + lambda_lm
        result = torch.empty_like(v)

        for i in range(n):
            xi = packed_vars[i].reshape(1, H, W).detach()
            vi = v[i].reshape(1, H, W).detach()

            # JVP: compute J_E(xi) @ vi via forward-mode AD
            _, Jvi = torch.autograd.functional.jvp(
                lambda x: self.encoder(x).reshape(-1),
                (xi,),
                (vi,),
                create_graph=False,
                strict=False,
            )

            # VJP: compute J_E(xi)^T @ Jvi via standard backward pass
            # effective Jacobian is J_E / obs_scale, so GN term is J_E^T J_E / obs_scale^2
            xi_rev = xi.detach().requires_grad_(True)
            with torch.enable_grad():
                enc_out = self.encoder(xi_rev).reshape(-1)
                JT_Jvi = torch.autograd.grad(enc_out, xi_rev, grad_outputs=Jvi.detach())[0]

            result[i] = JT_Jvi.reshape(-1).detach() / (self.obs_scale ** 2) + damping * v[i]

        return result
