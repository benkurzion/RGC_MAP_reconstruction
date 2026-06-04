import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, Optional

from reconstruction_alg.hqs_alg import BatchParallel_HQS_X_Problem
from convex_solver_base.optim_base import BatchParallelUnconstrainedProblem


class BatchLinearMeasurementProxProblem(BatchParallelUnconstrainedProblem, BatchParallel_HQS_X_Problem):
    """
    X-step for HQS with a Gaussian linear measurement likelihood:

        p(y | x) ∝ exp(-½ ‖y - Mx‖²)

    Minimizes per image in the batch:

        (½) ‖y - Mx‖²  +  (ρ/2) ‖x - z‖²

    where M is a fixed (n_measurements × n_pixels) matrix, y are the observed
    measurements for each image, and z is the HQS auxiliary variable.
    """

    def __init__(self,
                 batch: int,
                 measurement_matrix: np.ndarray,
                 image_shape: Tuple[int, int],
                 rho: float,
                 dtype: torch.dtype = torch.float32):
        """
        :param batch: batch size
        :param measurement_matrix: shape (n_measurements, height*width)
        :param image_shape: (height, width)
        :param rho: initial HQS coupling parameter ρ
        """
        super().__init__()

        self.batch_size = batch
        self.rho = rho
        self.height, self.width = image_shape
        self.n_pixels = self.height * self.width
        n_measurements = measurement_matrix.shape[0]

        assert measurement_matrix.shape == (n_measurements, self.n_pixels), \
            f'measurement_matrix must have shape (n_measurements, {self.n_pixels})'

        # shape (n_measurements, n_pixels)
        self.register_buffer('measurement_matrix',
                             torch.tensor(measurement_matrix, dtype=dtype))

        # observed measurements y, shape (batch, n_measurements); updated per batch
        self.register_buffer('observations',
                             torch.zeros((batch, n_measurements), dtype=dtype))

        # HQS auxiliary variable z, shape (batch, height, width)
        self.register_buffer('z_const_tensor',
                             torch.zeros((batch, self.height, self.width), dtype=dtype))

        # optimization variable: image, shape (batch, height, width)
        self.image = nn.Parameter(
            torch.empty((batch, self.height, self.width), dtype=dtype))
        nn.init.normal_(self.image, mean=0.0, std=1.0)

    def set_observations(self, observations: torch.Tensor) -> None:
        """Update the observed measurements y for the current batch.

        :param observations: shape (batch, n_measurements)
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

        # shape (batch, n_pixels)
        batched_image_flat = batched_image.reshape(self.batch_size, -1)

        # Mx: (batch, n_measurements)
        predicted = batched_image_flat @ self.measurement_matrix.T

        # residual: (batch, n_measurements)
        residual = self.observations - predicted

        # data fidelity per image: (batch,)
        data_loss = 0.5 * torch.sum(residual * residual, dim=1)

        # HQS prox penalty: (batch,)
        prox_diff = batched_image - self.z_const_tensor
        prox_loss = 0.5 * self.rho * torch.sum(prox_diff * prox_diff, dim=(1, 2))

        return data_loss + prox_loss

    def _packed_gradients_only(self, packed_variables: torch.Tensor, **kwargs) -> torch.Tensor:
        """Analytic gradient — avoids autograd overhead for this quadratic problem.

        packed_variables: (batch, n_pixels)
        returns:          (batch, n_pixels)
        """
        # M @ x: (batch, n_measurements)
        Mx = packed_variables @ self.measurement_matrix.T

        # M^T (Mx - y): (batch, n_pixels)
        data_grad = (Mx - self.observations) @ self.measurement_matrix

        # ρ (x - z): (batch, n_pixels)
        z_flat = self.z_const_tensor.reshape(self.batch_size, -1)
        prox_grad = self.rho * (packed_variables - z_flat)

        return data_grad + prox_grad

    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        return args[0]
