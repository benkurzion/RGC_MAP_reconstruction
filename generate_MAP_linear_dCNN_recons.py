import pickle
from typing import Tuple
import argparse

import tqdm
import numpy as np
import torch

import denoisers.denoiser_wrappers as denoiser_wrappers

from reconstruction_alg.linear_inverse_alg import BatchLinearMeasurementProxProblem
from reconstruction_alg.hqs_alg import (
    BatchParallel_HQS_XGenerator,
    BatchParallel_DirectSolve_HQS_ZGenerator,
    BatchParallel_UnblindDenoiserPrior_HQS_ZProb,
    scheduled_rho_fixed_lambda_single_hqs_solve,
)
from hyperparameters.hyperparameters import HQSHyperparameters, make_hqs_schedule


def generate_onebatch_linear_hqs_reconstruction(
        x_prob: BatchLinearMeasurementProxProblem,
        z_prob: BatchParallel_UnblindDenoiserPrior_HQS_ZProb,
        image_shape: Tuple[int, int],
        batch_observations: torch.Tensor,
        reconstruction_hyperparams: HQSHyperparameters,
        initialize_noise_level: float = 1e-3) -> torch.Tensor:
    """Run one batch of HQS with a linear measurement likelihood and dCNN prior.

    :param x_prob: linear measurement X-problem
    :param z_prob: DRUNet denoiser Z-problem
    :param image_shape: (height, width)
    :param batch_observations: shape (batch, n_measurements) — observed y = Mx + noise
    :param reconstruction_hyperparams: HQS schedule / prior weight / max_iter
    :param initialize_noise_level: std dev of Gaussian noise used to initialize z
    :return: reconstructed images, shape (batch, height, width)
    """
    height, width = image_shape
    batch_size = batch_observations.shape[0]

    schedule_rho = make_hqs_schedule(reconstruction_hyperparams)
    prior_weight = reconstruction_hyperparams.prior_weight
    max_iter = reconstruction_hyperparams.max_iter

    x_solver_iter = BatchParallel_HQS_XGenerator(first_niter=300, subsequent_niter=300)
    z_solver_iter = BatchParallel_DirectSolve_HQS_ZGenerator()

    # z(1) initialized as random Gaussian noise
    # since we don't have a signal to start with like in the original paper
    initialize_z_tensor = torch.randn((batch_size, height, width),
                                      dtype=torch.float32,
                                      device=batch_observations.device) * initialize_noise_level

    x_prob.reinitialize_variables(initialized_z_const=initialize_z_tensor)
    x_prob.set_observations(batch_observations)
    z_prob.reinitialize_variables()

    scheduled_rho_fixed_lambda_single_hqs_solve(
        x_prob,
        iter(x_solver_iter),
        z_prob,
        iter(z_solver_iter),
        iter(schedule_rho),
        prior_weight,
        max_iter,
        verbose=False,
        save_intermediates=False,
    )

    return x_prob.get_reconstructed_image()


def batch_parallel_generate_linear_hqs_reconstructions(
        observations: np.ndarray,
        measurement_matrix: np.ndarray,
        image_shape: Tuple[int, int],
        reconstruction_hyperparams: HQSHyperparameters,
        max_batch_size: int,
        device: torch.device,
        initialize_noise_level: float = 1e-3) -> np.ndarray:
    """Mass-generate reconstructions for MAP-linear-dCNN method.

    :param observations: shape (n_examples, n_measurements) — observed y for each image
    :param measurement_matrix: shape (n_measurements, height*width)
    :param image_shape: (height, width)
    :param reconstruction_hyperparams: HQS hyperparameters
    :param max_batch_size: number of images to reconstruct in parallel
    :param device: torch device
    :param initialize_noise_level: std dev for z(1) Gaussian initialization
    :return: reconstructed images, shape (n_examples, height, width)
    """
    n_examples, n_measurements = observations.shape
    height, width = image_shape

    observations_torch = torch.tensor(observations, dtype=torch.float32, device=device)

    schedule_rho = make_hqs_schedule(reconstruction_hyperparams)

    unblind_denoiser_model = denoiser_wrappers.load_zhang_drunet_unblind_denoiser(device)
    unblind_denoiser_callable = denoiser_wrappers.make_unblind_apply_zhang_dpir_denoiser(
        unblind_denoiser_model,
        (-1.0, 1.0), (0.0, 255))

    # build models sized for the first N-1 full batches
    x_prob = BatchLinearMeasurementProxProblem(
        max_batch_size,
        measurement_matrix,
        image_shape,
        schedule_rho[0],
    ).to(device)

    z_prob = BatchParallel_UnblindDenoiserPrior_HQS_ZProb(
        max_batch_size,
        unblind_denoiser_callable,
        image_shape,
        schedule_rho[0],
        prior_lambda=reconstruction_hyperparams.prior_weight,
    ).to(device)

    output_buffer = np.zeros((n_examples, height, width), dtype=np.float32)
    pbar = tqdm.tqdm(total=n_examples)

    for low in range(0, n_examples - max_batch_size + 1, max_batch_size):
        high = low + max_batch_size
        batch_obs = observations_torch[low:high, ...]

        recons = generate_onebatch_linear_hqs_reconstruction(
            x_prob, z_prob, image_shape, batch_obs,
            reconstruction_hyperparams,
            initialize_noise_level=initialize_noise_level,
        ).detach().cpu().numpy()

        output_buffer[low:high, :, :] = recons
        pbar.update(max_batch_size)

    del x_prob, z_prob

    # final (possibly smaller) batch
    low = (n_examples // max_batch_size) * max_batch_size
    high = n_examples
    eff_batch_size = high - low

    if eff_batch_size > 0:
        x_prob_final = BatchLinearMeasurementProxProblem(
            eff_batch_size,
            measurement_matrix,
            image_shape,
            schedule_rho[0],
        ).to(device)

        z_prob_final = BatchParallel_UnblindDenoiserPrior_HQS_ZProb(
            eff_batch_size,
            unblind_denoiser_callable,
            image_shape,
            schedule_rho[0],
            prior_lambda=reconstruction_hyperparams.prior_weight,
        ).to(device)

        batch_obs = observations_torch[low:high, ...]

        recons = generate_onebatch_linear_hqs_reconstruction(
            x_prob_final, z_prob_final, image_shape, batch_obs,
            reconstruction_hyperparams,
            initialize_noise_level=initialize_noise_level,
        ).detach().cpu().numpy()

        output_buffer[low:high, :, :] = recons
        pbar.update(eff_batch_size)

        del x_prob_final, z_prob_final

    pbar.close()
    del observations_torch, unblind_denoiser_model

    return output_buffer


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        "Mass-generate reconstructions for MAP-linear-dCNN method")
    parser.add_argument('output_path', type=str,
                        help='save path for reconstructions (.pkl)')
    parser.add_argument('measurement_matrix_path', type=str,
                        help='path to measurement matrix M, .npy, shape (n_measurements, height*width)')
    parser.add_argument('observations_path', type=str,
                        help='path to observations y, .npy, shape (n_examples, n_measurements)')
    parser.add_argument('ground_truth_path', type=str,
                        help='path to ground truth images, .npy, shape (n_examples, height, width)')
    parser.add_argument('-b', '--batch', type=int, default=16,
                        help='batch size for reconstruction')
    parser.add_argument('-n', '--noise_init', type=float, default=1e-3,
                        help='std dev of Gaussian noise for z(1) initialization')
    parser.add_argument('--rho_start', type=float, default=0.1)
    parser.add_argument('--rho_end', type=float, default=100.0)
    parser.add_argument('--prior_weight', type=float, default=0.1)
    parser.add_argument('--max_iter', type=int, default=25)
    parser.add_argument('-gpu', '--gpu', action='store_true', default=False,
                        help='use GPU')
    args = parser.parse_args()

    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    measurement_matrix = np.load(args.measurement_matrix_path)  # (n_measurements, height*width)
    observations = np.load(args.observations_path)               # (n_examples, n_measurements)
    ground_truth_images = np.load(args.ground_truth_path)        # (n_examples, height, width)

    n_examples, height, width = ground_truth_images.shape
    assert measurement_matrix.shape[1] == height * width, \
        f'measurement_matrix columns ({measurement_matrix.shape[1]}) must equal height*width ({height * width})'
    assert observations.shape == (n_examples, measurement_matrix.shape[0]), \
        f'observations shape mismatch'

    hyperparameters = HQSHyperparameters(
        rho_start=args.rho_start,
        rho_end=args.rho_end,
        prior_weight=args.prior_weight,
        max_iter=args.max_iter,
    )

    print("Generating MAP-linear-dCNN reconstructions")
    reconstructions = batch_parallel_generate_linear_hqs_reconstructions(
        observations,
        measurement_matrix,
        (height, width),
        hyperparameters,
        args.batch,
        device,
        initialize_noise_level=args.noise_init,
    )

    with open(args.output_path, 'wb') as pfile:
        pickle.dump({
            'ground_truth': ground_truth_images,
            'linear_hqs': reconstructions,
        }, pfile)

    print('done')
