import pickle
from typing import Optional, Tuple
import argparse

import matplotlib.pyplot as plt
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
        initialize_noise_level: float = 1e-3,
        track_distances: bool = False,
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    """Run one batch of HQS with a linear measurement likelihood and dCNN prior.

    :param x_prob: linear measurement X-problem
    :param z_prob: DRUNet denoiser Z-problem
    :param image_shape: (height, width)
    :param batch_observations: shape (batch, n_measurements) — observed y = Mx + noise
    :param reconstruction_hyperparams: HQS schedule / prior weight / max_iter
    :param initialize_noise_level: std dev of Gaussian noise used to initialize z
    :param track_distances: if True, return per-iteration distances (max_iter, 2) where
        column 0 is ||x_i - z_{i-1}|| (X-step displacement) and
        column 1 is ||z_i - x_i|| (Z-step displacement).
    :return: (reconstructed images shape (batch, H, W),
              distances shape (max_iter, 2) or None)
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

    intermediates = scheduled_rho_fixed_lambda_single_hqs_solve(
        x_prob,
        iter(x_solver_iter),
        z_prob,
        iter(z_solver_iter),
        iter(schedule_rho),
        prior_weight,
        max_iter,
        verbose=False,
        save_intermediates=track_distances,
    )

    distances = None
    if track_distances and intermediates:
        distances = np.zeros((len(intermediates), 2), dtype=np.float32)
        z_prev = initialize_z_tensor.cpu().numpy()  # z_{-1} is the initialization
        for i, (x_tensor, z_np) in enumerate(intermediates):
            x_np = x_tensor.cpu().numpy()                            # (batch, H, W)
            diff_x = (x_np - z_prev).reshape(batch_size, -1)        # x_i - z_{i-1}
            diff_z = (z_np - x_np).reshape(batch_size, -1)          # z_i - x_i
            distances[i, 0] = float(np.mean(np.linalg.norm(diff_x, axis=1)))
            distances[i, 1] = float(np.mean(np.linalg.norm(diff_z, axis=1)))
            z_prev = z_np

    return x_prob.get_reconstructed_image(), distances


def batch_parallel_generate_linear_hqs_reconstructions(
        observations: np.ndarray,
        measurement_matrix: np.ndarray,
        image_shape: Tuple[int, int],
        reconstruction_hyperparams: HQSHyperparameters,
        max_batch_size: int,
        device: torch.device,
        k : float,
        initialize_noise_level: float = 1e-3,
        track_distances: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Mass-generate reconstructions for MAP-linear-dCNN method.

    :param observations: shape (n_examples, n_measurements) — observed y for each image
    :param measurement_matrix: shape (n_measurements, height*width)
    :param image_shape: (height, width)
    :param reconstruction_hyperparams: HQS hyperparameters
    :param max_batch_size: number of images to reconstruct in parallel
    :param device: torch device
    :param k: steepness of non-linearity after applying linear measurement
    :param initialize_noise_level: std dev for z(1) Gaussian initialization
    :param track_distances: if True, return per-iteration HQS step distances (max_iter, 2)
        averaged over all examples.
    :return: (reconstructed images (n_examples, H, W),
              distances (max_iter, 2) averaged over all examples, or None)
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
        k=k,
    ).to(device)

    z_prob = BatchParallel_UnblindDenoiserPrior_HQS_ZProb(
        max_batch_size,
        unblind_denoiser_callable,
        image_shape,
        schedule_rho[0],
        prior_lambda=reconstruction_hyperparams.prior_weight,
    ).to(device)

    output_buffer = np.zeros((n_examples, height, width), dtype=np.float32)
    # accumulate weighted distances: list of (batch_size, distances array)
    weighted_distances = []
    pbar = tqdm.tqdm(total=n_examples)

    for low in range(0, n_examples - max_batch_size + 1, max_batch_size):
        high = low + max_batch_size
        batch_obs = observations_torch[low:high, ...]

        recons, distances = generate_onebatch_linear_hqs_reconstruction(
            x_prob, z_prob, image_shape, batch_obs,
            reconstruction_hyperparams,
            initialize_noise_level=initialize_noise_level,
            track_distances=track_distances,
        )
        output_buffer[low:high, :, :] = recons.detach().cpu().numpy()
        if distances is not None:
            weighted_distances.append((max_batch_size, distances))
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
            k=k,
        ).to(device)

        z_prob_final = BatchParallel_UnblindDenoiserPrior_HQS_ZProb(
            eff_batch_size,
            unblind_denoiser_callable,
            image_shape,
            schedule_rho[0],
            prior_lambda=reconstruction_hyperparams.prior_weight,
        ).to(device)

        batch_obs = observations_torch[low:high, ...]

        recons, distances = generate_onebatch_linear_hqs_reconstruction(
            x_prob_final, z_prob_final, image_shape, batch_obs,
            reconstruction_hyperparams,
            initialize_noise_level=initialize_noise_level,
            track_distances=track_distances,
        )
        output_buffer[low:high, :, :] = recons.detach().cpu().numpy()
        if distances is not None:
            weighted_distances.append((eff_batch_size, distances))
        pbar.update(eff_batch_size)

        del x_prob_final, z_prob_final

    pbar.close()
    del observations_torch, unblind_denoiser_model

    avg_distances = None
    if weighted_distances:
        total = sum(n for n, _ in weighted_distances)
        avg_distances = sum(n * d for n, d in weighted_distances) / total

    return output_buffer, avg_distances


if __name__ == '__main__':
    # for running:
    # python generate_MAP_linear_dCNN_recons.py gaussian_std_0_recon.pkl gaussian_std_0_resources/M.npy gaussian_std_0_resources/y.npy gaussian_std_0_resources/gt_images.npy --gpu
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
                        help='use GPU'),
    parser.add_argument('--k', type=float, default=0.0,
                    help='Steepness parameter for sigmoid non-linearity')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='plot per-iteration Euclidean distances from ground truth '
                             '(one point after X-step, one after Z-step per iteration)')
    args = parser.parse_args()

    device = torch.device('cuda') if args.gpu else torch.device('cpu')
    print(device)

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
    reconstructions, distances = batch_parallel_generate_linear_hqs_reconstructions(
        observations,
        measurement_matrix,
        (height, width),
        hyperparameters,
        args.batch,
        device,
        initialize_noise_level=args.noise_init,
        k=args.k,
        track_distances=args.plot,
    )

    with open(args.output_path, 'wb') as pfile:
        pickle.dump({
            'ground_truth': ground_truth_images,
            'linear_hqs': reconstructions,
        }, pfile)

    if args.plot and distances is not None:
        iters = np.arange(len(distances))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(iters - 0.15, distances[:, 0], label='After X-step (FISTA → measurement)',
                   marker='o', zorder=3)
        ax.scatter(iters + 0.15, distances[:, 1], label='After Z-step (denoiser → prior)',
                   marker='s', zorder=3)
        ax.set_xlabel('HQS Iteration')
        ax.set_ylabel('Mean Euclidean Distance from Ground Truth')
        ax.set_title('HQS per-iteration distances')
        ax.set_xticks(iters)
        ax.legend()
        plt.tight_layout()
        plt.savefig('hqs_distances.png', dpi=150, bbox_inches='tight')
        plt.show()

    print('done')
