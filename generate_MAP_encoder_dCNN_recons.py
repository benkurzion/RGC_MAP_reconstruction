import pickle
from typing import Tuple
import argparse

import tqdm
import numpy as np
import torch

import denoisers.denoiser_wrappers as denoiser_wrappers

from autoencoder.autoencoder import Neurips2017_Encoder
from reconstruction_alg.encoder_inverse_alg import BatchEncoderMeasurementProxProblem
from reconstruction_alg.hqs_alg import (
    BatchParallel_HQS_XGenerator,
    BatchParallel_LM_HQS_XGenerator,
    BatchParallel_DirectSolve_HQS_ZGenerator,
    BatchParallel_UnblindDenoiserPrior_HQS_ZProb,
    scheduled_rho_fixed_lambda_single_hqs_solve,
)
from convex_solver_base.unconstrained_optim import LevenbergMarquardtSolverParams
from hyperparameters.hyperparameters import HQSHyperparameters, make_hqs_schedule


def generate_onebatch_encoder_hqs_reconstruction(
        x_prob: BatchEncoderMeasurementProxProblem,
        z_prob: BatchParallel_UnblindDenoiserPrior_HQS_ZProb,
        image_shape: Tuple[int, int],
        batch_observations: torch.Tensor,
        reconstruction_hyperparams: HQSHyperparameters,
        initialize_noise_level: float = 1e-3,
        lm_params: LevenbergMarquardtSolverParams = None,
        verbose_lm: bool = False) -> torch.Tensor:
    """Run one batch of HQS with encoder measurement likelihood and dCNN prior.

    :param x_prob: encoder measurement X-problem
    :param z_prob: DRUNet denoiser Z-problem
    :param image_shape: (height, width)
    :param batch_observations: shape (batch, enc_out_size) — observed y = E(x_gt)
    :param reconstruction_hyperparams: HQS schedule / prior weight / max_iter
    :param initialize_noise_level: std dev of Gaussian noise used to initialize z
    :return: reconstructed images, shape (batch, height, width)
    """
    height, width = image_shape
    batch_size = batch_observations.shape[0]

    schedule_rho = make_hqs_schedule(reconstruction_hyperparams)
    prior_weight = reconstruction_hyperparams.prior_weight
    max_iter = reconstruction_hyperparams.max_iter

    if lm_params is not None:
        x_solver_iter = BatchParallel_LM_HQS_XGenerator(lm_params, verbose_lm=verbose_lm)
    else:
        x_solver_iter = BatchParallel_HQS_XGenerator(first_niter=300, subsequent_niter=300)
    z_solver_iter = BatchParallel_DirectSolve_HQS_ZGenerator()

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
        verbose=verbose_lm,
        save_intermediates=False,
    )

    return x_prob.get_reconstructed_image()


def batch_parallel_generate_encoder_hqs_reconstructions(
        observations: np.ndarray,
        encoder_weights_path: str,
        obs_scale: float,
        image_shape: Tuple[int, int],
        reconstruction_hyperparams: HQSHyperparameters,
        max_batch_size: int,
        device: torch.device,
        initialize_noise_level: float = 1e-3,
        lm_params: LevenbergMarquardtSolverParams = None,
        verbose_lm: bool = False) -> np.ndarray:
    """Mass-generate reconstructions for MAP-encoder-dCNN method.

    :param observations: shape (n_examples, enc_out_size) — observed encoder outputs
    :param encoder_weights_path: path to .pth state dict from prepare_encoder_measurements.py
    :param image_shape: (height, width)
    :param reconstruction_hyperparams: HQS hyperparameters
    :param max_batch_size: number of images to reconstruct in parallel
    :param device: torch device
    :param initialize_noise_level: std dev for z(1) Gaussian initialization
    :return: reconstructed images, shape (n_examples, height, width)
    """
    n_examples = observations.shape[0]
    height, width = image_shape

    observations_torch = torch.tensor(observations, dtype=torch.float32, device=device)

    schedule_rho = make_hqs_schedule(reconstruction_hyperparams)

    encoder = Neurips2017_Encoder(dropout_rate=0.0).to(device)
    encoder.load_state_dict(torch.load(encoder_weights_path, map_location=device))
    encoder.eval()

    unblind_denoiser_model = denoiser_wrappers.load_zhang_drunet_unblind_denoiser(device)
    unblind_denoiser_callable = denoiser_wrappers.make_unblind_apply_zhang_dpir_denoiser(
        unblind_denoiser_model,
        (-1.0, 1.0), (0.0, 255))

    x_prob = BatchEncoderMeasurementProxProblem(
        max_batch_size,
        encoder,
        image_shape,
        schedule_rho[0],
        obs_scale=obs_scale,
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

        recons = generate_onebatch_encoder_hqs_reconstruction(
            x_prob, z_prob, image_shape, batch_obs,
            reconstruction_hyperparams,
            initialize_noise_level=initialize_noise_level,
            lm_params=lm_params,
            verbose_lm=verbose_lm,
        ).detach().cpu().numpy()

        output_buffer[low:high, :, :] = recons
        pbar.update(max_batch_size)

    del x_prob, z_prob

    # final (possibly smaller) batch
    low = (n_examples // max_batch_size) * max_batch_size
    high = n_examples
    eff_batch_size = high - low

    if eff_batch_size > 0:
        x_prob_final = BatchEncoderMeasurementProxProblem(
            eff_batch_size,
            encoder,
            image_shape,
            schedule_rho[0],
            obs_scale=obs_scale,
        ).to(device)

        z_prob_final = BatchParallel_UnblindDenoiserPrior_HQS_ZProb(
            eff_batch_size,
            unblind_denoiser_callable,
            image_shape,
            schedule_rho[0],
            prior_lambda=reconstruction_hyperparams.prior_weight,
        ).to(device)

        batch_obs = observations_torch[low:high, ...]

        recons = generate_onebatch_encoder_hqs_reconstruction(
            x_prob_final, z_prob_final, image_shape, batch_obs,
            reconstruction_hyperparams,
            initialize_noise_level=initialize_noise_level,
            lm_params=lm_params,
            verbose_lm=verbose_lm,
        ).detach().cpu().numpy()

        output_buffer[low:high, :, :] = recons
        pbar.update(eff_batch_size)

        del x_prob_final, z_prob_final

    pbar.close()
    del observations_torch, unblind_denoiser_model, encoder

    return output_buffer


if __name__ == '__main__':

    #  python generate_MAP_encoder_dCNN_recons.py encoder_scaled_recon.pkl encoder_scaled_resources/encoder_weights.pth encoder_scaled_resources/obs_scale.npy encoder_scaled_resources/y.npy encoder_scaled_resources/gt_images.npy --gpu --rho_start 1.0
    parser = argparse.ArgumentParser(
        description='Mass-generate reconstructions for MAP-encoder-dCNN method')
    parser.add_argument('output_path', type=str,
                        help='save path for reconstructions (.pkl)')
    parser.add_argument('encoder_weights_path', type=str,
                        help='path to encoder state dict (.pth) from prepare_encoder_measurements.py')
    parser.add_argument('obs_scale_path', type=str,
                        help='path to obs_scale.npy from prepare_encoder_measurements.py')
    parser.add_argument('observations_path', type=str,
                        help='path to observations y (.npy), shape (n_examples, enc_out_size)')
    parser.add_argument('ground_truth_path', type=str,
                        help='path to ground truth images (.npy), shape (n_examples, height, width)')
    parser.add_argument('-b', '--batch', type=int, default=16,
                        help='batch size for reconstruction (default: 16)')
    parser.add_argument('-n', '--noise_init', type=float, default=1e-3,
                        help='std dev of Gaussian noise for z(1) initialization (default: 1e-3)')
    parser.add_argument('--rho_start', type=float, default=0.1)
    parser.add_argument('--rho_end', type=float, default=100.0)
    parser.add_argument('--prior_weight', type=float, default=0.1)
    parser.add_argument('--max_iter', type=int, default=25)
    parser.add_argument('--fista', action='store_true', default=False,
                        help='use FISTA for X-step instead of Levenberg-Marquardt')
    parser.add_argument('--lm_max_iter', type=int, default=15,
                        help='LM outer iterations per HQS step (default: 15)')
    parser.add_argument('--lm_cg_iter', type=int, default=20,
                        help='CG iterations per LM step (default: 20)')
    parser.add_argument('--lm_lambda0', type=float, default=1.0,
                        help='initial LM damping lambda (default: 1.0)')
    parser.add_argument('--verbose_lm', action='store_true', default=False,
                        help='print per-iteration LM accept/reject counts and warnings')
    parser.add_argument('-gpu', '--gpu', action='store_true', default=False,
                        help='use GPU')
    args = parser.parse_args()

    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    obs_scale = float(np.load(args.obs_scale_path))
    observations = np.load(args.observations_path)         # (n_examples, enc_out_size), already normalized
    ground_truth_images = np.load(args.ground_truth_path)  # (n_examples, height, width)
    print(f"obs_scale={obs_scale:.6f}")

    n_examples, height, width = ground_truth_images.shape
    assert observations.shape[0] == n_examples, \
        f'observations n_examples ({observations.shape[0]}) must match ground truth ({n_examples})'

    hyperparameters = HQSHyperparameters(
        rho_start=args.rho_start,
        rho_end=args.rho_end,
        prior_weight=args.prior_weight,
        max_iter=args.max_iter,
    )

    lm_params = None if args.fista else LevenbergMarquardtSolverParams(
        initial_lambda=args.lm_lambda0,
        max_iter=args.lm_max_iter,
        max_cg_iter=args.lm_cg_iter,
    )

    solver_name = 'FISTA' if args.fista else f'LM (max_iter={args.lm_max_iter}, cg_iter={args.lm_cg_iter})'
    print(f"Generating MAP-encoder-dCNN reconstructions  [X-step solver: {solver_name}]")
    reconstructions = batch_parallel_generate_encoder_hqs_reconstructions(
        observations,
        args.encoder_weights_path,
        obs_scale,
        (height, width),
        hyperparameters,
        args.batch,
        device,
        initialize_noise_level=args.noise_init,
        lm_params=lm_params,
        verbose_lm=args.verbose_lm,
    )

    with open(args.output_path, 'wb') as pfile:
        pickle.dump({
            'ground_truth': ground_truth_images,
            'encoder_hqs': reconstructions,
        }, pfile)

    print('done')
