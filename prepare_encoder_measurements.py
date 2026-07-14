import argparse
import os
import numpy as np
import torch
import tqdm

from data_util.load_data import load_stacked_dataset
from autoencoder.autoencoder import Neurips2017_Encoder


def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute encoder measurements y = E(x) for each ground-truth image '
                    'and save the frozen encoder weights so reconstruction uses the same E.')
    parser.add_argument('output_dir', type=str,
                        help='directory to save encoder_weights.pth, y.npy, gt_images.npy')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='encoder dropout rate (default: 0.0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for encoding pass (default: 32)')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='use GPU')
    parser.add_argument('--target_std', type=float, default=0.5,
                        help='target std for normalized encoder outputs, matching GLM binary '
                             'spike scale (default: 0.5)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    ground_truth_images, _ = load_stacked_dataset()
    # shape (n_examples, height, width), values in [-1, 1]
    n_examples, height, width = ground_truth_images.shape

    # random-init encoder — weights saved below so reconstruction loads the same ones
    encoder = Neurips2017_Encoder(dropout_rate=args.dropout).to(device)
    for param in encoder.parameters():
        param.requires_grad_(False)

    encoder.eval()

    images_torch = torch.tensor(ground_truth_images, dtype=torch.float32)

    # determine flattened encoder output size
    with torch.no_grad():
        dummy = torch.zeros((1, height, width), dtype=torch.float32, device=device)
        enc_out_size = encoder(dummy).numel()
    print(f"Encoder output size per image: {enc_out_size}")

    # encode all images
    observations = np.zeros((n_examples, enc_out_size), dtype=np.float32)

    for low in tqdm.trange(0, n_examples, args.batch_size, desc='encoding'):
        high = min(low + args.batch_size, n_examples)
        batch = images_torch[low:high].to(device)
        with torch.no_grad():
            enc_out = encoder(batch).reshape(high - low, -1)
        observations[low:high] = enc_out.cpu().numpy()

    # Estimate obs_scale from random noise images rather than from the ground truth,
    # so the normalization constant is a property of the encoder alone (no leakage).
    n_calib = 256
    with torch.no_grad():
        noise = torch.randn(n_calib, height, width, dtype=torch.float32, device=device)
        calib_out = encoder(noise).reshape(n_calib, -1).cpu().numpy()
    obs_scale = float(calib_out.std()) / args.target_std
    observations_normalized = observations / obs_scale
    print(f"Encoder output std (from {n_calib} random images): {float(calib_out.std()):.6f}")
    print(f"obs_scale (divisor to reach target_std={args.target_std}): {obs_scale:.6f}")
    print(f"Normalized observations std: {observations_normalized.std():.6f}")

    weights_path = os.path.join(args.output_dir, 'encoder_weights.pth')
    y_path       = os.path.join(args.output_dir, 'y.npy')
    gt_path      = os.path.join(args.output_dir, 'gt_images.npy')
    scale_path   = os.path.join(args.output_dir, 'obs_scale.npy')

    torch.save(encoder.state_dict(), weights_path)
    np.save(y_path, observations_normalized)
    np.save(gt_path, ground_truth_images)
    np.save(scale_path, np.array(obs_scale, dtype=np.float32))

    print("Saved:")
    print(f"  {weights_path}")
    print(f"  {y_path}   {observations_normalized.shape}  (normalized by obs_scale)")
    print(f"  {gt_path}  {ground_truth_images.shape}")
    print(f"  {scale_path}  obs_scale={obs_scale:.6f}  (normalized std={args.target_std})")


if __name__ == '__main__':
    main()
