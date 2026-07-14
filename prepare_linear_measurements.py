import argparse
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import tqdm

from data_util.load_data import load_stacked_dataset


def make_gaussian_blur_matrix(height: int, width: int, sigma: float) -> np.ndarray:
    """
    Creates a measurement matrix M (nxn) that applies a Gaussian blur onto the input image.

    :param height: image height
    :param width: image width
    :param sigma: Gaussian blur standard deviation in pixels
    :return: shape (n_pixels, n_pixels), float32
    """
    n_pixels = height * width
    M = np.zeros((n_pixels, n_pixels), dtype=np.float32)

    # wrapping in tqdm cuz the matrix could be huge
    # and going column by column could be slow
    for i in tqdm.trange(n_pixels):
        impulse = np.zeros(n_pixels, dtype=np.float32)
        impulse[i] = 1.0
        blurred = gaussian_filter(impulse.reshape(height, width), sigma=sigma)
        M[:, i] = blurred.ravel()

    return M # * 1000000 #scaling matrix


def make_random_matrix(height: int, width: int, n_measurements : int, type : str):
    n_pixels = height * width
    if type == 'uniform' :
        M = np.random.rand(n_measurements, n_pixels) #randn for standard normal distribution
    elif type == 'std_norm' :
        M = np.random.randn(n_measurements, n_pixels)
    return np.float32(M)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=str,
                        help='directory to save M.npy, y.npy, gt_images.npy')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Gaussian blur standard deviation in pixels (default: 2.0)')
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='std dev of Gaussian noise added to y (default: 0 = noiseless)')
    parser.add_argument('--type', type=str, default='gaussian', 
                        help='Measurement matrix type'),
    parser.add_argument('--k', type=float, default=0.0,
                    help='Steepness parameter for sigmoid non-linearity (0.0 = fully linear)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # load dataset
    ground_truth_images, _ = load_stacked_dataset()
    # shape (n_examples, height, width), values in [-1, 1]

    n_examples, height, width = ground_truth_images.shape
    n_pixels = height * width

    if args.type == 'gaussian':
        M = make_gaussian_blur_matrix(height, width, args.sigma)
    elif args.type in ['uniform', 'std_norm'] :
        M = make_random_matrix(height, width, 10240, type=args.type)
    else :
        raise ValueError("Invalid measurement type")
    
    print(f"  M shape: {M.shape},  memory: {M.nbytes / 1e6:.1f} MB")

    # compute observations y
    x_flat = ground_truth_images.reshape(n_examples, n_pixels).astype(np.float32)
    y_linear = (x_flat @ M.T).astype(np.float32)  # (n_examples, n_pixels)

    if args.k > 0:
        y = (np.tanh(args.k * y_linear) / np.tanh(args.k)).astype(np.float32)
    else:
        y = y_linear

    # optionally add noise to measurement model
    if args.noise_std > 0.0:
        y += np.random.randn(*y.shape).astype(np.float32) * args.noise_std

    M_path  = os.path.join(args.output_dir, 'M.npy')
    y_path  = os.path.join(args.output_dir, 'y.npy')
    gt_path = os.path.join(args.output_dir, 'gt_images.npy')

    np.save(M_path, M)
    np.save(y_path, y)
    np.save(gt_path, ground_truth_images)

    print("Saved:")
    print(f"  {M_path}   {M.shape}")
    print(f"  {y_path}   {y.shape}")
    print(f"  {gt_path}  {ground_truth_images.shape}")


if __name__ == '__main__':
    main()
