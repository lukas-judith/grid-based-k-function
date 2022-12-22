import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm
from scipy import signal
from skimage import exposure, restoration
from skimage.filters import gaussian
from scipy.ndimage.morphology import (
    binary_erosion,
    binary_dilation,
    binary_fill_holes,
    generate_binary_structure,
)
from scipy.ndimage.filters import maximum_filter

from utils import *


def load_image(path):
    """
    Returns image and metadata from filepath of .tif image.
    """
    meta_data = {}

    with TiffFile(path) as tif:
        # WARNING: metadata only for first page in this version
        for info in tif.pages[0].tags.values():
            key, value = info.name, info.value
            # toss unneccessary metadata
            # if not key in ["IJMetadataByteCounts", "IJMetadata"]:
            meta_data[key] = value
        im = tif.asarray()
    return im, meta_data


def scale_image(img, desired_int):
    """
    Scales the input image to have a specified total intensity.
    """
    total_int = np.sum(img)
    img_scaled = img * desired_int / total_int
    return img_scaled


def crop_image(img, mask):
    """
    Multiplies binary mask with image.
    """
    return img * mask


def normalize_image(img):
    """
    Maps image intensities to interval [0,1].
    """
    img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img_norm


def create_csr_img(mask, desired_int):
    """
    Generates uniformely random pixel intensities in a shape specified by mask.
    """
    rnd_array = np.random.random(size=mask.shape)
    csr_img = rnd_array * mask
    csr_img *= desired_int / np.sum(csr_img)
    return csr_img


def load_image2D(path, z, channel=0):
    """
    Load 3D cell image from .tif file and extract a desired z slice.
    Returns 2D image array and metadata.
    """
    print(f"Loading channel={channel}, z={z+1}")
    # get array with all pixel intensities and metadata of image
    im, metadata = load_image(path)

    # pick desired channel
    zxy_arr = im[:, channel, :, :]

    # extract 2D slice
    xy_array = zxy_arr[z, :, :].astype("float32")

    return xy_array, metadata


def get_nth_minimum_after_maximum(xs, ys, n, min_dist_to_max=0, max_position_thresh=1):
    """
    For a given 1D signal (xs should be normalized to [0,1]), find the global
    maximum and return the nth minimum after this global maximum. Exclude
    all minima that are too close to maximum, specified by min_dist_to_max.
    """
    if np.min(xs) < 0 or np.max(xs) > 1:
        print("Error! Input signal is not normalized to [0,1]!")
        return

    # convert to array in case that inputs are lists
    xs_ = np.array(xs)
    ys_ = np.array(ys)

    maximum = np.argmax(ys_)
    maximum_pos_x = xs[maximum]

    if maximum_pos_x > max_position_thresh:
        # position of maximum is outside of accepted range
        # from 0 to max_position_thresh
        return None

    minima = signal.find_peaks(-ys_)[0].astype(int)

    minima_x_pos = xs_[minima]
    minima_dist_to_max = minima_x_pos - xs_[maximum]
    # filter out all minima that are too close to maximum
    minima = minima[minima_dist_to_max > min_dist_to_max]

    # get nth minimum after maximum
    minima_after_maximum = minima[minima > maximum]

    if len(minima_after_maximum) < n:
        # minimum could not be found
        return None

    nth_min_after_max = minima_after_maximum[n - 1]
    return nth_min_after_max


def print_tif_series(filepath, destination="tif_series", channel=0):
    """
    From one .tif file, store every xy slice for all z values.
    """
    filename = os.path.basename(filepath)
    print(f"Creating series for .tif file {filename}")
    im, _ = load_image(filepath)

    # pick desired channel
    zxy_arr = im[:, channel, :, :]

    N_z = zxy_arr.shape[0]

    fig, ax = plt.subplots(N_z, 1, figsize=(5, 5 * N_z))

    # loop over all possible z values
    for z in range(N_z):
        # extract 2D slices
        xy_array = zxy_arr[z, :, :].astype("float32")
        ax[z].imshow(xy_array)
        ax[z].set_title(f"z = {z}")

    plt.savefig(
        os.path.join(destination, f"{filename[:-4]}_tif_series_channel{channel}")
    )
    print("Done!")


def extract_filedata(txt_filename, tif_folder):
    """
    From a .txt file, extract the relevant path, z slice and channel
    for a dataset consisting of .tif files.
    """
    data = []

    with open(txt_filename) as file:
        content = file.read()
        lines = content.split("\n")
        for line in lines:
            if len(line) > 1:
                filename, channel, z = line.split(" ")
                filepath = os.path.join(tif_folder, filename)
                data.append([filepath, int(channel), int(z) - 1])
    return data


def remove_background(
    img_array, rolling_ball_radius, savepath="./test", check_plot=False, z=None
):
    """
    Removes background using rolling ball algorithm.
    """
    kernel = restoration.ball_kernel(rolling_ball_radius, ndim=2)
    background = restoration.rolling_ball(img_array, kernel=kernel)
    signal = img_array - background

    # ---------------------------------------------------------------
    # create plots for checking if the method is working correctly
    # ---------------------------------------------------------------
    if check_plot:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].set_title("Original image")
        ax[0].imshow(img_array)
        ax[1].set_title("Signal")
        ax[1].imshow(signal)
        ax[2].set_title("Background")
        ax[2].imshow(background)
        print("Saving check plot at", savepath)
        plt.savefig(savepath)
        plt.close()

    return signal


def create_mask(
    img_array,
    sigma=20,
    iter_erode=35,
    iter_dilate=20,
    sigma_for_finding_minima=2,
    n_min=1,
    distance_min_max=0.05,
    max_position_thresh=0.1,
    z=None,
    postprocessing=True,
    savepath="./test.pdf",
    check_plot=False,
    custom_thresh=None,
):
    """
    Given a 2D image, computes a binary mask capturing the shape of the
    object(s) in the image. Uses binary dilation, erosion and filling holes
    to fill gaps and remove artifacts. When check_plot=True, plots of
    intermediate results during the computation are produced.
    Parameters can then be adjusted accordingly.

    Args postprocessing:
        img (np.ndarray): 2D input image
        sigma (float): standard deviation for Gaussian blur
        iter_erode (int): number of iterations for binary dilation of mask
                          (postprocessing)
        iter_dilate (int): number of iterations for binary erosion of mask
                          (postprocessing)

    Args for finding optimal cut-off value:
        sigma_for_finding_minima (float): standard deviation for smoothing the
            function of histogram counts for intensities of the blurred image;
            can be used to facilitate finding the optimal cut-off value for
            computing the binary 2D mask.
        n_min (int): number of minimum in function of histogram counts for
            intensities of the blurred image to be selected as the cut-off value
            for computing the binary 2D mask.
        distance_min_max (float): minimum distance (assumes normalized images,
            i.e. in interval [0,1]) that the minimum (of the function of
            histogram counts for intensities of the blurred image) is allowed
            to have from the global maximum.
        max_position_thresh (float): maximum position (assumes normalized
            images, i.e. in interval [0,1]) that the maximum (of the function of
            histogram counts for intensities of the blurred image)
            is allowed to have.
        custom_thresh (float): if not None, choose this value for the cut-off
            (instead of finding the value from the histogram counts).

    Returns:
        np.ndarray : 2D binary mask capturing the shape of the object(s)
        on the input image.
    """
    if len(img_array.shape) != 2:
        print("Error! Input image must be 2-dimensional!")
        return

    # map image intensities to the interval [0,1]
    img_normalized = normalize_image(img_array)

    # blur image and map resulting intensities to the interval [0,1]
    img_blur = img_normalized.copy()
    img_blur = gaussian(img_blur, sigma=sigma)
    img_blur_norm = normalize_image(img_blur)

    # create histogram of blurred image intensities in order to find optimal
    # cut-off for extracting the cell shape from the image
    counts, bin_edges = np.histogram(img_blur_norm.ravel(), bins=200)
    bin_middles = np.diff(bin_edges)
    bin_middles = [
        (bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)
    ]

    # find first local minimum in counts of the histogram, this should separate
    # the peak from the background intensities and the peak from the intensities
    # within the cell area

    # smooth values to facilitate finding the true local minima
    arr = gaussian(counts, sigma=sigma_for_finding_minima)

    # attempt to find first minimum by checking for the first point in the plot
    # after the maximum at which the count value start increasing after the
    # initial decrease
    x_cut = get_nth_minimum_after_maximum(
        xs=bin_middles,
        ys=arr,
        n=n_min,
        min_dist_to_max=distance_min_max,
        max_position_thresh=max_position_thresh,
    )

    # choose this as cut-off value above which we consider the pixels to be
    # within the cell
    if x_cut is None:
        # if no minimum can be found, discard the slice, i.e. set all values to
        # zero (assuming all values are >= 0)
        cut = 0
    else:
        cut = bin_middles[x_cut]

    mask = img_blur_norm > cut

    if not custom_thresh is None:
        mask = img_blur_norm > custom_thresh

    # some additional processing:
    # fill holes, erode artifacts from Gaussian blur
    # erode more than necessary, then dilate again to remove smaller objects
    if postprocessing:
        mask = binary_dilation(
            binary_erosion(binary_fill_holes(mask), iterations=iter_erode),
            iterations=iter_dilate,
        )

    # ---------------------------------------------------------------
    # create plots for checking if the method is working correctly
    # ---------------------------------------------------------------

    if check_plot:
        fig, ax = plt.subplots(3, 2, figsize=(10, 15))
        ax[0][0].set_title("Original image")
        ax[0][0].imshow(img_array)
        ax[0][1].set_title("Blurred image")
        ax[0][1].imshow(img_blur)
        ax[1][0].plot(bin_middles, counts)
        if not x_cut is None:
            ax[1][0].scatter(
                bin_middles[x_cut : x_cut + 1],
                counts[x_cut : x_cut + 1],
                color="r",
                marker="x",
                label=f"Cut-off = {cut:.4f}",
            )
            ax[1][0].set_ylabel("Counts")
            ax[1][0].set_xlabel("Pixel intensity")
            ax[1][0].set_yscale("log")
            ax[1][0].legend()
            ax[1][0].set_title("Smoothed values of histogram counts")
        else:
            ax[1][0].set_title("No acceptable cut-off value found")
        ax[1][1].set_title("Mask = blurred image after cut-off")
        ax[1][1].imshow(mask)
        ax[2][0].set_title("Overlay of orig. image with mask")
        ax[2][0].imshow(mask - img_normalized * 4)
        ax[2][1].set_title("Compare: orig. image, histogram equalized")
        ax[2][1].imshow(exposure.equalize_hist(img_array))

        print("Saving check plot at", savepath)
        plt.savefig(savepath)
        plt.close()

    return mask


def process_image2D(
    img_array,
    desired_int,
    mask_params,
    rm_background_params,
    folder=".",
    check_plot=False,
):
    """
    Performs preprocessing of 2D image, including background removal
    and computation of binary 2D mask capturing the object(s) on the image.
    Returns real and CSR image to be used in K-function computation,
    as well as the binary mask capturing the object(s) on the image.
    """
    # compute mask that captures the silhouette of the cell.
    cell_mask = create_mask(
        img_array, *mask_params, folder=folder, check_plot=check_plot
    )

    # remove background from cell image
    img_signal = remove_background(
        img_array, *rm_background_params, folder=folder, check_plot=check_plot
    )

    # crop out the object(s) using the mask
    img_cropped = crop_image(img_signal, cell_mask)

    # compute CSR image, which depicts the cell with uniform intensity
    # and scale real (cropped-out) image and CSR image to have the sum of pixel
    # intensities
    img_csr = create_csr_img(cell_mask, desired_int)
    img_real = scale_image(img_cropped, desired_int)

    # check if sum of pixel intensities is the same for real and CSR image
    if not np.isclose(np.sum(img_real), np.sum(img_csr), rtol=1e-05):
        print(
            "Error! Sum of all pixel intensities is different for real and CSR image!"
        )
        return

    return img_real, img_csr, cell_mask


# Code for extracting points from images borrowded from:
# https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py
"""
# MIT License
#
# Copyright (c) 2017 ZijunDeng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""


def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = image == 0

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1
    )

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks
