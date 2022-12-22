import numpy as np

from scipy import signal


def is_even(integer):
    """
    Checks if integer input is even.
    """
    assert isinstance(integer, int)
    return int(integer / 2) * 2 == integer


def distance_arr(arr, len_x=1, len_y=1):
    """
    Computes array whose elements are equal to their Euclidean distance
    from the center.
    """
    dimy, dimx = arr.shape
    # array should have equal dimensions, dimension should be odd
    if (dimx != dimy) or is_even(dimx):
        print("Error! Array has wrong shape!")
        return

    half_dim = int((dimx - 1) / 2)
    x_range = list(range(1, half_dim + 1))
    line = x_range[::-1] + [0] + x_range

    # array where every element is x distance from center
    diff_x = np.array(
        [
            line,
        ]
        * dimx
    )

    # array where every element is y distance from center
    diff_y = diff_x.T

    # compute array whose elements are equal to their Euclidean distance
    # from the center
    diff_xy = np.sqrt((diff_x * len_x) ** 2 + (diff_y * len_y) ** 2)

    return diff_xy


def cut_circle(arr, radius, diff_xy):
    """
    Takes a 2D array and sets all elements outside a circle
    (around center of image) with specified radius to zero.
    """
    mask = np.ones(arr.shape)
    mask[diff_xy >= radius] = 0
    return arr * mask


def ripleys_K_grid(img_arr, range_of_t, area=1, printout=False):
    """
    Computes Ripley's K function for a range of t. Utilizes FFT
    for fast computation of auto-correlation of the image.
    """
    K_values = []
    # assure datatype that does not cause errors
    arr = img_arr.astype("float32")
    # full array for the auto-correlation of the input image
    full_auto_corr = signal.correlate(arr, arr, method="fft")

    # array whose elements are equal to their Euclidean distance from the center
    diff_xy = distance_arr(full_auto_corr)

    # orig. K-func.: coefficient = area / n_points^2
    coeff = area / np.sum(full_auto_corr)

    for t in range_of_t:
        # circular mask for adding selected values
        # from autocorrelation function
        mask = np.ones(full_auto_corr.shape)
        mask[diff_xy > t] = 0
        dimy, dimx = mask.shape
        # center for odd dimensions
        center_y = int(dimy / 2)
        center_x = int(dimx / 2)
        # center should also be zero
        mask[center_y, center_x] = 0
        # array containing the auto-correlation up to distance t from center
        auto_corr_t = full_auto_corr * mask
        # sum over all elements
        K = np.sum(auto_corr_t) * coeff
        K_values.append(K)
    return K_values
