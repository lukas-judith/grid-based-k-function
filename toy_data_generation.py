import numpy as np

from skimage.filters import gaussian
from scipy.ndimage.morphology import binary_dilation


def cast_points_on_grid(points, d_orig, d_grid):
    """
    Casts list of 2D continuous point coordinates onto discrete grid.
    """
    points_scaled = points * d_grid/d_orig
    grid_idx = np.round(points_scaled).astype(int)
    # make sure no index is out of bounds
    grid_idx[grid_idx==d_grid] = d_grid-1
    # create 2D array with the indexed points
    val = 1
    grid = np.zeros((d_grid, d_grid))
    leftovers = grid_idx.copy()
    # gradually remove duplicates in the point indices,
    # increase the value on the grid if points overlap
    unique_vals, unique_idx = np.unique(leftovers, axis=0, return_index=True)
    grid[unique_vals[:, 0], unique_vals[:, 1]] = val
    # leftover point indices on grid after unique point indices have been removed
    leftovers = np.delete(leftovers, unique_idx, axis=0)

    while leftovers.size > 0:
        val += 1
        unique_vals, unique_idx = np.unique(leftovers, axis=0, return_index=True)
        grid[unique_vals[:, 0], unique_vals[:, 1]] = val
        leftovers = np.delete(leftovers, unique_idx, axis=0)
    return grid, grid_idx


def generate_points_and_grid(d, n, n_cluster=0, cluster_size=0, cluster_pos=None, dilate_points=False, pixel_sigma=1):
    """
    Generates random and, if specified, clustered points (Gaussian cluster shape). 
    Also casts points on discrete grid which is returned as array.
    """
    assert n >= n_cluster, 'Total number of points needs to be higher than number of clustered points'
    n_rnd = n - n_cluster

    # generate random points
    points2D = np.random.rand(n_rnd, 2)*(d-1)
    
    if n_cluster > 0 and not cluster_size > 0:
        cluster_size = int(d/4)

    if cluster_size > 0 and n_cluster > 0:
        # if not specified, cluster will be located in the center
        if cluster_pos is None:
            cluster_pos = (int(d/2), int(d/2))
        # generate clustered points
        # define size of cluster := FWHM = 2.355 * sigma (for Gaussian)
        sigma = cluster_size/2.355

        max_coord_val = d+1
        count = 0
        while max_coord_val > d:
            count += 1
            points2D_cluster = np.random.randn(n_cluster, 2) * sigma + np.array(cluster_pos).reshape(1, 2)
            max_coord_val = points2D_cluster.max()
            if count > 100:
                raise Exception("Generation of cluster points takes too long! Try a smaller cluster size!")

        all_points = np.vstack((points2D, points2D_cluster))
    else:
        all_points = points2D

    grid, grid_idx = cast_points_on_grid(all_points, d, d)
    sum_intensity=grid.sum()

    if dilate_points:
        grid = binary_dilation(grid, iterations=1)
        grid = gaussian(grid, sigma=pixel_sigma)

    grid = grid * sum_intensity/grid.sum()

    assert all_points.shape[0]==n, 'Wrong number of generated points!'
    assert np.round(grid.sum())==n, 'Sum of all pixel intensities should be equal to number of points!'
    
    return all_points, grid
