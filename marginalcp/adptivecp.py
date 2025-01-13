import numpy as np

import torch
import scipy.special
import math
import alphashape
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def score_L2(vector1, vector2):
    diff = (vector1 - vector2) ** 2
    if diff.dim() >1:
        sum_squared_diff = torch.sum(diff, dim=1)
    else:
        sum_squared_diff = diff

    # Compute the square root to get the Euclidean distance
    distances = torch.sqrt(sum_squared_diff)

    return distances


def normalized_squared_euclidean_distance(x, y):
    # Center the tensors by subtracting their mean
    x_centered = x - x.mean()
    y_centered = y - y.mean()

    # Compute the squared Euclidean distance between the centered tensors
    numerator = torch.sum((x_centered - y_centered) ** 2)

    # Compute the sum of squared norms of the centered tensors
    denominator = torch.sum(x_centered ** 2) + torch.sum(y_centered ** 2)

    # Compute the normalized squared Euclidean distance
    distance = 0.5 * numerator / denominator
    return distance

def generate_points_within_boundary(num_points, bounds):
    lower_bounds = torch.floor(bounds[0].squeeze())
    upper_bounds = torch.ceil(bounds[1].squeeze())
    lengths = upper_bounds - lower_bounds
    volume = torch.prod(lengths)

    # Generate random points in [0, 1] for each dimension and scale to bounds
    random_points = torch.from_numpy(np.random.uniform(0, 1, (num_points, len(bounds[0])))).to(lower_bounds.device)
    points = lower_bounds + random_points * (upper_bounds - lower_bounds)
    return points, volume



