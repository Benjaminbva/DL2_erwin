import numpy as np
from balltree import build_balltree
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

bs, num_points, dim = 16, 8, 3


points = torch.rand(num_points * bs, dim, dtype=torch.float32, device='cuda')
batch_idx = torch.repeat_interleave(torch.arange(bs, device='cuda'), num_points)
tree_idx, tree_mask = build_balltree(points, batch_idx)
grouped_points = points[tree_idx]

level_to_node_size = lambda level: 2**(level)

for level in range(0, 6):
    groups = grouped_points.reshape(-1, level_to_node_size(level), dim)
    num_groups = groups.shape[0]

    x = groups[:, :, 0].cpu()
    y = groups[:, :, 1].cpu()
    z = groups[:, :, 2].cpu()

    colors = cm.get_cmap('nipy_spectral', num_groups)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(num_groups):
        ax.scatter(x[i], y[i], z[i], color=colors(i), s=10)

    ax.set_title(f'3D Visualization - Level {level} - {num_groups} groups')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
