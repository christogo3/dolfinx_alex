import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import os

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

# Generate random points in 3D space
np.random.seed(0)
points = np.random.rand(30, 3)

# Compute convex hull
hull = ConvexHull(points)

# Plot convex hull
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.plot(points[:, 0], points[:, 1], points[:, 2], 'o')

# Plot convex hull
for simplex in hull.simplices:
    simplex = np.append(simplex, simplex[0])  # Close the loop
    ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'r-')

plt.savefig(os.path.join(script_path, 'test.png'))
