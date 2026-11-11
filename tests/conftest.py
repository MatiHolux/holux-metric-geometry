import numpy as np
import open3d as o3d
import pytest


@pytest.fixture
def synthetic_ply(tmp_path):
    """Piso (z=0) + pared (x=0) + 300 puntos de ruido disperso."""
    rng = np.random.default_rng(0)
    floor = np.column_stack([rng.uniform(0, 5, 20000), rng.uniform(0, 5, 20000), rng.normal(0, 0.002, 20000)])
    wall = np.column_stack([rng.normal(0, 0.002, 15000), rng.uniform(0, 5, 15000), rng.uniform(0, 3, 15000)])
    noise = rng.uniform([-1, -1, -1], [6, 6, 4], size=(300, 3))
    pts = np.vstack([floor, wall, noise])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rng.uniform(0, 1, pts.shape))
    path = tmp_path / "synthetic.ply"
    o3d.io.write_point_cloud(str(path), pcd)
    return path
