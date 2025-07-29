import argparse
import numpy as np
from tqdm import tqdm
import datoviz as dvz

from ellipsoid_py import Ellipsoid3d as Ellipsoid
from ellipsoid_py import EllipsoidVec3d as EllipsoidVec
from dvz_layer_manager import PanelManager, vector_field_visuals

parser = argparse.ArgumentParser()

parser.add_argument(
    "--res",
    type=int,
    default=200,
    help="Resolution for the grid of query points.",
)
args = parser.parse_args()

res = args.res

# Define the range of query points based on the dataset
xyz_range = None
xyz_range = np.linspace(np.array([-2.5, -3.0, 0.1]), np.array([0.0, 0.5, 1.8]), num=res)
z_level = 85
query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)
qp = np.ascontiguousarray(query_points[:, :, z_level, :].reshape((res * res, 3)))

# Set up the view parameters for visualization
view_params = {}
view_params["initial"] = (1.46818, -4.70855, 2.94219)
view_params["initial_lookat"] = (
    -1.471043381764978,
    -1.9081805440813935,
    0.65445400820226307,
)
view_params["initial_up"] = (0, 0, 1)

# Load npz file for the gmm
dataset = "livingroom1"
bandwidth = 0.02
gmm = np.load(f"./data/{dataset}_gmm_final_{bandwidth}.npz")

# Extract the necessary components from the GMM
means = gmm["means"]
covariances = gmm["covariances"]
n_components = gmm["n_components"]

# Create ellipsoids from the GMM parameters
max_nsig = 2.5  # Maximum number of standard deviations for scaling
ellipsoids = EllipsoidVec()
for i in tqdm(range(n_components), desc="Creating Ellipsoids"):
    mean = means[i]
    covariance = covariances[i].reshape((3, 3))
    prec = np.linalg.inv(covariance)
    L, P = np.linalg.eig(prec)
    scale_mat = np.diag(np.sqrt(L)) / max_nsig
    ellipsoids.add(mean, P @ scale_mat @ scale_mat.T @ P.T)

# Moving ellipsoid
a2 = np.array([0.1, 0.2, 0.3])
T2 = np.eye(4)
T2[:3, :3] = np.array(
    [
        [0.7071068, -0.7071068, 0.0000000],
        [0.7071068, 0.7071068, 0.0000000],
        [0.0000000, 0.0000000, 1.0000000],
    ]
)
noise = 0.04
that_cov = np.array([[noise, 0.0, 0.0], [0.0, noise, 0.0], [0.0, 0.0, noise]])
distances = np.full(len(qp), np.nan)
grads = np.full((len(qp), 3), np.nan)
coll_probs = np.full(len(qp), np.nan)
for i in tqdm(
    range(len(qp)),
    desc=f"Processing {len(qp)} positions",
):
    T2[:3, 3] = qp[i, :]
    E2 = Ellipsoid(T2[:3, 3], a2, T2[:3, :3])
    success, dist, grad_unorm, coll_prob = ellipsoids.dgcp_to(E2, that_cov)
    if success:
        distances[i] = dist
        grads[i] = grad_unorm / np.linalg.norm(grad_unorm)
        coll_probs[i] = coll_prob

# validity masks
valid_dist = ~np.isnan(distances)
valid_coll_prob = ~np.isnan(coll_probs)

# compute colors
min_dist = np.min(distances[valid_dist])
max_dist = np.max(distances[valid_dist])
min_prob = np.min(coll_probs[valid_coll_prob])
max_prob = np.max(coll_probs[valid_coll_prob])
colors_dist = dvz.cmap("cividis", distances[valid_dist], vmin=min_dist, vmax=max_dist)
colors_prob = dvz.cmap(
    "reds", coll_probs[valid_coll_prob], vmin=min_prob, vmax=max_prob
)

# Visualization
app = dvz.App()
figure = app.figure(width=1920, height=1080, gui=True)
panel = figure.panel()
panel.camera(
    initial=view_params["initial"],
    initial_lookat=view_params["initial_lookat"],
    initial_up=view_params["initial_up"],
)
# panel.orbit(center=view_params["initial_lookat"], axis=view_params["initial_up"], period=20)

pm = PanelManager(panel)
pm.init_layer("distance")
pm.init_layer("gradient")
pm.init_layer("collision probability")

sc = dvz.ShapeCollection()
for i, ellipsoid in enumerate(tqdm(ellipsoids, desc="Visualizing Ellipsoids")):
    sc.add_sphere(
        scale=tuple(2 * ellipsoid.axes_lengths()),
        transform=ellipsoid.tf(),
        color=(255, 255, 255, 255),
    )
pm.add_visual("gmm", app.mesh(sc, lighting=True, depth_test=True))
pm.add_to_layer("distance", "gmm")
pm.add_to_layer("gradient", "gmm")
pm.add_to_layer("collision probability", "gmm")

# heatmap visualizations
pm.add_visual(
    "distance_heatmap",
    app.basic(
        "point_list",
        position=qp[valid_dist, :],
        color=colors_dist,
        depth_test=True,
        size=10.0,
    ),
)
pm.add_to_layer("distance", "distance_heatmap")

pm.add_visual(
    "coll_prob_heatmap",
    app.basic(
        "point_list",
        position=qp[valid_coll_prob, :],
        color=colors_prob,
        depth_test=True,
        size=10.0,
    ),
)
pm.add_to_layer("collision probability", "coll_prob_heatmap")

# Default colorbar is for distances
colorbar = figure.colorbar(cw=100)
colorbar.set_cmap("cividis")
colorbar.set_range(min_dist, max_dist)

# Set up visual for the gradient
arrow_shafts, arrow_shaft_colors, arrow_heads, arrow_head_colors = vector_field_visuals(
    qp, grads, mask=valid_dist
)
pm.add_visual(
    "arrow_shaft_visual",
    app.basic(
        "line_list", position=arrow_shafts, color=arrow_shaft_colors, depth_test=True
    ),
)
pm.add_visual(
    "arrow_head_visual",
    app.basic(
        "triangle_list",
        position=arrow_heads,
        color=arrow_head_colors,
        depth_test=True,
    ),
)
pm.add_to_layer("gradient", "arrow_shaft_visual")
pm.add_to_layer("gradient", "arrow_head_visual")

pm.show_layer("distance")  # Defaults to distance layer

dropdown_selected = dvz.Out(0)


@app.connect(figure)
def on_gui(ev):
    dvz.gui_pos(dvz.vec2(25, 25), dvz.vec2(0, 0))
    dvz.gui_size(dvz.vec2(150, 40))

    dvz.gui_begin("Hidden Title", 1)

    if dvz.gui_dropdown(
        "Mode", 3, ["distance", "gradient", "collision probability"], dropdown_selected, 0
    ):
        if dropdown_selected.value == 0:
            pm.show_layer("distance")
            colorbar.show(True)
            colorbar.set_cmap("cividis")
            colorbar.set_range(min_dist, max_dist)
            figure.update()
        elif dropdown_selected.value == 1:
            pm.show_layer("gradient")
            colorbar.show(False)
            figure.update()
        elif dropdown_selected.value == 2:
            pm.show_layer("collision probability")
            colorbar.show(True)
            colorbar.set_cmap("reds")
            colorbar.set_range(min_prob, max_prob)
            figure.update()

    dvz.gui_end()


app.run()
app.destroy()
