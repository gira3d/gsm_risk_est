# GSM Risk Estimation

This package provides tools for estimating distance, gradient, and collision probability from Gaussian Surface Models (GSM). It leverages the `ellipsoid_utils` library to represent the environment as a collection of ellipsoids derived from a Gaussian Mixture Model (GMM) and performs efficient risk-related computations.

## Features

- **GMM Representation:** Loads pre-trained Gaussian Mixture Models that represent a 3D environment.
- **Ellipsoidal Environment:** Converts GMM components into a vector of ellipsoids for efficient geometric processing.
- **Risk Estimation:** Calculates the minimum distance, gradient, and collision probability from a query ellipsoid (e.g., a robot) to the entire environment model.
- **Visualization:** Includes an example to compute and visualize risk-based heatmaps over a 3D scene using `datoviz`.

## Dependencies

- C++17 compiler
- CMake (>= 3.15)
- `ellipsoid_utils` package from this workspace.
- Python (>= 3.8)
- colcon (for building)
- Python packages: `numpy`, `tqdm`, `datoviz`

## Building

This package is part of the `gira3d-map-ops` meta-package and is designed to be built with `colcon`. Please refer to the top-level `README.md` for build instructions.

## Example

The `examples` directory contains a Python script to demonstrate the functionality of the package.

`dist_coll_prob_from_gmm.py`: This script loads a GMM representing a 3D scene (e.g., a living room). It then computes the distance and collision probability from a moving query ellipsoid to the scene across a grid of points. The results are visualized as 3D heatmaps, showing safe and high-risk areas.

### Running the Example

After building the workspace, you can run the example from the root of the workspace:

```bash
python wet/src/gsm_risk_est/examples/dist_coll_prob_from_gmm.py
```

The script will generate visualizations of the distance field, gradient field, and collision probability field within the environment.

<img width="2032" height="1167" alt="Screenshot 2025-07-29 at 13 29 50" src="https://github.com/user-attachments/assets/7448b490-168c-4d06-b734-cf99df711e86" />
<img width="2032" height="1167" alt="Screenshot 2025-07-29 at 13 29 57" src="https://github.com/user-attachments/assets/5c0239a3-600a-430e-91f3-196c054f50c4" />
<img width="2032" height="1167" alt="Screenshot 2025-07-29 at 13 30 02" src="https://github.com/user-attachments/assets/ce4379b0-c92a-4039-a70b-87ec26567952" />


## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

## Paper
A technical introduction to the theory behind this work is provided in our paper, available [here](https://www.kshitijgoel.com/goel-distance-2025/index.html).

```
@inproceedings{Goel_Distance_and_Collision_2025,
author = {Goel, Kshitij and Tabib, Wennie},
title = {{Distance and Collision Probability Estimation from Gaussian Surface Models}},
journal = {IEEE/RSJ International Conference on Intelligent Robots and Systems},
year = {2025}
}
```

## Acknowledgements
This work was supported in part by an Uber Presidential Fellowship. This
material is based upon work supported by, or in part by, the Army Research
Laboratory and the Army Research Office under contract/grant number
W911NF-25-2-0153.
