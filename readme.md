# GPS Aided Beam Prediction and Tracking for UAV mmWave Communication

This is a research code repository for GPS-aided beam prediction and tracking for mmWave communication.

In this work, there are 3 folders inside folder `notebooks`:
1. Folder `minmaxgeo_uebsvector` contains the proposed model for beam prediction and tracking that uses min-max normalized UAV's geodetic position (latitude and longitude) and UAV-BS unit vector as input. This folder contains the code implementation
of this work: [GPS-Aided Deep Learning for Beam Prediction and Tracking in UAV mmWave Communication](https://arxiv.org/abs/2505.17530).
2. Folder `uebsvector_logscaledheight` contains the proposed model for beam prediction and tracking that uses UAV-BS unit vector and log scaled UAV's height as input.
3. Folder `baseline` contains the baseline models for beam prediction and tracking as described in [3] (NOTE: these implementations are not the official code from the original authors).

## Notebook Content:

1. `00_test_drone_cnn_ed_rnn_experiment` and `01_drone_cnn_ed_rnn_experiment.ipynb`: a notebook to train the proposed model.
2. `02_visualization_combination.ipynb` and `03_visualization.ipynb`: a notebook to visualize the evaluation metrics.
3. `00_test_onnx.ipynb`: a notebook to measure the inference time using PyTorch model and ONNX model.
4. `00_test_dataset_label.ipynb`: visualize label distribution using various data set splitting methods.
5. `00_test_drone_base_prediction.ipynb`: train a baseline beam prediction model[3].
6. `00_test_drone_base_tracking.ipynb`: train a baseline beam tracking model[3].

## How to run the code
1. Clone this repo.
2. Enter the repo directory through the terminal.
3. Run `poetry install` to install the dependencies (run `pip install poetry` if you haven't installed poetry yet).
4. Run `poetry update` to update the libraries version.
5. Run `poetry shell` to activate the virtual environment.
6. Download end extract the zip file data set Scenario 23 from [DeepSense6G website](https://www.deepsense6g.net) [1].
7. Create folder `data/raw/` inside the repo.
8. Put the data set into folder `data/raw/`. The dataset folder should be named in the format of `Scenario{scenario_number}`. This folder should contains `scenario{scenario_number}.csv` file.
9. To run the jupyter notebook, open the jupyter notebook and make sure to set the kernel to the poetry environment.
10. Inside the notebook, make sure to change the repository path to the correct path (e.g. `sys.path.append('F:/repo/gpsbeam')`).
11. Run the notebook.

## How to Reproduce the Result

1. To reproduce the proposed model result, run `01_drone_cnn_ed_rnn_experiment.ipynb`.
2. Visualize the result by running `02_visualization_combination.ipynb` and `02_visualization.ipynb`.

## Additional Information
- The preprocessed dataset will be saved in `data/processed/`.
- Experiment result will be saved in `data/experiment_result/`.

## Thanks to
This repository is inspired by the following repositories: 
- [Position-Beam-Prediction](https://github.com/jmoraispk/Position-Beam-Prediction)
- [lidar_beam_tracking](https://github.com/acyiobs/lidar_beam_tracking)
- [Vision-Position-Beam-Prediction](https://github.com/gourangc/Vision-Position-Beam-Prediction)

Thanks to the tutorials from the following sources:
- [Deepsense Workshop Spring 2024](https://www.deepsense6g.net/deepsense6g-workshop-spring-2024/)
- [DeepSense6G Tutorials](https://www.deepsense6g.net/tutorials/)

## Reference:

[1] A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, and N. Srinivas, “DeepSense 6G: large-scale real-world multi-modal sensing and communication datasets,” to be available on arXiv, 2022. [Online]. Available: https://www.DeepSense6G.net

[2] Charan, G., Hredzak, A., Stoddard, C., Berrey, B., Seth, M., Nunez, H., & Alkhateeb, A. (2022, December). Towards real-world 6G drone communication: Position and camera aided beam prediction. In GLOBECOM 2022-2022 IEEE Global Communications Conference (pp. 2951-2956). IEEE.

[3] Charan, G., & Alkhateeb, A. (2024). Sensing-Aided 6G Drone Communications: Real-World Datasets and Demonstration. arXiv preprint arXiv:2412.04734. [Online].Available: https://arxiv.org/abs/2412.04734
