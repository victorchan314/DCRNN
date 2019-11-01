import argparse
import ast
import os
import sys

parent_dir = os.path.abspath("/Users/victorchan/Desktop/UC Berkeley/Research/Code")
sys.path.append(parent_dir)

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from lib import data_utils

DETECTOR_DATA_FREQUENCY = dt.timedelta(minutes=5)



def load_predictions_from_path(path):
    if os.path.isdir(path):
        horizons = []
        predictions_array = []
        groundtruth_array = []

        for d in os.listdir(path):
            parameters = d.split("_")
            for p in parameters:
                if p.startswith("sh"):
                    horizon = int(p[2:])
                    horizons.append(horizon)

                    predictions_path = os.path.join(path, d, "predictions.npz")
                    predictions_file = np.load(predictions_path)
                    predictions_array.append(predictions_file["predictions"])
                    groundtruth_array.append(predictions_file["groundtruth"])

        predictions = np.vstack([p for _, p in sorted(zip(horizons, predictions_array), key=lambda x: x[0])])
        groundtruth = np.vstack([g for _, g in sorted(zip(horizons, groundtruth_array), key=lambda x: x[0])])

        return predictions, groundtruth, sorted(horizons)
    elif os.path.isfile(path):
        predictions_file = np.load(args.predictions)
        predictions = predictions_file["predictions"]
        groundtruth = predictions_file["groundtruth"]

        return predictions, groundtruth, range(1, predictions.shape[0] + 1)
    else:
        raise ValueError("Path is invalid")

def extract_flat_data(timestamps_array, groundtruth_array):
    stretches = data_utils.get_stretches(timestamps_array[:, 0], DETECTOR_DATA_FREQUENCY)
    timestamps = data_utils.flatten_circulant_like_matrix_by_stretches(timestamps_array, stretches)
    groundtruth_array_transposed = np.swapaxes(groundtruth_array, 0, 1)
    groundtruth = data_utils.flatten_circulant_like_matrix_by_stretches(groundtruth_array_transposed, stretches)

    return timestamps, groundtruth

def fit_dates_to_timestamps(full_x, original_x, new_x):
    return new_x[np.isin(full_x, original_x)]

def plot_predictions(y, y_hat, x, timestamps_array, horizon, sensors, horizons=None, by_horizon=True, save_dir=None,
                     title=None, figsize=None, xlabel="", ylabel="", num_xticks=12, xticks_datetime_precision="D", verbose=0):
    if not isinstance(sensors, list):
        sensors = [sensors]

    for sensor in sensors:
        fig, ax = plt.subplots(figsize=figsize)
        plt.title(title + " Detector {}".format(sensor))
        fig.autofmt_xdate()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        xticks = np.arange(x.shape[0])
        xticks_labels = np.datetime_as_string(x, unit=xticks_datetime_precision)
        xticks_locs = [xticks[x.shape[0] // num_xticks * i] for i in range(num_xticks)] + [xticks[-1]]
        xticks_spaced_labels = [xticks_labels[x.shape[0] // num_xticks * i] for i in range(num_xticks)] + [xticks_labels[-1]]

        plt.plot(xticks, y[:, sensor], label="Ground Truth")
        plt.xticks(xticks_locs, xticks_spaced_labels)

        cmap = plt.get_cmap('jet')

        horizons = horizons or range(1, horizon + 1)

        if by_horizon:
            for i, h in enumerate(horizons):
                stretches = data_utils.get_stretches(timestamps_array[:, h - 1], DETECTOR_DATA_FREQUENCY)
                color = cmap(i / len(horizons))

                for start, end in stretches:
                    x_stretch = timestamps_array[start:end, h - 1]
                    y_hat_stretch = y_hat[h - 1, start:end, sensor]
                    x_stretch_range = fit_dates_to_timestamps(x, x_stretch, xticks)

                    if start == 0:
                        label = "Horizon {} predictions".format(h)
                    else:
                        label = None

                    plt.plot(x_stretch_range, y_hat_stretch, label=label, c=color, alpha=0.3)

            plt.legend(loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0.)
            plt.show()
        else:
            stretches = data_utils.get_stretches(timestamps_array[:, 0], DETECTOR_DATA_FREQUENCY)
            for i, (start, end) in enumerate(stretches):
                for t in range(start, end):
                    color = cmap((t - start) / (end - start - 1))
                    x_stretch = timestamps_array[t, :]
                    y_hat_stretch = y_hat[:, t, sensor]
                    x_stretch_range = fit_dates_to_timestamps(x, x_stretch, xticks)

                    if start == 0:
                        label = "Time {} predictions".format(t)
                    else:
                        label = None

                    plt.plot(x_stretch_range, y_hat_stretch, label=label, c=color, alpha=0.3)

            plt.legend(loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0.)
            plt.show()

def plot_predictions_all_sensors(y, y_hat, horizon, timestamps=None, horizons=None, save_dir=None, title=None, figsize=None, verbose=0):
    nrows = int(np.floor(np.sqrt(y.shape[2])))
    ncols = int(np.ceil(y.shape[2] / nrows))
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    plt.title(title)

    groundtruth = np.hstack((y[0, :, 0], y[-1, -horizon + 1:, 0]))
    if timestamps is not None:
        x = np.hstack((timestamps.T[0, :], timestamps.T[-1, -horizon + 1:]))
    else:
        x = np.arange(len(groundtruth))

    plt.plot(x, groundtruth, label="Ground Truth")

    horizons = horizons or range(horizon)
    
    for row in range(nrows):
        for col in range(ncols):
            plt.subplot(nrows, ncols, nrows*row + col + 1)
            if row == nrows - 1:
                plt.xticks(rotation = 60)
            else:
                plt.xticks([])

            for h in horizons:
                plt.plot(timestamps[:, h], y_hat[h, :, 0], label="Horizon {} predictions".format(h))

            if row == 0 and col == ncols - 1:
                plt.legend(loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0.)

    plt.show()



def main(args):
    verbose = args.verbose or 0

    # Shape of predictions and groundtruth is (seq_len or horizon, time, sensor)
    predictions_array, groundtruth_array, horizons = load_predictions_from_path(args.predictions)
    timestamps_array = np.load(args.timestamps)["timestamps_y"]

    horizon = int(args.horizon or timestamps_array.shape[1])
    sensors = ast.literal_eval(args.sensors) if args.sensors else range(predictions_array.shape[2])
    horizons = ast.literal_eval(args.horizons) if args.horizons else horizons
    by_horizon = not args.by_time

    timestamps_array = data_utils.pad_array(timestamps_array, horizon)
    groundtruth_array = data_utils.pad_array(groundtruth_array, horizon)

    timestamps, groundtruth = extract_flat_data(timestamps_array, groundtruth_array)

    plot_predictions(groundtruth, predictions_array, timestamps, timestamps_array, horizon,
                     sensors=sensors, horizons=horizons, by_horizon=by_horizon, save_dir=args.graph_save_dir,
                     title="DCRNN Predictions", figsize=(20, 8), xlabel="Time", ylabel="Flow", verbose=verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="If file, npz file of predictions. If directory, then parent directory of single horizon data")
    parser.add_argument("timestamps", help="npz file of timestamps")
    parser.add_argument("--horizon", help="number of time steps in horizon. If not provided, it will be inferred from the shape of predictions")
    parser.add_argument("-s", "--sensors", help="sensors to plot graphs for")
    parser.add_argument("--horizons", help="horizons to plot on each graph")
    parser.add_argument("--by_time", action="store_true", help="plot predictions based on horizon (default) or by timestamp")
    parser.add_argument("--graph_save_dir", help="directory to save graphs to")
    parser.add_argument("-v", "--verbose", action="count", help="verbosity of script")
    args = parser.parse_args()

    main(args)
