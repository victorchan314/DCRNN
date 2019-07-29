import argparse
import os
import sys

parent_dir = os.path.abspath("/Users/victorchan/Desktop/UC Berkeley/Research/Code")
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import utils



DETECTOR_DATA_FREQUENCY = dt.timedelta(minutes=5)



def extract_flat_data(timestamps_array, groundtruth_array):
    stretches = utils.get_stretches(timestamps_array[:, 0], DETECTOR_DATA_FREQUENCY)
    timestamps = utils.flatten_circulant_like_matrix_by_stretches(timestamps_array, stretches)
    groundtruth_array_transposed = np.swapaxes(groundtruth_array, 0, 1)
    groundtruth = utils.flatten_circulant_like_matrix_by_stretches(groundtruth_array_transposed, stretches)

    return timestamps, groundtruth

def fit_dates_to_timestamps(full_x, original_x, new_x):
    return new_x[np.isin(full_x, original_x)]

def plot_predictions(y, y_hat, x, timestamps_array, horizon, sensors, horizons=None, save_dir=None, title=None, figsize=None, xlabel="", ylabel="", num_xticks=12, xticks_datetime_precision="D", verbose=0):
    if not isinstance(sensors, list):
        sensors = [sensors]

    for sensor in sensors:
        fig, ax = plt.subplots(figsize=figsize)
        plt.title(title + " Detector {}".format(sensor))
        fig.autofmt_xdate()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        xticks = np.arange(x.shape[0])
        xticks_locs = [xticks[x.shape[0] // num_xticks * i] for i in range(num_xticks + 1)]
        xticks_labels = np.datetime_as_string(x, unit=xticks_datetime_precision)
        xticks_spaced_labels = [xticks_labels[x.shape[0] // num_xticks * i] for i in range(num_xticks + 1)]

        plt.plot(xticks, y[:, sensor], label="Ground Truth")
        plt.xticks(xticks_locs, xticks_spaced_labels)

        cmap = plt.get_cmap('jet')

        horizons = horizons or range(horizon)
        
        for i, h in enumerate(horizons):
            stretches = utils.get_stretches(timestamps_array[:, h], DETECTOR_DATA_FREQUENCY)
            color = cmap(i / len(horizons))

            for start, end in stretches:
                x_stretch = timestamps_array[start:end, h]
                y_hat_stretch = y_hat[h, start:end, sensor]
                x_stretch_range = fit_dates_to_timestamps(x, x_stretch, xticks)

                if start == 0:
                    label = "Horizon {} predictions".format(h)
                else:
                    label = None

                plt.plot(x_stretch_range, y_hat_stretch, label=label, c=color)

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
    predictions_file = np.load(args.predictions)

    # Shape of predictions and groundtruth is (seq_len or horizon, time, sensor)
    predictions_array = predictions_file["predictions"]
    groundtruth_array = predictions_file["groundtruth"]
    timestamps_array = np.load(args.timestamps)["timestamps_y"]
    horizon = args.horizon or predictions_array.shape[0]

    timestamps, groundtruth = extract_flat_data(timestamps_array, groundtruth_array)

    #plot_predictions(groundtruth, predictions, horizon, sensors=[0, 4, 8, 12], timestamps=timestamps, horizons=[0, 4, 8, 11], save_dir=args.graph_save_dir, title="DCRNN Predictions", figsize=(20, 8), xlabel="Time", ylabel="Flow", verbose=verbose)
    plot_predictions(groundtruth, predictions_array, timestamps, timestamps_array, horizon, sensors=[0], horizons=[0, 2, 4, 6, 8, 10], save_dir=args.graph_save_dir, title="DCRNN Predictions", figsize=(20, 8), xlabel="Time", ylabel="Flow", verbose=verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="npz file of predictions")
    parser.add_argument("timestamps", help="npz file of timestamps")
    parser.add_argument("--horizon", help="number of time steps in horizon. If not provided, it will be inferred from the shape of predictions")
    parser.add_argument("--graph_save_dir", help="directory to save graphs to")
    parser.add_argument("-v", "--verbose", action="count", help="verbosity of script")
    args = parser.parse_args()

    main(args)
