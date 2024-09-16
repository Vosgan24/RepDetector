import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks


def load_imu_data(file_path):
    acc_x = []
    acc_y = []
    acc_z = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        i = 0
        while i < len(lines):
            if lines[i].strip():
                acc_x.append(float(lines[i].strip()))
                acc_y.append(float(lines[i + 1].strip()))
                acc_z.append(float(lines[i + 2].strip()) - 1.0)
                i += 4
            else:
                i += 1

    imu_df = pd.DataFrame({
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z
    })

    return imu_df

def normalize_data(df):
    baseline = (df['acc_x'][0] + df['acc_y'][0] + df['acc_z'][0]) / 3

    df['acc_x'] = df['acc_x'] - baseline
    df['acc_y'] = df['acc_y'] - baseline
    df['acc_z'] = df['acc_z'] - baseline

    return df

def calculate_combined_acc(df):
    df['abs_acc_x'] = df['acc_x'].abs()
    df['abs_acc_y'] = df['acc_y'].abs()
    df['abs_acc_z'] = df['acc_z'].abs()

    df['combined_acc'] = df['abs_acc_x'] + df['abs_acc_y'] + df['abs_acc_z']

    return df

def plot_smoothed_combined_acc_with_peaks(df):
    combined_acc = df['combined_acc'].values
    x_values = np.arange(len(combined_acc))

    cubic_spline = CubicSpline(x_values, combined_acc)

    x_fine = np.linspace(0, len(combined_acc) - 1, 500)
    y_fine = cubic_spline(x_fine)

    peaks, properties = find_peaks(
        y_fine,
        prominence=0.3,
        width=3,
        rel_height=0.5
    )

    plt.figure(figsize=(10, 6))
    plt.plot(x_fine, y_fine, label='Smoothed Combined Acceleration (Cubic Spline)', color='r')
    plt.plot(x_fine[peaks], y_fine[peaks], 'ro', label='Detected Peaks')

    plt.xlabel('Frame')
    plt.ylabel('Acceleration')
    plt.title('Smoothed Combined Absolute Acceleration with Detected Peaks')
    plt.legend()
    plt.grid(True)
    plt.show()

    num_reps = len(peaks)
    print(f"Number of detected repetitions: {num_reps}")


file_path = './labTest0912.csv'
imu_data = load_imu_data(file_path)
imu_data = normalize_data(imu_data)
imu_data = calculate_combined_acc(imu_data)
plot_smoothed_combined_acc_with_peaks(imu_data)
