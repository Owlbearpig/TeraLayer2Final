import matplotlib.pyplot as plt
import numpy as np
from numpy import nan_to_num, array
from numpy.fft import ifft
from consts import *
from scipy import signal


def do_fft(data_td, shift=None):
    t, y = data_td[:, 0].real, data_td[:, 1].real
    n = len(y)
    dt = float(np.mean(np.diff(t)))
    Y = np.conj(np.fft.fft(y, n))

    # Y = np.fft.fft(y, n)
    f = np.fft.fftfreq(len(t), dt)

    if shift is not None:
        shift = int(shift / dt) * dt
        Y = Y * np.exp(1j * shift * 2 * pi * f)

    idx_range = (f >= 0)
    # return array([f, Y]).T
    return array([f[idx_range], Y[idx_range]]).T

def do_rfft(data_td):
    data_td = nan_to_num(data_td)

    dt = float(np.mean(np.diff(data_td[:, 0])))
    freqs, data_fd = np.fft.rfftfreq(n=len(data_td[:, 0]), d=dt), np.conj(np.fft.rfft(data_td[:, 1]))

    return array([freqs, data_fd]).T

def do_irfft(data_fd):
    f_axis, y_fd = data_fd[:, 0].real, data_fd[:, 1]

    y_td = np.fft.irfft(np.conj(y_fd))
    df = np.mean(np.diff(f_axis))
    n = len(y_td)
    t = np.arange(0, n) / (n * df)

    return array([t, y_td]).T

def do_ifft(data_fd, hermitian=True, shift=0, flip=False):
    freqs, y_fd = data_fd[:, 0].real, data_fd[:, 1]
    y_fd = nan_to_num(y_fd)

    if hermitian:
        # y_fd = np.concatenate((np.conj(y_fd), np.flip(y_fd[1:])))
        y_fd = np.concatenate((y_fd, np.flip(np.conj(y_fd[1:]))))
        """
        * ``a[0]`` should contain the zero frequency term,
        * ``a[1:n//2]`` should contain the positive-frequency terms,
        * ``a[n//2 + 1:]`` should contain the negative-frequency terms, in
          increasing order starting from the most negative frequency.
        """

    y_td = ifft(y_fd)

    df = np.mean(np.diff(freqs))
    n = len(y_td)
    t = np.arange(0, n) / (n * df)

    # t = np.linspace(0, len(y_td)*df, len(y_td))
    # t += 885

    if flip:
        y_td = np.flip(y_td)

    dt = np.mean(np.diff(t))
    shift = int(shift / dt)
    y_td = np.roll(y_td, shift)

    return array([t, y_td]).T


def moving_average(data, window_size=2):
    # Define a kernel for the moving average
    kernel = np.ones(window_size) / window_size

    # Use np.convolve to apply the moving average filter
    result = np.convolve(data, kernel, mode='same')

    return result


def window(data_td, win_width=None, win_start=None, en_plot=False, slope=0.15, label=""):
    t, y = data_td[:, 0].real, data_td[:, 1].real

    pulse_width = 10  # ps
    dt = np.mean(np.diff(t))

    # shift = int(100 / dt)

    # y = np.roll(y, shift)

    if win_width is None:
        win_width = int(pulse_width / dt)
    else:
        win_width = int(win_width / dt)

    if win_width > len(y):
        win_width = len(y)

    if win_start is None:
        win_center = np.argmax(np.abs(y))
        win_start = win_center - int(win_width / 2)
    else:
        win_start = int(win_start / dt)

    if win_start < 0:
        win_start = 0

    pre_pad = np.zeros(win_start)
    # window_arr = signal.windows.hamming(win_width)
    # window_arr = signal.windows.hanning(win_width)
    # window_arr = signal.windows.triang(win_width)
    window_arr = signal.windows.tukey(win_width, slope)

    post_pad = np.zeros(len(y) - win_width - win_start)

    window_arr = np.concatenate((pre_pad, window_arr, post_pad))

    y_win = y * window_arr

    if en_plot:
        plt.figure("Windowing")
        plt.plot(t, y, label=f"{label} before windowing")
        plt.plot(t, np.max(np.abs(y)) * window_arr, label="Window")
        plt.plot(t, y_win, label=f"{label} after windowing")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (nA)")
        plt.legend()

    # y_win = np.roll(y_win, -shift)

    return np.array([t, y_win], dtype=float).T


def x_ax_crossing(data_td):
    t, y = data_td[:, 0], data_td[:, 1]

    start_idx = np.argmax(np.abs(y))

    turning_idx = 0
    for idx in range(start_idx, len(y)):
        if not int(np.sign(y[idx-1]) + np.sign(y[idx])):
            turning_idx = idx
            break

    y1, y2 = y[turning_idx - 1], y[turning_idx]
    t1, t2 = t[turning_idx - 1], t[turning_idx]

    t0 = (y1 * t2 - t1 * y2) / (y1 - y2)

    return t0
