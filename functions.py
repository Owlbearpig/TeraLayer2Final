import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey

import pandas as pd
from numpy import nan_to_num
from numpy.fft import fft, ifft, fftfreq
from consts import *
import time
import string
from scipy import signal


def find_files(top_dir=ROOT_DIR, search_str='', file_extension=''):
    results = [Path(os.path.join(root, name))
               for root, dirs, files in os.walk(top_dir)
               for name in files if name.endswith(file_extension) and search_str in str(name)]
    return results


def read_csv(file_path):
    return array(pd.read_csv(file_path, usecols=[i for i in range(5)]))


def avg_runtime(fun, *args, **kwargs):
    repeats = 100
    t0 = time.perf_counter()
    for _ in range(repeats):
        fun(*args, **kwargs)

    print(f'{fun.__name__}: {1e6 * (time.perf_counter() - t0) / repeats} \u03BCs / func. eval. ({repeats} calls)')


def load_ref_file():
    r = read_csv(matlab_data_dir / 'ref_1000x.csv')
    slice_0, slice_1 = settings['data_range_idx']

    return r[:]
    # return r[slice_0:slice_1]


def f_axis():
    r = load_ref_file()
    slice_0, slice_1 = settings['data_range_idx']
    return r[slice_0:slice_1, 0] * MHz


def lam_axis():
    return c0 / f_axis()


def load_files(sample_file_idx=0, data_type='amplitude'):
    slice_0, slice_1 = settings['data_range_idx']

    r = load_ref_file()
    b = read_csv(matlab_data_dir / 'BG_1000.csv')
    s = read_csv(matlab_data_dir / 'Kopf_1x' / f'Kopf_1x_{sample_file_idx:04}')

    if data_type == 'amplitude':
        # return r[:, 1], b[slice_0:slice_1, 1], s[slice_0:slice_1, 1]
        return r[slice_0:slice_1, 1], b[slice_0:slice_1, 1], s[slice_0:slice_1, 1]
    else:  # phase data columns, ref values are also present in each measurement file
        # return s[slice_0:slice_1, 4], b[slice_0:slice_1, 2], s[slice_0:slice_1, 2]
        return s[slice_0:slice_1, 4], b[slice_0:slice_1, 2], s[slice_0:slice_1, 2]


def format_data(mask=None, sample_file_idx=0, verbose=False):
    f = f_axis()
    r, b, s = load_files(sample_file_idx)

    lam = lam_axis()
    rr = r - b
    ss = s - b
    reflectance = ss / rr

    reflectivity = (reflectance ** 2).real  # imaginary part should be 0

    if mask is not None:
        if verbose:
            print(f[mask] / GHz, 'Selected frequencies (GHz)')
        return lam[mask], reflectivity[mask]
    else:
        return lam, reflectivity


def format_data_avg(mask=None, verbose=True):
    if verbose:
        print(f"using average of all {data_file_cnt} data files")

    s_all = []
    for sample_file_idx in range(data_file_cnt):
        _, _, s = load_files(sample_file_idx)
        s_all.append(s)

    s_avg = np.mean(np.array(s_all), axis=0)

    r, b, _ = load_files(0)
    lam, f = lam_axis(), f_axis()

    rr = r - b
    ss = s_avg - b
    reflectance = ss / rr

    reflectivity = (reflectance ** 2).real  # imaginary part should be 0

    if mask is not None:
        if verbose:
            print(f[mask] / GHz, 'Selected frequencies (GHz)')
        return lam[mask], reflectivity[mask]
    else:
        return lam, reflectivity


def residuals(p, fun, x, y0):
    return (fun(x, p) - y0) ** 2


# could be a wrapper
def weighted_residuals(p, fun, x, y0, w):
    return w * residuals(p, fun, x, y0)



def map_maskname(mask):
    mask_map = {'custom_mask_420': custom_mask_420,
                'default_mask': default_mask,
                'full_range_mask_new': full_range_mask_new,
                }
    return mask_map[mask]


def get_phase_measured(sample_file_idx=0, mask=None):
    f = f_axis()
    r, b, s = load_files(sample_file_idx, data_type='phase')

    if mask is not None:
        return f[mask], r[mask], b[mask], s[mask]
    else:
        full_range = (f < 1000 * GHz) * (f > 250 * GHz)
        # full_range = (f > 250 * GHz)
        return f[full_range], r[full_range], b[full_range], s[full_range]


def get_full_measurement(sample_file_idx=0, mask=None, f_slice=None):
    f = f_axis()
    r_pha, b_pha, s_pha = load_files(sample_file_idx, data_type='phase')
    r_amp, b_amp, s_amp = load_files(sample_file_idx, data_type='amplitude')

    r_z, b_z, s_z = r_amp * np.exp(1j * r_pha), b_amp * np.exp(1j * b_pha), s_amp * np.exp(1j * s_pha)

    if mask is not None:
        return f[mask], r_z[mask], b_z[mask], s_z[mask]
    else:
        full_range = (f > -5000 * GHz)
        if f_slice is None:
            full_range = (f <= 1900 * GHz) * (f >= -100 * GHz)
        else:
            full_range = (f >= f_slice[0] * GHz) * (f <= f_slice[1] * GHz)
        return f[full_range], r_z[full_range], b_z[full_range], s_z[full_range]
        # return f[234:-2702], r_z[234:-2702], b_z[234:-2702], s_z[234:-2702]


def get_freq_idx(freqs):
    f = f_axis()
    res = []
    for freq in freqs:
        res.append(np.argmin(np.abs(f / GHz - freq)))

    return res


def std_err(arr, sigma=1):
    arr = np.array(arr)
    return sigma * np.std(arr) / np.sqrt(len(arr))

    std = np.std(arr, axis=0)
    return sigma * std / np.sqrt(np.arange(1, len(arr)+1))

def shift(data_td, shift=0):
    t = data_td[:, 0].real
    dt = np.mean(np.diff(t))

    shift = int(shift / dt)

    ret = data_td.copy()
    ret[:, 1] = np.roll(ret[:, 1], shift)

    return ret


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


def zero_pad(data_fd, mult=6):
    df = np.mean(np.diff(data_fd[:, 0])).real
    f_max = data_fd[-1, 0].real
    eps = 1e-15
    freq_extension = np.arange(f_max, f_max + mult * f_max, df)
    freqs_long = np.concatenate((data_fd[:, 0], freq_extension))
    value_axis_long = np.concatenate((data_fd[:, 1], eps + np.zeros_like(freq_extension)))

    ret = np.array([freqs_long, value_axis_long]).T

    return ret


def filtering(data_td, wn=(0.001, 9.999), filt_type="bandpass", order=5):
    dt = np.mean(np.diff(data_td[:, 0].real))
    fs = 1 / dt

    # sos = signal.butter(N=order, Wn=wn, btype=filt_type, fs=fs, output='sos')
    ba = signal.butter(N=order, Wn=wn, btype=filt_type, fs=fs, output='ba')
    # sos = signal.bessel(N=order, Wn=wn, btype=filt_type, fs=fs, output='ba')
    # data_td_filtered = signal.sosfilt(sos, data_td[:, 1])
    data_td_filtered = signal.filtfilt(*ba, data_td[:, 1])

    data_td_filtered = array([data_td[:, 0], data_td_filtered]).T

    return data_td_filtered


def moving_average(data, window_size=2):
    # Define a kernel for the moving average
    kernel = np.ones(window_size) / window_size

    # Use np.convolve to apply the moving average filter
    result = np.convolve(data, kernel, mode='same')

    return result


def mult_2x2_matrix_chain(arr_in):
    # setup einsum_str (input shape)
    cnt = len(arr_in)
    s0 = string.ascii_lowercase + string.ascii_uppercase
    einsum_str = ''
    for i in range(cnt):
        einsum_str += s0[i] + s0[i + 1] + s0[cnt + 2] + ','

    # remove last comma
    einsum_str = einsum_str[:-1]
    # set output part of einsum_str
    einsum_str += '->' + s0[0] + s0[cnt] + s0[cnt + 2]

    M_out = np.einsum(einsum_str, *arr_in)

    return M_out


def count_minima(y):
    mean_val = np.mean(y)
    y_minima = y[y < mean_val]

    # min distance to last saddle point
    zero_passes = 0
    last_minima = np.inf
    threshold_distance = 0  # min distance between minima
    was_close0 = False
    for idx, isclose0 in enumerate(np.isclose(np.diff(y_minima), 0, atol=2e-7)):
        dist_last_minima = abs(idx - last_minima)
        if isclose0 * (dist_last_minima > threshold_distance) * (not was_close0):
            zero_passes += 1
            print(dist_last_minima)
            last_minima = idx
        if isclose0:
            was_close0 = True
        else:
            was_close0 = False
    print("minima count :", zero_passes)


def noise_gen(freqs, enabled, scale=1, seed=None):
    np.random.seed(seed)

    ret = np.ones_like(freqs)
    if enabled:
        noise = np.random.normal(0, scale, len(freqs))
        ret += noise

    return ret


def gen_p_sols(cnt=100, seed=421, p0_=None, layer_cnt=3, bounds=None):
    np.random.seed(seed)

    if not bounds:
        bounds = [(20, 300), (500, 700), (50, 300)]

    def rand_sol():
        if p0_ is None:
            sol_ = [int(i) for i in [uniform(*bounds[0]), uniform(*bounds[1]), uniform(*bounds[2])]]
        else:
            sol_ = [int(i) for i in [uniform(p0_[j] - 30, p0_[j] + 30) for j in range(3)]]

        try:
            sol_[layer_cnt:] = (len(sol_) - layer_cnt) * [0]
        except IndexError:
            print("Check solution layer count")
            exit("223")

        return sol_

    p_sols = []
    for _ in range(cnt):
        p_sols.append(rand_sol())

    return array(p_sols, dtype=float)


def sell_meier(l, *args):
    l_sqrd = l ** 2
    p = array([*args])
    s = ones(len(l_sqrd))
    for i in range(0, len(p), 2):
        s += p[i] * l_sqrd / (l_sqrd - p[i + 1])
    s = nan_to_num(s)

    return np.sqrt(s)


def unwrap(data_fd, only_ang=False):
    if data_fd.ndim == 2:
        data = nan_to_num(data_fd[:, 1])
    else:
        data = nan_to_num(data_fd)
    if only_ang:
        ret = np.angle(data)
    else:
        ret = np.unwrap(np.angle(data))

    if data_fd.ndim == 2:
        return array([data_fd[:, 0].real, ret]).T
    else:
        return ret


def to_db(data_fd):
    if data_fd.ndim == 2:
        return 20 * np.log10(np.abs(data_fd[:, 1]))
    else:
        return 20 * np.log10(np.abs(data_fd))


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



def phase_correction(data_fd, disable=False, fit_range=None, en_plot=False, extrapolate=False, rewrap=False,
                     ret_fd=False, both=False):
    freqs = data_fd[:, 0].real

    if disable:
        return array([freqs, np.unwrap(np.angle(data_fd[:, 1]))]).T

    phase_unwrapped = unwrap(data_fd)

    if fit_range is None:
        fit_range = [0.40, 0.75]

    fit_slice = (freqs >= fit_range[0]) * (freqs <= fit_range[1])
    p = np.polyfit(freqs[fit_slice], phase_unwrapped[fit_slice, 1], 1)

    phase_corrected = phase_unwrapped[:, 1] - p[1].real

    if en_plot:
        plt.figure("phase_correction")
        plt.plot(freqs, phase_unwrapped[:, 1], label="Unwrapped phase")
        plt.plot(freqs, phase_corrected, label="Shifted phase")
        plt.plot(freqs, freqs * p[0].real, label="Lin. fit (slope*freq)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad)")
        plt.legend()

    if extrapolate:
        phase_corrected = p[0].real * freqs

    if rewrap:
        phase_corrected = np.angle(np.exp(1j * phase_corrected))

    y = np.abs(data_fd[:, 1]) * np.exp(1j * phase_corrected)
    if both:
        return do_ifft(array([freqs, y]).T), array([freqs, y]).T

    if ret_fd:
        return array([freqs, y]).T
    else:
        return array([freqs, phase_corrected]).T


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


if __name__ == '__main__':
    from consts import wide_mask

    # sample_idx = 10
    # lam_w, R_w = format_data(wide_mask, sample_file_idx=sample_idx)
    # print(lam_w, R_w)

    # lam, R = format_data(default_mask, sample_file_idx=sample_idx)
    # print(lam, R)

    # lam, R = format_data(custom_mask_420, sample_file_idx=sample_idx)
    # print(lam, R)

    # lam, R_avg = format_data_avg()
    # plot_R(lam, R_avg)

    # f, r, b, s = get_phase_measured(sample_file_idx=10)

    # print(get_freq_idx([421., 521., 651., 801., 851., 951.]))
    # print(get_freq_idx([300, 351, 500, 600, 800, 951]))
    np.random.seed(123)

    cnt, m = 3, 1
    arr_in = np.random.random((cnt, 2, 2, m))
    A = arr_in[0, :, :, 0]
    B = arr_in[1, :, :, 0]
    C = arr_in[2, :, :, 0]
    out = np.dot(np.dot(A, B), C)
    # print(arr_in[0, :, :, 0])
    print(mult_2x2_matrix_chain(arr_in)[:, :, 0])
    print(out[:, :])
