from helpers import plt_show
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_settings import mpl_style_params
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter

font_size = 26

mpl.rcParams = mpl_style_params()

base_dir = Path(__file__).parent / "res_files"

def parse_result_dir(path_):
    # csv header
    """
    quant = #idx, p0x0, p0x1, p0y, x0, x1, x2, y, f1_neval, f2_neval,
    p0_grid_runtime (us), f1_time (us), f2_time (us), total_time (us)
    res_arr_2d = (max_eval, sweep_idx, quant)

    quant = #idx, x0, x1, x2, y, f_neval, total_time (us)
    res_arr_3d = (max_eval, sweep_idx, quant)
    """

    def read_res_file(f_):
        df = pd.read_csv(f_, sep=", ", decimal=".", engine="python")
        return df.values

    p = Path(path_)

    result_files = [file for file in p.iterdir() if "result" in str(file.name)]
    print(result_files)
    res_arr_3d_ = []
    res_arr_2d_ = []
    for file in result_files:
        max_eval = int(str(file.stem).split("_")[-1])
        if max_eval <= 4:
            continue

        if "full_opt" in str(file):
            res_arr_3d_.append((max_eval, read_res_file(file)))
        else:
            res_arr_2d_.append((max_eval, read_res_file(file)))

    res_arr_2d_sorted = sorted(res_arr_2d_, key=lambda x: x[0])
    res_arr_3d_sorted = sorted(res_arr_3d_, key=lambda x: x[0])

    # throw away the max_eval value (it"s == 0 idx since the array is sorted)
    res_arr_2d_ = np.array([a[1] for a in res_arr_2d_sorted])
    res_arr_3d_ = np.array([a[1] for a in res_arr_3d_sorted])

    return res_arr_2d_, res_arr_3d_


def rel_deviation_v_runtime():

    def plot_data(path_, ax0_, ax1_):
        res_arr_2d, res_arr_3d = parse_result_dir(path_)

        idx_offset = res_arr_2d.shape[-1] == 13

        # baseline_2d = np.tile(np.array([70.279181786, 663.39110262, 37.347813018000004]), (1000, 1))
        # baseline_2d = res_arr_2d[-1, :, 4:7]
        baseline_2d = res_arr_2d[-1, :, 3:6]

        # baseline_3d = np.tile(np.array([70.279181786, 663.39110262, 37.347813018000004]), (1000, 1))
        # baseline_3d = res_arr_3d[-1, :, 1:4]
        baseline_3d = res_arr_3d[-1, :, 1:4]

        deviation_2d_rel = np.sqrt(np.diff(res_arr_2d[:, :, 4-idx_offset], axis=0) ** 2 +
                                   np.diff(res_arr_2d[:, :, 5-idx_offset], axis=0) ** 2 +
                                   np.diff(res_arr_2d[:, :, 6-idx_offset], axis=0) ** 2)
        deviation_3d_rel = np.sqrt(np.diff(res_arr_3d[:, :, 1], axis=0) ** 2 +
                                   np.diff(res_arr_3d[:, :, 2], axis=0) ** 2 +
                                   np.diff(res_arr_3d[:, :, 3], axis=0) ** 2)

        deviation_2d = np.sqrt((baseline_2d[:, 0] - res_arr_2d[:, :, 4-idx_offset]) ** 2 +
                               (baseline_2d[:, 1] - res_arr_2d[:, :, 5-idx_offset]) ** 2 +
                               (baseline_2d[:, 2] - res_arr_2d[:, :, 6-idx_offset]) ** 2)
        deviation_3d = np.sqrt((baseline_3d[:, 0] - res_arr_3d[:, :, 1]) ** 2 +
                               (baseline_3d[:, 1] - res_arr_3d[:, :, 2]) ** 2 +
                               (baseline_3d[:, 2] - res_arr_3d[:, :, 3]) ** 2)
        print(baseline_2d[:, 0], res_arr_2d[:, :, 4-idx_offset])
        deviation_2d_avg = np.mean(deviation_2d, axis=1)
        deviation_3d_avg = np.mean(deviation_3d, axis=1)

        deviation_2d_rel_avg = np.mean(deviation_2d_rel, axis=1)
        deviation_3d_rel_avg = np.mean(deviation_3d_rel, axis=1)

        run_time_avg_2d = (np.mean(res_arr_2d[:, :, -1], axis=1) / 1e3)
        run_time_avg_3d = (np.mean(res_arr_3d[:, :, -1], axis=1) / 1e3)

        # print(run_time_avg_2d, np.std(res_arr_2d[:, :, -1], axis=1))

        selected_max_eval = res_arr_2d.shape[0]-1
        #print(np.sort(res_arr_2d[selected_max_eval, :, 4 - idx_offset])[:10])
        #print(np.sort(res_arr_2d[selected_max_eval, :, 5 - idx_offset])[:10])
        #print(np.sort(res_arr_2d[selected_max_eval, :, 6 - idx_offset])[-10:])
        # print(np.max(res_arr_2d[selected_max_eval, :, 5-idx_offset]))
        # 2d
        mean = [np.mean(res_arr_2d[selected_max_eval, :, 4 + i-idx_offset]) for i in range(3)]
        std = [np.std(res_arr_2d[selected_max_eval, :, 4 + i-idx_offset]) for i in range(3)]
        y_val = np.mean(res_arr_2d[selected_max_eval, :, 7-idx_offset])
        runtime_mean = np.mean(res_arr_2d[selected_max_eval, :, 13-idx_offset])
        runtime_std = np.std(res_arr_2d[selected_max_eval, :, 13-idx_offset])
        print(mean, y_val, "\n", std)
        print(runtime_mean, runtime_std)

        # 3d
        mean = [np.mean(res_arr_3d[selected_max_eval, :, 1 + i]) for i in range(3)]
        std = [np.std(res_arr_3d[selected_max_eval, :, 1 + i]) for i in range(3)]
        y_val = np.mean(res_arr_3d[selected_max_eval, :, 4])
        runtime_mean = np.mean(res_arr_3d[selected_max_eval, :, 6])
        runtime_std = np.std(res_arr_3d[selected_max_eval, :, 6])
        print(mean, y_val, "\n", std)
        print(runtime_mean, runtime_std)

        """
        print(run_time_avg_2d)
        print(deviation_2d_avg)
        print(np.mean(res_arr_2d[:, :, 7], axis=1))
        print(np.mean(res_arr_2d[:, :, 8], axis=1))
        print(np.mean(res_arr_3d[:, :, 5], axis=1))
        """

        ax0_.scatter(run_time_avg_2d[:], deviation_2d_avg[:], s=10, zorder=1)
        ax0_.scatter(run_time_avg_3d[:], deviation_3d_avg[:], s=10, zorder=2)

        ax1_.scatter(run_time_avg_2d[1:], deviation_2d_rel_avg[:], s=10, zorder=3)
        ax1_.scatter(run_time_avg_3d[1:], deviation_3d_rel_avg[:], s=10, zorder=4)

    p_6f = base_dir / "6_near_flips"# "6_near_flips" # 6_x_last
    p_1420f = base_dir / "1420"

    fig, (ax0, ax1) = plt.subplots(2, 1, num="dev_vs_runtime", sharex=True)
    plt.subplots_adjust(hspace=0.059)

    ax0.grid(alpha=0.2)
    ax1.grid(alpha=0.2)

    ax0.set_ylabel(r"Absolute deviation (µm)", size=font_size, y=0.48)
    ax1.set_ylabel(r"Relative deviation (µm)", size=font_size, y=0.42)

    ax1.set_xlabel("Runtime (ms)", size=font_size)

    ax0.tick_params(axis="both", which="major", labelsize=font_size, width=3, length=8, pad=10)
    ax1.tick_params(axis="both", which="major", labelsize=font_size, width=3, length=8, pad=10)
    ax0.tick_params(which="minor", width=2.5, length=6)
    ax1.tick_params(which="minor", width=2.5, length=6)
    #ax0.yaxis.set_major_locator(ticker.MaxNLocator(5))
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(5))

    sa = r"\bf{$F_1$ and $F_2$}" + "\n" + r"\bf{(6 freq.)"
    ax0.annotate(sa, xy=(0.6, 0.15), xytext=(0.9, 0.11),
                 arrowprops=dict(facecolor="tab:blue", shrink=0.12, ec="tab:blue"),
                 size=font_size - 4, c="tab:blue", va="top")
    sa = r"\bf{$F_2$ only}" + "\n" + r"\bf{(6 freq.)"
    ax0.annotate(sa, xy=(3.6, 0.2), xytext=(8, 5),
                 arrowprops=dict(facecolor="tab:orange", shrink=0.12, ec="tab:orange"),
                 size=font_size - 4, c="tab:orange", va="top")

    sa = r"\bf{$F_1$ and $F_2$}" + "\n" + r"\bf{(1420 freq.)"
    ax0.annotate(sa, xy=(100, 0.15), xytext=(10, 0.1),
                 arrowprops=dict(facecolor="tab:green", shrink=0.12, ec="tab:green"),
                 size=font_size - 4, c="tab:green", va="top")
    sa = r"\bf{$F_2$ only}" + "\n" + r"\bf{(1420 freq.)"
    ax0.annotate(sa, xy=(300, 0.7), xytext=(160, 0.01),
                 arrowprops=dict(facecolor="tab:red", shrink=0.12, ec="tab:red"),
                 size=font_size - 4, c="tab:red", va="center")

    plot_data(p_6f, ax0, ax1)
    plot_data(p_1420f, ax0, ax1)

    ax0.set_xscale("log")
    ax1.set_xscale("log")
    ax0.set_yscale("log")
    ax1.set_yscale("log")

    def decimal_formatter(x, pos):
        if x >= 1:
            return f"{x:.0f}"
        if 0.1 <= x < 1.0:
            return f"{x:.1f}"
        elif 0.01 <= x < 0.1:
            return f"{x:.2f}"
        else:
            return f"{x:.3f}"

    ax0.yaxis.set_major_formatter(FuncFormatter(decimal_formatter))
    ax1.yaxis.set_major_formatter(FuncFormatter(decimal_formatter))
    ax0.xaxis.set_major_formatter(ScalarFormatter())
    ax1.xaxis.set_major_formatter(ScalarFormatter())

    ax0.set_ylim(0.0011, 25)
    ax1.set_ylim(0.0011, 25)

    ax0.set_xlim(0.19, 959)
    ax1.set_xlim(0.19, 959)


if __name__ == "__main__":
    rel_deviation_v_runtime()

    plt_show(mpl, en_save=0)
