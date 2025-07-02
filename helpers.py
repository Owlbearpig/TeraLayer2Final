import logging
import warnings
from typing import Callable, Iterable

import pandas as pd
from scipy.optimize import root_scalar
import numpy as np
from pathlib import Path
from enum import Enum
import json


def is_iterable(obj):
    """
    print(is_iterable(3.4))
    print(is_iterable([3.4]))
    """
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True


def save_fig(fig_num_, mpl=None, save_dir=None, filename=None, **kwargs):
    if mpl is None:
        import matplotlib as mpl
    if "format" not in kwargs:
        kwargs["format"] = "pdf"
    if "fig_size" not in kwargs:
        kwargs["fig_size"] = (12, 9)

    plt = mpl.pyplot

    rcParams = mpl.rcParams

    if save_dir is None:
        save_dir = Path(rcParams["savefig.directory"])

    fig = plt.figure(fig_num_)

    if filename is None:
        try:
            filename_s = str(fig.canvas.get_window_title())
        except AttributeError:
            filename_s = str(fig.canvas.manager.get_window_title())
    else:
        filename_s = str(filename)

    unwanted_chars = ["(", ")"]
    for char in unwanted_chars:
        filename_s = filename_s.replace(char, '')
    filename_s.replace(" ", "_")
    location = save_dir / (filename_s + ".{}".format(kwargs["format"]))
    fig.set_size_inches(kwargs["fig_size"], forward=False)
    plt.subplots_adjust(wspace=0.3)
    kwargs.pop("fig_size")
    plt.savefig(location, bbox_inches='tight', dpi=300, pad_inches=0, **kwargs)
    print(f"Saved figure {filename_s} at {location}")


def plt_show(mpl_, en_save=False, **kwargs):
    plt_ = mpl_.pyplot
    for fig_num in plt_.get_fignums():
        fig = plt_.figure(fig_num)
        for ax in fig.get_axes():
            h, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend()

        if en_save:
            save_fig(fig_num, mpl_, **kwargs)
    plt_.show()


def read_opt_res_file(file_path):
    opt_res_dict = {}
    with open(file_path, 'r') as file:
        first_line = file.readline()
        if not first_line:
            print("opt res file empty")
            return {}

        splits_line0 = first_line.split(" ")
        d1_truth, d2_truth, d3_truth = [float(di.replace(",", "")) for di in splits_line0[-3:]]
        opt_res_dict.update({"d1_truth": d1_truth, "d2_truth": d2_truth, "d3_truth": d3_truth})

        skip_lines = 3
        d1, d2, d3 = [], [], []
        for i, line in enumerate(file.readlines()):
            if i < skip_lines:
                continue
            splits = line.split(",")

            d1.append(float(splits[1].replace(" ", "")))
            d2.append(float(splits[2].replace(" ", "")))
            d3.append(float(splits[3].replace(" ", "")))

    opt_res_dict.update({"results_d1": d1, "results_d2": d2, "results_d3": d3})

    return opt_res_dict


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
        return df

    p = Path(path_)

    return read_res_file(p)


def write_line_to_file(fp, line, mode="a", insert_idx=None, enable=True):
    if not enable:
        return

    if not fp.exists():
        fp.touch()

    if insert_idx is not None:
        with open(fp, "r") as file:
            lines = file.readlines()
            lines.insert(insert_idx, line)

        with open(fp, "w") as file:
            file.writelines(lines)
    else:
        with open(fp, mode) as file:
            file.write(line)


def export_config(options_):
    if not options_["en_save"]:
        return
    opt_save_dir = options_["save_dir"]
    with open(opt_save_dir / "options.json", "w") as opt_file:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Enum) or isinstance(obj, Path):
                    return str(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        try:
            json.dump(options_, opt_file, indent=4, cls=NpEncoder)
        except TypeError as te:
            print(options_)
            raise te


# Function to format y-axis labels in terms of π
def format_func(value, tick_number):
    N = int(np.round(value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi$"
    elif N == -1:
        return r"-$\pi$"
    else:
        return r"${0}\pi$".format(N)


def reduce_sweep_cnt(data_dict, reduced_len):
    reduced_data_dict = {}
    for k in data_dict:
        arr = np.array(data_dict[k])
        if not np.issubdtype(arr.dtype, np.number):
            reduced_data_dict[k] = arr
            continue

        if arr.ndim == 2 and arr.shape[0] > reduced_len:
            arr = arr[:reduced_len, :]
        else:
            arr = arr

        reduced_data_dict[k] = arr

    return reduced_data_dict


