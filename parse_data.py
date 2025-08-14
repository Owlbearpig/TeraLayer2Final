import json
import numpy as np
import logging
from consts import data_root
from measurement import Measurement, ModelMeasurement, MeasTypeEnum, SystemEnum
from functions import window, x_ax_crossing, do_rfft, do_irfft
from helpers import reduce_sweep_cnt
from pathlib import Path
from tqdm import tqdm


# search for these dirs in data_root
sub_dirs = [
    # "Discrete Frequencies - WaveSource",
    # "Discrete Frequencies - WaveSource all sweeps",
    # "Discrete Frequencies - WaveSource (PIC-Freuqency Set)",
    # "Discrete Frequencies - WaveSource (PIC-Frequency Set) all sweeps",
    # "T-Sweeper",
    # "Discrete Frequencies - PIC all sweeps",
    # "Discrete Frequencies - PIC",
    # "2024_07_24 - Dry Air - Copy",
    # "2024_07_24 - Dry Air - Copy - 50 sweeps",
    # "2024_07_24 - Dry Air - Frequency corrected",
    # "2024_07_24 - Dry Air - Frequency Corrected & Background removal",
    # "2024_07_24 - Dry Air - Frequency Corrected & Background removal - 5 averaged sweeps",
    # "2024_07_24 - Dry Air - Frequency Corrected & Background removal - 200 sweeps",
    # "2024_07_24 - Dry Air - Copy - 50 averaged sweeps",
    # "2024_07_24 - Dry Air - Copy - 5 averaged sweeps",
    # "2024_07_24 - Dry Air - Copy - 10 averaged sweeps",
    "2024_07_24 - Dry Air - Copy - 5 averaged sweeps",
    # "ECOPS - td", # original dataset, must be transformed first
    # "ECOPS - fd - 200 sweeps",
    # "ECOPS - fd - 100 sweeps - even",
    # "ECOPS - fd - 100 sweeps - odd",
    # "ECOPS - fd - 2500 sweeps - even",
    # "ECOPS - fd - 2500 sweeps - odd"
    # "ECOPS - fd - full - 5000 sweeps - even"
    # "ECOPS - fd - full - 100 sweeps - even",
    # "ECOPS - td - full",
    # "ECOPS - fd - full - 10000 sweeps",
]
reduced_len = 10000  # also sets sweep moving avg. window size


def filter_files(included_filetypes=None):
    all_files = [path for path in data_root.rglob('*') if path.is_file()]

    excluded_files = ["01_Gold_Plate_short.csv"]
    if included_filetypes is None:
        included_filetypes = ["csv", "json", "npz"]

    for sub_dir in sub_dirs:
        if all([sub_dir not in file.parent.name for file in all_files]):
            raise FileNotFoundError(f"Folder \"{sub_dir}\" not found or is empty")

    filtered_files = []
    for file in all_files:
        if file.name in excluded_files:
            continue
        if file.suffix[1:] not in included_filetypes:
            continue
        if file.parent.name not in sub_dirs:
            continue
        filtered_files.append(file)

    return filtered_files


def parse_measurements(ret_info=False, options=None):
    files = filter_files()

    measurements = {"all": [Measurement(file_path, reduced_len, options) for file_path in files]}
    measurements["refs"] = [meas for meas in measurements["all"] if meas.meas_type == MeasTypeEnum.Reference]
    measurements["sams"] = [meas for meas in measurements["all"] if meas.meas_type == MeasTypeEnum.Sample]
    measurements["bkgs"] = [meas for meas in measurements["all"] if meas.meas_type == MeasTypeEnum.Background]

    info = {"sub_dirs": sub_dirs, "files": files}

    if ret_info:
        return measurements, info
    else:
        return measurements


def transform_dataset():
    # reads dir with json files -> reduce sweep cnt (min(sweep_cnt, reduced_len))
    # -> fd transform -> save as npz -> split dict into even, odd
    # -> save both odd and even dicts as npz (each with "reduced_len" sweeps / 2).
    # 3 new dirs in the end.

    files = filter_files(included_filetypes=["json"])
    if not files:
        return

    for file_path in files:
        if "left" not in str(file_path.name): # "left" (Sam), "Gold" (Ref)
            pass

        with open(file_path, "r") as f:
            logging.info(f"Reading file: {file_path}")

            json_dict = json.load(f)
            np_dict_td = {}
            for k in json_dict:
                np_dict_td[k] = np.array(json_dict[k])

            sweep_cnt = np_dict_td["Terahertz Signal (nA)"].shape[0]
            # if reduced length is above sweep_cnt choose sweep_cnt as max
            sweep_cnt = min(reduced_len, sweep_cnt)
            np_dict_td = reduce_sweep_cnt(np_dict_td, sweep_cnt)

            time_axis = np_dict_td["Time (ps)"]
            amp_td = np_dict_td["Terahertz Signal (nA)"]

            freq_axis = do_rfft(np.array([time_axis, amp_td[0]]).T)[:, 0]
            zero_crossings = np.zeros(sweep_cnt)
            amp_fd, phase = np.zeros((2, sweep_cnt, len(freq_axis)))
            for sweep_idx in tqdm(range(sweep_cnt)):
                amp_td[sweep_idx] -= np.mean(amp_td[sweep_idx][:20])
                y_td = np.array([time_axis, amp_td[sweep_idx]]).T
                #"""
                if "background" not in file_path.name.lower():
                    y_td = window(y_td, en_plot=False, win_width=40)
                #"""
                y_fd = do_rfft(y_td)

                zero_crossings[sweep_idx] = x_ax_crossing(y_td)
                # if even shift to pulse pos of even, same for odd
                dt = zero_crossings[sweep_idx % 2] - zero_crossings[sweep_idx]
                dphi = 2 * np.pi * freq_axis * dt

                amp_fd[sweep_idx] = np.abs(y_fd[:, 1])
                phase[sweep_idx] = np.angle(y_fd[:, 1] * np.exp(1j * dphi))

            np_dict_fd = {"Measurement": file_path.stem,
                          "Frequency [THz]": freq_axis.real,
                          "Linear Amplitude [a.u.]": amp_fd.real,
                          "Phase [rad]": phase.real
                          }

            # save reduced len fd dict
            parts = file_path.parts
            out_dir_base = str(parts[-2]).replace("td", "fd")
            out_dir_name = out_dir_base + f" - {sweep_cnt} sweeps"
            output_dir = Path(file_path.parents[1]) / out_dir_name
            output_dir.mkdir(parents=False, exist_ok=True)
            np.savez(str(output_dir / file_path.stem), **np_dict_fd)

            # split dict and save both
            split_dict_even, split_dict_odd = split_npz(np_dict_fd)
            even_out_dir = Path(file_path.parents[1]) / (out_dir_base + f" - {sweep_cnt // 2} sweeps - even")
            odd_out_dir = Path(file_path.parents[1]) / (out_dir_base + f" - {sweep_cnt // 2} sweeps - odd")
            even_out_dir.mkdir(parents=False, exist_ok=True)
            odd_out_dir.mkdir(parents=False, exist_ok=True)

            np.savez(str(even_out_dir / file_path.stem), **split_dict_even)
            np.savez(str(odd_out_dir / file_path.stem), **split_dict_odd)


def split_npz(data_dict):
    split_dict_even, split_dict_odd = {}, {}
    for k in data_dict:
        arr = np.array(data_dict[k])
        if not np.issubdtype(arr.dtype, np.number):
            split_dict_even[k] = arr
            split_dict_odd[k] = arr
            continue

        if arr.ndim == 2:
            split_dict_even[k] = arr[0::2, :]
            split_dict_odd[k] = arr[1::2, :]
        else:
            split_dict_even[k] = arr
            split_dict_odd[k] = arr

    return split_dict_even, split_dict_odd


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    transform_dataset()

    # all_measurements = parse_measurements()

    """
    all_measurements = parse_measurements()

    # print(all_measurements["all"])
    for i, measurement in enumerate(all_measurements["all"]):
        print(i, measurement.file_path)

    # print(all_measurements[98].freq)

    # tmm = ModelMeasurement(all_measurements[85])
    """


