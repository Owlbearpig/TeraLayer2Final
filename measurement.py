from enum import Enum
import numpy as np
from samples import SamplesEnum
import json
import pandas as pd
from datetime import datetime
from consts import c_thz, thea, data_root
from tmm_package import coh_tmm_slim_no_checks
from typing import List
from pathlib import Path
from helpers import reduce_sweep_cnt


class SystemEnum(Enum):
    PIC = 1
    WaveSource = 2
    WaveSourcePicFreq = 3
    TSweeper = 4
    ECOPS = 5
    Model = 6


class MeasTypeEnum(Enum):
    Background = 1
    Reference = 2
    Sample = 3


def _set_options(options_=None) -> dict:
    # set default values if key not in options_
    _default_options = {}
    if not options_:
        return _default_options

    for k in _default_options:
        if k not in options_:
            options_[k] = _default_options[k]

    return options_


class Measurement:
    file_path = None
    freq = None
    freq_OSA = None
    name = None
    amp = None
    amp_avg = None
    phase = None
    phase_avg = None
    n_sweeps = None
    meas_type = None
    system = None
    timestamp = None
    sample = None
    r = None
    r_avg = None
    r_std = None
    pulse_shift = None

    def __init__(self, file_path, reduced_len=1, options=None):
        if options is None:
            self.options = {}
        else:
            self.options = options

        self.reduced_len = reduced_len
        self._parse_file(file_path)

    def __repr__(self):
        fp = self.file_path
        atr_strs = [f"file: {fp.name}", f"system: {self.system.name}",
                 f"sweep cnt: {self.n_sweeps}", f"freq cnt: {len(self.freq)}"]
        s = "(" + ", ".join(atr_strs) + ")"
        return s

    def time_diff(self, meas):
        if isinstance(self.timestamp, datetime):
            return (self.timestamp - meas.timestamp).total_seconds()
        else:
            return self.timestamp - meas.timestamp

    def _set_meas_type(self):
        file_name_lower = str(self.file_path.name).lower()
        bkg_strs = ["background", "bkg"]
        ref_strs = ["short", "goldplatte", "reference"]

        for bkg_str in bkg_strs:
            if bkg_str in file_name_lower:
                self.meas_type = MeasTypeEnum.Background
                return

        for ref_str in ref_strs:
            if ref_str in file_name_lower:
                self.meas_type = MeasTypeEnum.Reference
                return

        self.meas_type = MeasTypeEnum.Sample

    def _set_system(self):
        if "Discrete Frequencies - PIC" in str(self.file_path):
            self.system = SystemEnum.PIC
        elif ("Discrete Frequencies - WaveSource" in str(self.file_path) and
              ("PIC" not in str(self.file_path))):
            self.system = SystemEnum.WaveSource
        elif "WaveSource" in str(self.file_path) and "PIC" in str(self.file_path):
            self.system = SystemEnum.WaveSourcePicFreq
        elif "ECOPS" in str(self.file_path):
            self.system = SystemEnum.ECOPS
        else:
            self.system = SystemEnum.TSweeper

    def _set_sample(self):
        if not self.meas_type == MeasTypeEnum.Sample:
            return

        file_path_str = str(self.file_path).lower()

        def string_match(s_list, full_match=False):
            matches = [s in file_path_str for s in s_list]
            if full_match:
                return all(matches)
            else:
                return any(matches)

        if string_match(["cube"]):
            self.sample = SamplesEnum.blueCube
        elif string_match(["fp"]):
            if string_match(["probe2"]):
                self.sample = SamplesEnum.fpSample2
            elif string_match(["probe3"]):
                self.sample = SamplesEnum.fpSample3
            elif string_match(["probe5_plastic"]):
                self.sample = SamplesEnum.fpSample5Plastic
            elif string_match(["probe5_ceramic"]):
                self.sample = SamplesEnum.fpSample5ceramic
            elif string_match(["probe6"]):
                self.sample = SamplesEnum.fpSample6
        elif string_match(["op_blue", "pos1"], full_match=True):
            self.sample = SamplesEnum.opBluePos1
        elif string_match(["op_blue", "pos2"], full_match=True):
            self.sample = SamplesEnum.opBluePos2
        elif string_match(["op_black", "pos1"], full_match=True):
            self.sample = SamplesEnum.opBlackPos1
        elif string_match(["op_black", "pos2"], full_match=True):
            self.sample = SamplesEnum.opBlackPos2
        elif string_match(["op_red", "pos1"], full_match=True):
            self.sample = SamplesEnum.opRedPos1
        elif string_match(["op_red", "pos2"], full_match=True):
            self.sample = SamplesEnum.opRedPos2
        elif string_match(["op_darkred", "pos1"], full_match=True):
            self.sample = SamplesEnum.opDarkRedPos1
        elif string_match(["op_darkred", "pos2"], full_match=True):
            self.sample = SamplesEnum.opDarkRedPos2
        elif string_match(["ampelmann"]):
            if string_match(["right", "rechts"]):
                self.sample = SamplesEnum.ampelMannRight
            elif string_match(["left", "links"]):
                self.sample = SamplesEnum.ampelMannLeft
            elif string_match(["r_left", "r_links"]):
                self.sample = SamplesEnum.ampelMannLeft
            else:
                self.sample = SamplesEnum.ampelMannOld
        elif string_match(["op_tool"]):
            if string_match(["red_pos1"]):
                self.sample = SamplesEnum.opToolRedPos1
            elif string_match(["red_pos2"]):
                self.sample = SamplesEnum.opToolRedPos2
            elif string_match(["blue_pos1"]):
                self.sample = SamplesEnum.opToolBluePos1
            elif string_match(["blue_pos2"]):
                self.sample = SamplesEnum.opToolBluePos2
        elif string_match(["ceramic", "keramik"]):
            if string_match(["schwarz", "black"]):
                self.sample = SamplesEnum.bwCeramicBlackUp
            elif string_match(["weis", "white"]):
                self.sample = SamplesEnum.bwCeramicWhiteUp

        if not self.sample:
            self.sample = SamplesEnum.empty

    def _parse_csv_file(self):
        with open(self.file_path, "r") as file:
            first_5_lines = [file.readline().strip() for _ in range(5)]

        self.timestamp = datetime.strptime(first_5_lines[1].split(" ")[-1], "%Y-%m-%dT%H:%M:%S.%fZ")

        pd_df = pd.read_csv(self.file_path, skiprows=4)
        self.freq = np.array(pd_df["Frequency (THz)"], dtype=float)

        amp, phase = pd_df["Amplitude Signal (a.u.)"], pd_df["Phase Signal (rad)"]

        self.amp = np.array(amp, dtype=float)
        self.phase = np.array(phase, dtype=float)
        self.n_sweeps = 1

    def _parse_json_or_npz_file(self):
        def LoadData(fName='DataDict.json'):
            """
            Created on Thu Jan 11 14:23:21 2024

            @author: schwenson
            """

            with open(fName, 'r') as f:
                DataDict = json.load(f)

            # change all lists back to numpy arrays
            for key in DataDict.keys():
                if type(DataDict[key]) == list:
                    DataDict[key] = np.array(DataDict[key], dtype=float)

            return DataDict

        file_suffix = self.file_path.suffix[1:]

        # if json convert to npz for faster load times.
        # Saves full set and on second load saves first self.reduced_len sweeps
        if file_suffix == "json":
            json_dict = LoadData(self.file_path)
            if self.file_path.stat().st_size > 1e6:
                np.savez(str(self.file_path.with_suffix("")), **json_dict)
                if self.file_path.with_suffix(".npz").exists():
                    self.file_path.with_suffix(".json").unlink()
        elif file_suffix == "npz":
            json_dict = np.load(self.file_path)
            self._save_reduced_datasets(json_dict)
        else:
            return

        self.freq = json_dict["Frequency [THz]"]

        sorted_indices = np.argsort(self.freq)

        self.freq = self.freq[sorted_indices]

        try:
            self.freq_OSA = json_dict["Frequency [THz] (OSA measurement)"]
        except KeyError:
            self.freq_OSA = self.freq

        amp_key = [key for key in json_dict if ("Amp" in key) and ("(raw)" not in key)][0]
        phase_key = [key for key in json_dict if ("Phase" in key) and ("(raw)" not in key)][0]

        self.name = json_dict["Measurement"]
        self.n_sweeps = len(json_dict[amp_key])
        try:
            self.timestamp = json_dict["measure #"]
        except KeyError:
            self.timestamp = 0

        self.freq_OSA = self.freq_OSA[sorted_indices]  # ! Assume order is the same

        self.freq = self.freq_OSA

        phase = json_dict[phase_key][:, sorted_indices]
        amp = json_dict[amp_key][:, sorted_indices]

        phase *= np.sign(self.freq)

        self.amp = np.array(amp, dtype=float)

        phase_sign = 1
        if self.system == SystemEnum.PIC:
            phase_sign = -1
        self.phase = phase_sign * np.array(phase, dtype=float)

    def _parse_file(self, file_path):
        self.file_path = Path(file_path)

        self._set_meas_type()
        self._set_system()
        self._set_sample()

        if ".json" in str(self.file_path):
            self._parse_json_or_npz_file()
        elif ".npz" in str(self.file_path):
            self._parse_json_or_npz_file()
        elif ".csv" in str(self.file_path) and ("lock" not in str(self.file_path)):
            self._parse_csv_file()
        else:
            print(f"Can't parse {file_path.stem}")

        self._set_mean_vals()

    def _set_mean_vals(self):
        # average amp and phase over multiple consecutive measurements of same sample
        if self.n_sweeps == 1:
            self.amp_avg = self.amp
            self.phase_avg = self.phase
        else:
            self.amp_avg = np.mean(self.amp, axis=0)
            self.phase_avg = self._calculate_mean_phase()


    def _calculate_mean_phase(self,  phase=None, axis=0):
        if phase is None:
            phase = self.phase

        if np.isclose(phase, np.zeros_like(phase)).all():
            return phase

        phase_cart = np.exp(1j * phase)
        phase_avg = np.angle(np.mean(phase_cart, axis=axis))

        return phase_avg

    def _save_reduced_datasets(self, full_data_dict):
        if self.file_path.stat().st_size < 0.1e9:
            return

        parts = self.file_path.parts
        reduced_dataset_dir = Path(*[*parts[:-2], parts[-2] + f" - {self.reduced_len} sweeps"])
        avg_dataset_dir = Path(*[*parts[:-2], parts[-2] + f" - {self.reduced_len} averaged sweeps"])

        if not (reduced_dataset_dir / parts[-1]).exists():
            reduced_dataset_dir.mkdir(parents=False, exist_ok=True)
            reduced_data_dict = reduce_sweep_cnt(full_data_dict, self.reduced_len)
            np.savez(str(reduced_dataset_dir / parts[-1]), **reduced_data_dict)

        if not (avg_dataset_dir / parts[-1]).exists():
            slice_len = self.reduced_len
            avg_dataset_dir.mkdir(parents=False, exist_ok=True)
            avg_data_dict = {}
            for k in full_data_dict:
                arr = full_data_dict[k]
                if arr.ndim != 2:
                    continue

                if np.issubdtype(arr.dtype, np.number) and arr.shape[0] > slice_len:
                    if "phase" in k.lower():
                        mean_calc = self._calculate_mean_phase
                    elif "amp" in k.lower():
                        mean_calc = np.mean
                    else:
                        break

                    m = arr.shape[0] // slice_len
                    averaged_arr = np.zeros((m, arr.shape[1]))
                    for i in range(m):
                        averaged_arr[i] = mean_calc(arr[i*slice_len:(i+1)*slice_len], axis=0)
                    arr = averaged_arr

                avg_data_dict[k] = arr

            np.savez(str(avg_dataset_dir / parts[-1]), **avg_data_dict)


class ModelMeasurement(Measurement):
    def __init__(self, sample, ref=None):
        if ref is None:
            file = Path(r"2024_07_24 - Dry Air - Frequency Corrected & Background removal") / "Goldplatte.npz"
            ref_meas_path = data_root / "T-Sweeper" / file
        else:
            ref_meas_path = ref.file_path

        self.ref_meas = Measurement(ref_meas_path)
        super().__init__(self.ref_meas.file_path)

        self.sample = sample
        self.r_std = None
        self.meas_type = MeasTypeEnum.Sample
        self.name = f"Model {self.sample}"

        self.simulate_sam_measurement()
        self.system = SystemEnum.Model
        self.n_sweeps = 5000 # statistic when adding noise

    def simulate_sam_measurement(self, fast=False, selected_freqs=None):
        err_conf = np.seterr(divide='ignore')
        has_iron_core = self.sample.value.has_iron_core

        selected_f_idx = None
        if selected_freqs is not None:
            selected_f_idx = [np.argmin(np.abs(f-self.freq)) for f in selected_freqs]

        n = self.sample.value.get_ref_idx(self.freq)

        d_truth = self.sample.value.thicknesses

        skip_mod = 4

        r_mod = np.zeros_like(self.freq, dtype=complex)
        for f_idx, freq in enumerate(self.freq):
            if (selected_f_idx is not None) and (f_idx not in selected_f_idx):
                continue

            if fast and f_idx > 1500:
                continue

            if fast and (f_idx % skip_mod) != 0:
                r_mod[f_idx] = r_mod[f_idx - 1]
                continue

            lam_vac = c_thz / freq
            if has_iron_core:
                d_ = np.array([np.inf, *d_truth, 10, np.inf], dtype=float)
            else:
                d_ = np.array([np.inf, *d_truth, np.inf], dtype=float)
            r_mod[f_idx] = -1 * coh_tmm_slim_no_checks("s", n[f_idx], d_, thea, lam_vac)

        for f_idx in range(len(self.freq)):
            if fast and (f_idx % skip_mod) != 0:
                continue
                # r_mod[f_idx] = r_mod[f_idx - 1]

        # ref_meas = find_nearest_meas()
        ref_meas = self.ref_meas
        ref_fd = np.array([ref_meas.freq, ref_meas.amp_avg * np.exp(1j * ref_meas.phase_avg)]).T

        self.amp, self.phase, self.r = np.zeros((3, self.n_sweeps, len(self.freq)), dtype=complex)
        for sweep_idx in range(self.n_sweeps):
            self.amp[sweep_idx], self.phase[sweep_idx] = np.abs(ref_fd[:, 1] * r_mod), np.angle(ref_fd[:, 1] * r_mod)
            self.r[sweep_idx] = r_mod

        self.amp_avg, self.phase_avg, self.r_avg = self.amp[0], self.phase[0], self.r[0]

        np.seterr(**err_conf)


def find_nearest_meas(meas1: Measurement, meas_list: List[Measurement]):
    all_meas_same_set = []
    for meas2 in meas_list:
        if meas1.file_path.suffix[1:] == meas2.file_path.suffix[1:]:
            all_meas_same_set.append(meas2)

    abs_time_diffs = []
    for meas2 in all_meas_same_set:
        abs_time_diff = np.abs(meas2.time_diff(meas1))
        abs_time_diffs.append((abs_time_diff, meas2))

    sorted_time_diffs = sorted(abs_time_diffs, key=lambda x: x[0])

    closest_meas = sorted_time_diffs[0][1]
    if meas1.system != SystemEnum.TSweeper:
        closest_meas = sorted_time_diffs[0][1]

    return closest_meas
