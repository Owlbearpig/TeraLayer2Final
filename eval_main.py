import random
from pathlib import Path
from parse_data import *
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Button, Slider
from matplotlib.ticker import MultipleLocator, FuncFormatter
from helpers import plt_show, read_opt_res_file, format_func, export_config, parse_result_dir
from mpl_settings import mpl_style_params, result_dir
from consts import thea, c_thz
from tmm_package import coh_tmm_slim, list_snell, interface_r, coh_tmm_slim_no_checks
from functools import partial
from functions import do_ifft, moving_average, std_err
from measurement import ModelMeasurement, Measurement, SamplesEnum, SystemEnum, find_nearest_meas
import logging
from copy import deepcopy
from numpy import polyfit
import subprocess
from scipy.special import comb

# TODO FIX THIS SHIT
logging.getLogger('matplotlib').setLevel(logging.WARNING)

_default_options = {"selected_system": SystemEnum.TSweeper,
                    "selected_sample": SamplesEnum.ampelMannLeft,
                    "en_save": 0,
                    "selected_sweep": 0,  # selected_sweep: int, None (=average) or "random"
                    "less_plots": 1,
                    "debug_info": 0,
                    "freq_selection": [0.307, 0.459, 0.747, 0.82, 0.960, 1.2],
                    "th_0": 8 * np.pi / 180,
                    "pol": "s",
                    "en_mv_avg": 0,
                    "en_print": 1,
                    "use_r_meas": 0,
                    "use_avg_ref": 0,
                    "c_proj_path": None,

                    # modifies measurement data
                    "ri_shift": None,  # example: ri_shift = [0, 0.1, 0] -> ri = ri + [0, 0.1, 0] (added through get_ri)
                    "meas_f_axis_shift": None,  # in THz. Shifts f_axis in shift_freq_axis method
                    "noise_scaling": None, #
                    "inc_angle": thea,
                    "calc_f_axis_shift": None, # Shifts the freqs. used in the coe calculation (see shift_f_axis method)
                    }


def sel_freqs_at_minima(cnt=6):
    # phase minima
    range2 = (152, 159)
    range3 = (232, 239)
    range4 = (309, 317)
    range5 = (386, 397)
    range6 = (459, 478)
    range7 = (535, 556)
    range8 = (607, 643)
    range9 = (679, 767)
    range10 = (752, 831)
    range11 = (823, 909)
    range12 = (889, 988)
    range13 = (961, 1247)
    range14 = (1044, 1247)
    range15 = (1220, 1247)

    ranges = [range2, range3, range4, range5, range6, range7, range8, range9, range10, range11, range12, range13, range14, range15]
    ranges = [(r[0]-5, r[0]+5) for r in ranges]

    selected_ranges = random.sample(ranges, cnt)
    selected_freqs = np.sort(np.array([random.randint(*sel_range) for sel_range in selected_ranges]))

    return selected_freqs * 1e-3 # return in THz

def shift_f_axis(f_axis, f_axis_shift=None):
    # Absolute shifts in THz
    if f_axis_shift is None:
        return f_axis

    if not isinstance(f_axis_shift, dict):
        logging.warning("f_axis_shift must be a dictionary. Ignoring f_axis_shift option")
        return f_axis

    if "rel_shift" in f_axis_shift:
        f_axis = f_axis * (1 + f_axis_shift["rel_shift"])
    elif "abs_shift" in f_axis_shift:
        f_axis = f_axis + f_axis_shift["abs_shift"]

    return f_axis


def calculate_coe(meas, options_, sweep_cnt=5000):
    # For 1 and 3 layers (2 layers not implemented)

    layer_cnt = options_["selected_sample"].value.layers
    logging.basicConfig(level=logging.WARNING)

    # np.set_printoptions(precision=5)

    coe_cnt, k_cnt = 8, 5
    freq_cnt = len(options_["freq_selection"])
    print(f"sweep_cnt: {sweep_cnt}, freq_cnt: {freq_cnt}")

    freq_selection = options_["freq_selection"]
    freq_shift = options_["calc_f_axis_shift"]

    f_axis_meas = meas.freq
    f_axis_calc = shift_f_axis(meas.freq, freq_shift)

    lam_vac = c_thz / f_axis_calc

    n_list = options_["selected_sample"].value.get_ref_idx(f_axis_calc)
    r_options = {"freqs": f_axis_calc, "th_0": options_["th_0"], "pol": options_["pol"]}
    r_fn = options_["selected_sample"].value.get_r(r_options)
    def coe_1d():
        # coe_ needs tmm update (see 3D case)
        coe_ = np.zeros((sweep_cnt, coe_cnt, freq_cnt), dtype=complex)
        k_ = np.zeros((sweep_cnt, k_cnt, freq_cnt), dtype=complex)
        for sweep_idx in tqdm(range(sweep_cnt)):
            for m, sel_f in enumerate(options_["freq_selection"]):
                f_idx = np.argmin(np.abs(f_axis_meas - sel_f))
                r_exp_ = -meas.r[sweep_idx, f_idx]

                coe_[sweep_idx, 0, m] = (r_fn[m, 0, 1] - r_exp_)
                coe_[sweep_idx, 0, m] *= 1 / (r_exp_ * r_fn[m, 0, 1] * r_fn[m, 1, 2] - r_fn[:, 1, 2])
                k_[sweep_idx, 1, m] = 2 * np.pi * f_axis_calc[m] * n_list[m, 1] / c_thz

        w_ = np.ones_like(f_axis_calc, dtype=complex)

        return coe_, k_, w_

    def coe_3d():
        r_mod = np.zeros(freq_cnt, dtype=complex)
        # First idx: [re-arranged tmm, tmm]
        coe_ = np.zeros((2, sweep_cnt, coe_cnt, freq_cnt), dtype=complex)
        k_ = np.zeros((sweep_cnt, k_cnt, freq_cnt), dtype=complex)
        for sweep_idx in tqdm(range(sweep_cnt)):
            th_list = np.zeros((freq_cnt, layer_cnt+2), dtype=complex)
            for m, sel_f in enumerate(freq_selection):
                f_idx = np.argmin(np.abs(f_axis_meas - sel_f))
                th_list[m] = list_snell(n_list[m], thea).T
                k_[sweep_idx, :, m] = 2 * np.pi * n_list[m] * np.cos(th_list[m]) / lam_vac[f_idx]

                # for testing
                r_mod[m] = -coh_tmm_slim("s",
                                         n_list[m],
                                         [np.inf, 73, 660, 43, np.inf],
                                         thea, lam_vac[f_idx])

            f_idx_sel = [np.argmin(np.abs(f - f_axis_meas)) for f in freq_selection]
            if options_["selected_sweep"] is None:
                r_exp_ = meas.r_avg[f_idx_sel]
            else:
                r_exp_ = meas.r[sweep_idx, f_idx_sel]

            # r_exp_ = r_mod
            # r_exp_mod[-1] = r_mod[-1]
            # r_exp_mod[2] = r_mod[2]

            r = -np.abs(r_exp_) * np.exp(1j * np.angle(r_exp_))

            c0 = r * r_fn[:, 0, 1] * r_fn[:, 3, 4] - r_fn[:, 3, 4]
            c1 = r * r_fn[:, 1, 2] * r_fn[:, 3, 4] - r_fn[:, 0, 1] * r_fn[:, 1, 2] * r_fn[:, 3, 4]
            c2 = (r * r_fn[:, 0, 1] * r_fn[:, 1, 2] * r_fn[:, 2, 3] - r_fn[:, 1, 2] * r_fn[:, 2, 3]) * r_fn[:, 3, 4]
            c3 = r * r_fn[:, 0, 1] * r_fn[:, 2, 3] - r_fn[:, 2, 3]
            c4 = r * r_fn[:, 2, 3] * r_fn[:, 3, 4] - r_fn[:, 0, 1] * r_fn[:, 2, 3] * r_fn[:, 3, 4]
            c5 = r * r_fn[:, 1, 2] * r_fn[:, 2, 3] - r_fn[:, 0, 1] * r_fn[:, 1, 2] * r_fn[:, 2, 3]
            c6 = r * r_fn[:, 0, 1] * r_fn[:, 1, 2] - r_fn[:, 1, 2]
            c7 = r - r_fn[:, 0, 1]

            coe_[0][sweep_idx] = [c0, c1, c2, c3, c4, c5, c6, c7]
            coe_[1][sweep_idx] = [r, r_fn[:, 0, 1], r_fn[:, 1, 2], r_fn[:, 2, 3], r_fn[:, 3, 4],
                                  c5, c6, c7] # last 3 are unused
            """
            if sweep_idx == options_["selected_sweep"]:
                print(sweep_idx, coe_[1][sweep_idx][0].real, coe_[1][sweep_idx][0].imag)
                for i in range(4):
                    print(f"r{i}:", coe_[1][sweep_idx][i+1])
            """
        f_idx_sel = [np.argmin(np.abs(f - f_axis_meas)) for f in freq_selection]

        # sweeps = np.arange(sam_meas.r.shape[0])
        # sweeps = sweeps[sweeps != 4273] # sweep 4273 is broken...
        real_weights = 1/np.std(meas.r.real[:4250, f_idx_sel], axis=0)
        imag_weights = 1/np.std(meas.r.imag[:4250, f_idx_sel], axis=0)

        # w_ = np.ones(len(f_idx_sel), dtype=complex) # real_weights + 1j * imag_weights
        w_ = real_weights + 1j * imag_weights

        return coe_, k_, w_

    if layer_cnt == 1:
        coe, k, w = coe_1d()
    elif layer_cnt == 3:
        coe, k, w = coe_3d()
    else:
        raise NotImplemented

    c_proj_path = options_["c_proj_path"]

    coe.tofile(str(c_proj_path / "c.bin"))
    k.tofile(str(c_proj_path / "k.bin"))
    w.tofile(str(c_proj_path / "w.bin"))


def simulated_measurements(options_):
    sam_cnt = 1
    d_truth = np.zeros((sam_cnt, 3))
    n_truth = np.zeros((sam_cnt, 3, 2), dtype=complex)

    for k in range(sam_cnt):
        d_truth[k, 0] = 63.5 # 40 + np.random.random() * 150
        d_truth[k, 1] = 658.3 # 520 + np.random.random() * 170
        d_truth[k, 2] = 48.4 #  50 + np.random.random() * 140

    for k in range(sam_cnt):
        n_truth[k, 0] = (1.1, 4.0 + 0.010j + k)
        n_truth[k, 1] = (1.5, 6.0 + 0.010j)
        n_truth[k, 2] = (1.1, 4.0 + 0.010j)

        n_truth[k, 0] = (1.527 - 0.000j, 1.532 - 0.11j)
        n_truth[k, 1] = (2.80 - 0.000j, 2.82 - 0.015j)
        n_truth[k, 2] = (1.527 - 0.000j, 1.532 - 0.11j)

    options_["noise_scaling"] = 0.0
    # Regarding the freq. selection:
    # (it seems that one should simply sel. freq. where the noise is lowest,
    # coincidentally that is at the turning points in case of the real measurement.
    # => Simulations without noise can be evaluated using any freq. set. giving the correct result.)

    jl_eval = JumpingLaserEval(options_)
    refs = jl_eval.measurements["refs"]

    mod_meas0 = ModelMeasurement(options_["selected_sample"], refs[0])

    mod_meas_list = []
    with open(str(options_["c_proj_path"] / "truths.txt"), "w") as file:
        file.write("# sam idx, d_truth, n_truth\n")
        for k in range(sam_cnt):
            print("simulating sample", k, d_truth[k], [n_truth[k][m] for m in range(len(d_truth[k]))], "\n")
            file.write(f"{k}, {d_truth[k]} ")
            file.write(f"{[list(n_truth[k, i_]) for i_ in range(n_truth.shape[1])]}\n")
            options_["of_test"] = d_truth[k]

            new_mod_meas = deepcopy(mod_meas0)
            new_mod_sam = SamplesEnum.ampelMannLeft
            new_mod_sam.value.set_thicknesses(d_truth[k])

            new_ref_idx = [*n_truth[k]]
            new_mod_sam.value.set_ref_idx(new_ref_idx)
            np.set_printoptions(precision=3)
            set_ref_idx = new_mod_sam.value.get_ref_idx(jl_eval.options["freq_selection"])
            print("Refractive indices at f_sel:\n", set_ref_idx)
            new_mod_meas.sample = new_mod_sam
            new_mod_meas.simulate_sam_measurement(selected_freqs=options_["freq_selection"])
            jl_eval.add_noise(new_mod_meas)

            mod_meas_list.append(new_mod_meas)

    return mod_meas_list



class JumpingLaserEval:
    measurements = None

    def __init__(self, new_options=None):
        np.set_printoptions(precision=2)
        self.options = self.load_measurements_and_set_options(new_options)
        export_config(self.options)

        self.plot_vars = {"truth_line_exists": False, "colors": ['red', 'green', 'blue', 'orange', 'purple']}

    def load_measurements_and_set_options(self, options_=None):
        # set default values if key not in options_
        if options_ is None:
            options_ = {}
        if "of_test" in options_:
            options_["en_save"] = 0

        for k in _default_options:
            if k not in options_:
                print(f"Using default value: ({k}, {_default_options[k]})")
                options_[k] = _default_options[k]
        options_["freq_selection"] = np.array(options_["freq_selection"])

        self.measurements, info_dict = parse_measurements(ret_info=True, options=options_)
        options_.update(info_dict)

        selected_sys_meas = [meas for meas in self.measurements["sams"] if meas.system == options_["selected_system"]]
        if not selected_sys_meas:
            raise Exception("Selected system not in parsed files")

        n_sweeps_ = selected_sys_meas[0].n_sweeps
        if str(options_["selected_sweep"]).lower() == "random":
            options_["selected_sweep"] = np.random.randint(0, n_sweeps_)

        sel_sweep = options_["selected_sweep"]
        if sel_sweep is not None and (sel_sweep > n_sweeps_):
            raise Exception(f"Selected sweep ({sel_sweep}) too high. Dataset contains {n_sweeps_} sweeps")

        if options_["debug_info"]:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        return options_

    def plot_all_sweeps(self, measurements, freq_idx=2):
        for system in SystemEnum:
            if system == SystemEnum.TSweeper:
                continue

            meas_same_system = [meas for meas in measurements if (system == meas.system) and (meas.n_sweeps == 1)]

            fig_num = f"{system.name}"
            fig, (ax0, ax1) = plt.subplots(2, 1, num=fig_num)
            ax0.set_title(f"{system.name}")
            ax1.set_xlabel("Meas idx")
            ax0.set_ylabel("Amplitude (dB)")
            ax1.set_ylabel("Phase (rad)")
            for meas in meas_same_system:
                amp_db, phase = 20 * np.log10(meas.amp), meas.phase
                print(meas)
                print(f"Freqs: {meas.freq}")
                print(f"Direct mean amp.: {np.mean(amp_db, axis=0)}±{np.std(amp_db, axis=0)}")
                print(f"Direct mean phase: {np.mean(phase, axis=0)}±{np.std(phase, axis=0)}")
                for i, freq in enumerate(meas.freq):
                    if i != freq_idx:
                        continue
                    ax0.plot(amp_db[:, i], label=f"{meas.name} {meas.timestamp} {np.round(freq, 2)} THz")
                    ax1.plot(phase[:, i], label=f"{meas.name} {meas.timestamp} {np.round(freq, 2)} THz")
            print()

    def fix_r_phi_sign(self, meas: Measurement):
        if meas.n_sweeps == 1:
            print(f"Skipping {meas}, #sweeps: {meas.n_sweeps}")
            return

        meas_tsweeper = None
        for meas_ in self.measurements["sams"]:
            if (meas.sample == meas_.sample) and (meas_.system == SystemEnum.TSweeper):
                meas_tsweeper = meas_
                break

        if not meas_tsweeper:
            print(f"No TSweeper measurement for {meas}")
            return

        for freq_idx, freq in enumerate(meas.freq):
            tsweeper_r_avg = meas_tsweeper.r_avg[np.argmin(np.abs(meas_tsweeper.freq - freq))]
            tsweeper_r_phi_avg, meas_r_phi_avg = np.angle(tsweeper_r_avg), np.angle(meas.r_avg[freq_idx])

            if np.abs(tsweeper_r_phi_avg - meas_r_phi_avg) <= np.abs(tsweeper_r_phi_avg - -meas_r_phi_avg):
                meas.r_avg[freq_idx] = np.abs(meas.r_avg[freq_idx]) * np.exp(1j * meas_r_phi_avg)
            else:
                meas.r_avg[freq_idx] = np.abs(meas.r_avg[freq_idx]) * np.exp(-1j * meas_r_phi_avg)

    def shift_meas_freq_axis(self, sam_meas_: Measurement, ref_meas_: Measurement):
        if sam_meas_.system == SystemEnum.PIC:
            shifts = {SamplesEnum.fpSample3: 0.006,
                      SamplesEnum.ampelMannRight: 0.0,
                      SamplesEnum.fpSample5ceramic: 0.0,
                      SamplesEnum.fpSample2: 0.003,
                      SamplesEnum.fpSample5Plastic: -0.006,
                      SamplesEnum.fpSample6: 0.0,
                      SamplesEnum.bwCeramicBlackUp: 0.006,
                      SamplesEnum.ampelMannLeft: 0.005} # 0.005
        elif sam_meas_.system == SystemEnum.TSweeper:
            shifts = {SamplesEnum.ampelMannRight: 0.0,
                      SamplesEnum.fpSample5ceramic: 0.0,
                      SamplesEnum.fpSample2: 0.0,
                      SamplesEnum.fpSample5Plastic: -0.006,
                      SamplesEnum.fpSample6: 0.0}
            if sam_meas_.file_path.suffix != ".csv":
                shifts = {SamplesEnum.ampelMannLeft: 0.000, # 0.025 used with old dataset
                          SamplesEnum.ampelMannRight: 0.022, }
        elif sam_meas_.system == SystemEnum.WaveSource:
            shifts = {SamplesEnum.ampelMannLeft: 0.000} # tried 0.005, default 0.0
        else:
            shifts = {}

        try:
            shift = shifts[sam_meas_.sample]
        except KeyError:
            shift = 0

        if self.options["meas_f_axis_shift"] is None:
            manual_shift = 0.000
        else:
            manual_shift = self.options["meas_f_axis_shift"]

        sam_meas_.freq += shift + manual_shift
        ref_meas_.freq += shift + manual_shift

    def fix_phase_slope(self, sam_meas_: Measurement):
        # fixes offset at higher frequencies (> 0.8 THz)
        if sam_meas_.system == SystemEnum.TSweeper:
            pulse_shifts = {SamplesEnum.blueCube: 2.6, SamplesEnum.fpSample2: 0.24, SamplesEnum.fpSample3: 0.28,
                            SamplesEnum.fpSample5ceramic: 0.28, SamplesEnum.fpSample5Plastic: 0.39,
                            SamplesEnum.fpSample6: 0.1, SamplesEnum.bwCeramicWhiteUp: 0.20,
                            SamplesEnum.bwCeramicBlackUp: 0.26,
                            SamplesEnum.ampelMannRight: -0.02,
                            SamplesEnum.ampelMannLeft: 0.0, # 0.20 default
                            SamplesEnum.opBlackPos1: 0.1,
                            SamplesEnum.opRedPos1: 0.2,  # not done
                            SamplesEnum.opBluePos1: 0.43}
            if sam_meas_.file_path.suffix != ".csv":
                pulse_shifts[SamplesEnum.ampelMannLeft] = 0.049 # 0.104 default
                pulse_shifts[SamplesEnum.ampelMannRight] = 0.26
        elif sam_meas_.system == SystemEnum.PIC:
            pulse_shifts = {SamplesEnum.fpSample2: 0.09, SamplesEnum.fpSample3: 0.09,
                            SamplesEnum.fpSample5ceramic: -0.16,
                            SamplesEnum.fpSample6: 0.2, SamplesEnum.bwCeramicBlackUp: 0.01,
                            SamplesEnum.bwCeramicWhiteUp: -0.069,
                            # SamplesEnum.ampelMannRight: -0.080,  # best
                            SamplesEnum.ampelMannRight: -0.080,  # -0.080
                            SamplesEnum.ampelMannLeft: 0.825,  # 0.825 default # 0.60 ? 0.550 ?? 0.850
                            SamplesEnum.opBlackPos1: -0.7,
                            SamplesEnum.opBluePos1: -0.95}
        elif sam_meas_.system == SystemEnum.WaveSource:
            pulse_shifts = {SamplesEnum.ampelMannLeft: 0.45, # 0.49 default
                            SamplesEnum.ampelMannRight: 0.52, # not optimized
                            }
        elif sam_meas_.system == SystemEnum.ECOPS:
            if "even" in str(sam_meas_.file_path):
                pulse_shifts = {SamplesEnum.ampelMannLeft: 0.11, # even sweeps dataset
                                }
            else:
                pulse_shifts = {SamplesEnum.ampelMannLeft: 0.00,
                                }
        else:
            return

        try:
            pulse_shift = pulse_shifts[sam_meas_.sample]
        except KeyError:
            pulse_shift = 0

        sam_meas_.pulse_shift = pulse_shift

        phase_correction = -2 * np.pi * sam_meas_.freq * pulse_shift

        sam_meas_.phase += phase_correction
        sam_meas_.phase_avg += phase_correction

    def fix_tsweeper_offset(self, sam_meas_: Measurement):
        amp_avg, phi_avg = np.abs(sam_meas_.r_avg), np.angle(sam_meas_.r_avg)
        amp, phi = np.abs(sam_meas_.r), np.angle(sam_meas_.r)

        if sam_meas_.system == SystemEnum.TSweeper:
            offsets = {SamplesEnum.blueCube: 2.6,
                       SamplesEnum.fpSample2: -np.mean(phi_avg[540:1650]),
                       SamplesEnum.fpSample3: -1.37,
                       SamplesEnum.fpSample5ceramic: -1.13,
                       SamplesEnum.fpSample5Plastic: 0.39,
                       SamplesEnum.fpSample6: -1.1,
                       SamplesEnum.bwCeramicWhiteUp: 0.20,
                       SamplesEnum.ampelMannRight: 0.29,
                       SamplesEnum.ampelMannLeft: 0.50, # 0.0 default
                       }
        else:
            return

        try:
            offset = offsets[sam_meas_.sample]
        except KeyError:
            offset = 0

        sam_meas_.r_avg = amp_avg * np.exp(1j * (phi_avg + offset))
        sam_meas_.r = amp * np.exp(1j * (phi + offset))

    def fix_phase_offset(self, sam_meas_):
        # We can only do this for the (broadband) TSweeper measurements
        if self.options["selected_system"] != SystemEnum.TSweeper:
            return

        freq = sam_meas_.freq

        if self.options["use_r_meas"] and ("R_" not in str(sam_meas_.file_path)):
            return

        freq_slice = (0.600 < freq) * (freq < 0.800)
        freq, phi_avg = freq[freq_slice], np.angle(sam_meas_.r_avg)[freq_slice]

        phi_avg_uw = np.unwrap(phi_avg, discont=0.8, period=np.pi)  # discont=np.pi*0.8
        coe_avg = polyfit(freq, phi_avg_uw, deg=1)
        print(coe_avg[0], coe_avg[1])

        phi_diffs = np.diff(phi_avg, append=0)
        no_jump_mask = np.abs(phi_diffs) < 0.1

        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, num="phase fix")
        ax0.plot(freq, phi_avg_uw)
        ax0.plot(freq, freq * coe_avg[0] + coe_avg[1], label="avg", ls="--")
        ax1.plot(freq, phi_diffs)
        ax2.plot(freq[no_jump_mask], phi_diffs[no_jump_mask])
        ax0.set_ylabel("Unwrapped phase (rad)")
        ax1.set_ylabel("Phase diff (rad)")
        ax2.set_ylabel("Phase diff masked (rad)")
        ax2.set_xlabel("Frequency (THz)")

    def add_noise(self, meas):
        scale = self.options["noise_scaling"]
        if scale is None or np.isclose(scale, 0.0):
            return

        if meas.r_std is None:
            logging.info("r_std is None. Recalculating the reflection coefficient")
            self.calc_sample_refl_coe()

        f_axis = meas.freq

        rr_std, ri_std = meas.r_std.real, meas.r_std.imag
        noise_r = scale * rr_std
        noise_i = scale * ri_std

        for f_idx in range(len(f_axis)):
            r_avg_real = np.random.normal(meas.r_avg.real[f_idx], noise_r[f_idx], meas.n_sweeps)
            r_avg_imag = np.random.normal(meas.r_avg.imag[f_idx], noise_i[f_idx], meas.n_sweeps)
            meas.r.real[:, f_idx], meas.r.imag[:, f_idx] = r_avg_real, r_avg_imag
            # meas.r_avg.real[f_idx], meas.r_avg.imag[f_idx] = r_avg_real, r_avg_imag

        # can we calculate the noise based on the sample? (it seems to be sam. dep.) lower amp. -> lower noise
        # can set phi for each sweep to phi_avg, to simulate amp noise only, or the other way around.


    def calc_sample_refl_coe(self):
        selected_sample = self.options["selected_sample"]
        sample_meas = [meas for meas in self.measurements["all"] if meas.sample == selected_sample]
        for sam_meas in sample_meas:
            is_r_meas = "R_" in str(sam_meas.file_path.name)[:3]

            ref_meas = find_nearest_meas(sam_meas, self.measurements["refs"])
            self.shift_meas_freq_axis(sam_meas, ref_meas)
            self.fix_phase_slope(sam_meas)

            if is_r_meas:
                amp_ref, phi_ref = 1, 0
                amp_ref_avg, phi_ref_avg = 1, 0
            else:
                amp_ref_avg, phi_ref_avg = ref_meas.amp_avg, ref_meas.phase_avg
                if self.options["use_avg_ref"]:
                    amp_ref, phi_ref = amp_ref_avg, phi_ref_avg
                else:
                    amp_ref, phi_ref = ref_meas.amp, ref_meas.phase

            amp_sam, phi_sam = sam_meas.amp, sam_meas.phase
            amp_sam_avg, phi_sam_avg = sam_meas.amp_avg, sam_meas.phase_avg

            phase_sign_ = -1 # np.sign(ref_meas.freq)
            if sam_meas.system in [SystemEnum.PIC, SystemEnum.WaveSource]:
                phase_sign_ = 1 # np.sign(ref_meas.freq)

            phi_diff, phi_diff_avg = phi_sam - phi_ref, phi_sam_avg - phi_ref_avg
            # phi_diff, phi_diff_avg = np.unwrap(phi_diff), np.unwrap(phi_diff_avg)

            amp_ratio = amp_sam / amp_ref
            amp_ratio_avg = amp_sam_avg / amp_ref_avg

            if sam_meas.system == SystemEnum.TSweeper:
                phi_diff_avg = moving_average(phi_diff_avg, window_size=2)
                amp_ratio_avg = moving_average(amp_ratio_avg, window_size=2)


            sam_meas.r = amp_ratio * np.exp(phase_sign_ * 1j * phi_diff)
            sam_meas.r_avg = amp_ratio_avg * np.exp(phase_sign_ * 1j * phi_diff_avg)

            sam_meas.r = np.nan_to_num(sam_meas.r)
            sam_meas.r_avg = np.nan_to_num(sam_meas.r_avg)

            if sam_meas.system == SystemEnum.TSweeper:
                if options["en_mv_avg"]:
                    for sweep_idx in range(sam_meas.n_sweeps):
                        sam_meas.r[sweep_idx].real = moving_average(sam_meas.r[sweep_idx].real, window_size=3)
                        sam_meas.r[sweep_idx].imag = moving_average(sam_meas.r[sweep_idx].imag, window_size=3)

            self.fix_tsweeper_offset(sam_meas)
            self.fix_phase_offset(sam_meas)

            if sam_meas.n_sweeps != 1:
                sam_meas.r_std = np.std(sam_meas.r.real, axis=0) + 1j * np.std(sam_meas.r.imag, axis=0)

        for sam_meas in sample_meas:
            # fix_r_phi_sign(sam_meas)
            pass

        return sample_meas

    def plot_model_refl_coe(self, thicknesses):
        # thicknesses = [sweep_idx, layer_idx]
        # read optimization result and plot (for publication)
        selected_system = self.options["selected_system"]
        selected_sweep_ = self.options["selected_sweep"]
        selected_sample = self.options["selected_sample"]

        selected_sample.value.set_thicknesses(thicknesses[selected_sweep_])

        font_size = 26
        en_legend = False
        # ts_color, pic_color, mod_color = "tab:blue", "tab:red", "black"
        ts_color, pic_color, mod_color = "blue", "red", "black"

        min_freq, max_freq = 0.04, 1.55

        fig_r_num = f"r_model_{selected_sample.name}_{selected_sweep_}"
        fig_r, (ax0_r, ax1_r) = plt.subplots(nrows=2, ncols=1, num=fig_r_num, sharex=True)
        ax0_r.set_ylabel("Amplitude (dB)", size=font_size)
        ax1_r.set_ylabel("Phase (rad)", size=font_size)
        ax1_r.set_xlabel("Frequency (THz)", size=font_size)
        ax0_r.set_xlim((min_freq - 0.1, max_freq + 0.1))
        ax1_r.set_xlim((min_freq - 0.1, max_freq + 0.1))
        ax0_r.set_ylim((-40, 15))

        ax1_r.yaxis.set_major_locator(MultipleLocator(base=np.pi))
        ax1_r.yaxis.set_major_formatter(FuncFormatter(format_func))
        ax1_r.set_ylim(-np.pi * 1.1, np.pi * 1.1)

        ax0_r.grid(False), ax1_r.grid(False)

        ax0_r.tick_params(axis='both', which='major', labelsize=font_size)
        ax0_r.tick_params(axis='both', which='minor', labelsize=font_size)
        ax1_r.tick_params(axis='both', which='major', labelsize=font_size)
        ax1_r.tick_params(axis='both', which='minor', labelsize=font_size)

        """
        d1_, d2_, d3_ = new_thicknesses
        s = f"Optimization result ({d1_}, {d2_}, {d3_}) $\mu$m"
        s = f"Optimization result"
        ax0_r.annotate(s, xy=(0.232, -36), xytext=(0.35, -35),  # -45 below fig.
                       arrowprops=dict(facecolor=mod_color, shrink=0.12, ec=mod_color),
                       size=font_size-4, c=mod_color, va='center')
        """
        if not en_legend:
            ax0_r.annotate(r"\bf{Full CW-spectrum}", xy=(0.78, -9), xytext=(-0.05, 10),
                           arrowprops=dict(facecolor=ts_color, shrink=0.12, ec=ts_color),
                           size=font_size - 4, c=ts_color, va='top')
            ax0_r.annotate(r"\bf{Selected}\\\bf{frequencies}", xy=(1.20, -5.0), xytext=(0.9, 5.5),
                           arrowprops=dict(facecolor=pic_color, shrink=0.12, ec=pic_color),
                           size=font_size - 4, c=pic_color, ha="center")
            ax0_r.annotate(r"\bf{Fit to selected}\\\bf{frequencies}", xy=(0.9, -24.1), xytext=(1.00, -32.0),
                           arrowprops=dict(facecolor=mod_color, shrink=0.12, ec=mod_color),
                           size=font_size - 4, c=mod_color, ha="left")

        for meas in self.measurements["all"]:
            if meas.sample != selected_sample:
                continue
            limits = (min_freq < meas.freq) * (meas.freq < max_freq)
            freq = meas.freq[limits]

            legend_label = en_legend * str(meas.system.name)

            r = meas.r_avg[limits] if not selected_sweep_ else meas.r[selected_sweep_][limits]
            r_db, r_phi = 20*np.log10(np.abs(r)), np.angle(r)

            if meas.system == SystemEnum.TSweeper:
                ax0_r.plot(freq, r_db, label=legend_label, c=ts_color, zorder=1, lw=7, alpha=0.75, marker="")
                ax1_r.plot(freq, r_phi, label=legend_label, c=ts_color, zorder=1, lw=7, alpha=0.75, marker="")

            f_idx = [np.argmin(np.abs(f - freq)) for f in self.options["freq_selection"]]
            if meas.system == selected_system:  # usually PIC
                ax0_r.scatter(freq[f_idx], r_db[f_idx],
                              label=en_legend*"Selected frequencies", s=100, zorder=3, c=pic_color)
                ax1_r.scatter(freq[f_idx], r_phi[f_idx],
                              label=en_legend*"Selected frequencies", s=100, zorder=3, c=pic_color)

        mod_meas = ModelMeasurement(selected_sample)
        limits = (min_freq < mod_meas.freq) * (mod_meas.freq < max_freq)
        freq = mod_meas.freq[limits]
        legend_label = en_legend * fr"Optimization result\\({thicknesses[selected_sweep_]}) µm"
        r_mod = mod_meas.r_avg[limits]
        ax0_r.plot(freq, 20*np.log10(np.abs(r_mod)), label=legend_label, c=mod_color, lw=4, zorder=2)
        ax1_r.plot(freq, np.angle(r_mod), label=legend_label, c=mod_color, lw=4, zorder=2)

    def plot_sample_refl_coe_simple(self, meas=None):
        selected_system = self.options["selected_system"]
        selected_sample = self.options["selected_sample"]
        selected_sweep_ = self.options["selected_sweep"]

        title = "Reflection coefficient"

        if meas is None:
            shown_systems = [selected_system, SystemEnum.TSweeper, SystemEnum.Model]
            sample_meas = [meas for meas in self.measurements["all"] if
                           (meas.sample == selected_sample and meas.system in shown_systems)]
        else:
            sample_meas = [meas]
            title += f" ({meas.system.name})"

        sweep_s = f"sweep {selected_sweep_}"

        fig_r_num = f"r_{selected_sample.name}_{sweep_s}"
        fig_r, (ax0_r, ax1_r) = plt.subplots(nrows=2, ncols=1, num=fig_r_num)
        ax0_r.set_title(title)
        ax0_r.set_ylabel("Amplitude (dB)")
        ax1_r.set_ylabel("Phase (rad)")
        ax1_r.set_xlabel("Frequency (THz)")
        ax0_r.set_xlim((-0.150, 1.6))
        ax1_r.set_xlim((-0.150, 1.6))
        ax0_r.set_ylim((-40, 15))

        for sam_meas in sample_meas:
            freq = sam_meas.freq

            if sam_meas.n_sweeps == 1:
                r = sam_meas.r_avg
            else:
                r = sam_meas.r_avg if not selected_sweep_ else sam_meas.r[selected_sweep_]

            r_db, r_phi = 20 * np.log10(np.abs(r)), np.angle(r)

            ax0_r.plot(freq, r_db, label="Full spectrum", c="grey", zorder=8)
            ax1_r.plot(freq, r_phi, label="Full spectrum", c="grey", zorder=8)

            f_idx_selection = [np.argmin(np.abs(f - freq)) for f in self.options["freq_selection"]]

            freq = freq[f_idx_selection]
            r_db = r_db[f_idx_selection]
            r_phi = r_phi[f_idx_selection]

            ax0_r.scatter(freq, r_db, label="Selected frequencies", s=40, zorder=9, c="blue")
            ax1_r.scatter(freq, r_phi, label="Selected frequencies", s=40, zorder=9, c="blue")


    def plot_sample_refl_coe(self):
        selected_system = self.options["selected_system"]
        selected_sample = self.options["selected_sample"]
        selected_sweep_ = self.options["selected_sweep"]
        less_plots = self.options["less_plots"]

        shown_systems = [selected_system, SystemEnum.TSweeper, SystemEnum.Model]
        sample_meas = [meas for meas in self.measurements["all"] if
                       (meas.sample == selected_sample and
                        meas.system in shown_systems)]

        if self.options["use_r_meas"] and len(sample_meas) > 1:
            sample_meas = [meas for meas in sample_meas if "R_" in str(meas.file_path.name)]

        sample = selected_sample.value
        layer_cnt = sample.layers

        if selected_sweep_ is None:
            title = f"Reflection coefficient. Sample: {selected_sample.name}, Mean all sweeps"
            sweep_s = f"All sweeps"
        else:
            title = f"Reflection coefficient. Sample: {selected_sample.name}. Sweep {selected_sweep_}"
            sweep_s = f"sweep {selected_sweep_}"

        fig_r_num = f"r_{selected_sample.name}_{sweep_s}"
        fig_r, (ax0_r, ax1_r) = plt.subplots(nrows=2, ncols=1, num=fig_r_num)
        fig_r.subplots_adjust(left=0.05 + 0.05 * layer_cnt, bottom=0.15 + 0.05 * layer_cnt)
        ax0_r.set_title(title)
        ax0_r.set_ylabel("Amplitude (dB)")
        ax1_r.set_ylabel("Phase (rad)")
        ax1_r.set_xlabel("Frequency (THz)")
        ax0_r.set_xlim((-0.150, 1.6))
        ax1_r.set_xlim((-0.150, 1.6))
        ax0_r.set_ylim((-40, 15))

        n_sliders, k_sliders, d_sliders = [], [], []
        n_slider_axes, k_slider_axes, d_slider_axes = [], [], []
        for layer_idx in range(layer_cnt):
            layer_n, layer_k = sample.ref_idx[layer_idx].real, sample.ref_idx[layer_idx].imag
            layer_n_min, layer_n_max = layer_n[0], layer_n[1]
            layer_k_min, layer_k_max = layer_k[0], layer_k[1]

            if np.isclose(layer_n_min, layer_n_max):
                layer_n_max += 0.001

            if np.isclose(layer_k_min, layer_k_max):
                layer_k_max += 0.0001

            n_slider_axes.append(fig_r.add_axes([0.15, 0.10 - 0.05 * layer_idx, 0.16, 0.03]))
            label = fr"n_{layer_idx + 1}"
            n_slider = RangeSlider(ax=n_slider_axes[layer_idx],
                                   label=label,
                                   valmin=layer_n_min * 0.95,
                                   valmax=layer_n_max * 1.05,
                                   valinit=(layer_n_min, layer_n_max),
                                   )
            n_sliders.append(n_slider)

            k_slider_axes.append(fig_r.add_axes([0.60, 0.10 - 0.05 * layer_idx, 0.16, 0.03]))
            label = fr"k_{layer_idx + 1}"
            k_slider = RangeSlider(ax=k_slider_axes[layer_idx],
                                   label=label,
                                   valmin=0,
                                   valmax=np.abs(layer_k_max) * 1.2,
                                   valinit=(np.abs(layer_k_min), np.abs(layer_k_max)),
                                   )
            k_sliders.append(k_slider)

            layer_thickness = sample.thicknesses[layer_idx]

            d_slider_axes.append(fig_r.add_axes([0.03 + 0.05 * layer_idx, 0.20, 0.03, 0.60]))
            d_slider = Slider(ax=d_slider_axes[layer_idx],
                              label=fr"Thickness\\layer\\{layer_idx + 1}",
                              valmin=layer_thickness * 0.5,
                              valmax=layer_thickness * 2.0,
                              valinit=layer_thickness,
                              orientation='vertical',
                              )
            d_sliders.append(d_slider)

        resetax0 = fig_r.add_axes([0.005, 0.01, 0.05, 0.04])
        reset_but = Button(resetax0, 'Reset', hovercolor='0.975')

        def reset0(event):
            for slider_ in (n_sliders + k_sliders + d_sliders):
                slider_.reset()

        reset_but.on_clicked(reset0)

        for sam_meas in sample_meas:
            ref_meas = find_nearest_meas(sam_meas, self.measurements["refs"])

            legend_label = str(sam_meas.system.name)
            if self.options["debug_info"]:
                legend_label += fr"\\Pulse shift {sam_meas.pulse_shift}"

            freq = sam_meas.freq

            if sam_meas.n_sweeps == 1:
                r = sam_meas.r_avg
            else:
                r = sam_meas.r_avg if not selected_sweep_ else sam_meas.r[selected_sweep_]

            r_db, r_phi = 20 * np.log10(np.abs(r)), np.angle(r)
            """
            r_phi_unwrapped = np.unwrap(np.angle(r))
            plt.figure("unwrapped phase")
            plt.plot(freq, r_phi_unwrapped)
            plt.plot(freq, np.angle(r))
            """
            if sam_meas.system == SystemEnum.TSweeper:
                ax0_r.plot(freq, r_db, label="TSweeper (Full spectrum)", c="grey", zorder=8)
                ax1_r.plot(freq, r_phi, label="TSweeper (Full spectrum)", c="grey", zorder=8)
            else:
                if len(r_db) < 10:
                    ax0_r.scatter(freq, r_db, label=legend_label, s=40, zorder=9, c="red")
                    ax1_r.scatter(freq, r_phi, label=legend_label, s=40, zorder=9, c="red")
                else:
                    ax0_r.plot(freq, r_db, label=legend_label, zorder=9, c="red")
                    ax1_r.plot(freq, r_phi, label=legend_label, zorder=9, c="red")

            f_idx_selection = [np.argmin(np.abs(f - freq)) for f in self.options["freq_selection"]]

            freq = freq[f_idx_selection]
            r_db = r_db[f_idx_selection]
            r_phi = r_phi[f_idx_selection]

            #ax0_r.scatter(freq, r_db, label=legend_label + " (selected)", s=40, zorder=9, c="blue")
            #ax1_r.scatter(freq, r_phi, label=legend_label + " (selected)", s=40, zorder=9, c="blue")

            amp_ref, phi_ref = ref_meas.amp, ref_meas.phase
            amp_sam, phi_sam = sam_meas.amp, sam_meas.phase

            if sam_meas.n_sweeps != 1 and not less_plots:
                fig, (ax0, ax1) = plt.subplots(2, 1, num=str(ref_meas))
                ax0.set_title(f"{ref_meas}")
                ax1.set_xlabel("Sweep index")
                ax0.set_ylabel("Amplitude (dB)")
                ax1.set_ylabel("Phase (rad)")

                for i, freq in enumerate(ref_meas.freq):
                    ax0.plot(20 * np.log10(amp_ref[:, i]), label=f"{np.round(freq, 2)} THz")
                    ax1.plot(phi_ref[:, i], label=f"{np.round(freq, 2)} THz")

                fig, (ax0, ax1) = plt.subplots(2, 1, num=str(sam_meas))
                ax0.set_title(f"{sam_meas}")
                ax1.set_xlabel("Sweep index")
                ax0.set_ylabel("Amplitude (dB)")
                ax1.set_ylabel("Phase (rad)")

                for i, freq in enumerate(sam_meas.freq):
                    ax0.plot(20 * np.log10(amp_sam[:, i]), label=f"{np.round(freq, 2)} THz")
                    ax1.plot(phi_sam[:, i], label=f"{np.round(freq, 2)} THz")
            # consider making two functions
            if (sam_meas.n_sweeps == 1) and not less_plots:
                plt.figure("TSWeeper amp")
                plt.title("TSWeeper amp")
                plt.xlabel("Frequency (THz)")
                plt.ylabel("Amplitude (dB)")
                plt.xlim((-0.150, 2.1))

                plt.plot(ref_meas.freq[:-2], 20 * np.log10(ref_meas.amp[:-2]), label=ref_meas)
                plt.plot(sam_meas.freq[:-2], 20 * np.log10(sam_meas.amp[:-2]), label=sam_meas)
                # plt.plot(bkg_meas.freq, np.log10(np.abs(bkg_meas.amp)), label="background")

                plt.figure("TSWeeper phase")
                plt.title("TSWeeper phase")
                plt.xlabel("Frequency (THz)")
                plt.ylabel("Phase (rad)")
                plt.xlim((-0.150, 2.1))

                plt.plot(ref_meas.freq[:-2], ref_meas.phase[:-2], label=ref_meas)
                plt.plot(sam_meas.freq[:-2], sam_meas.phase[:-2], label=sam_meas)
                # plt.plot(bkg_meas.freq, bkg_meas.phase, label="background")

            if less_plots or sam_meas.n_sweeps <= 1:
                continue

            fig_num = f"r_all_sweeps_{sam_meas.system.name}_{sam_meas.sample.name}"
            freq_idx = None

            title = f"Reflection coefficient {sam_meas.sample.name}. {sam_meas.system.name} all sweeps"
            if freq_idx is not None:
                title += f" {np.round(sam_meas.freq[freq_idx], 2)} THz"

            if not plt.fignum_exists(fig_num):
                fig, (ax0, ax1) = plt.subplots(2, 1, num=fig_num)
                ax0.set_title(title)
                ax1.set_xlabel("Aufnahme Index")
                ax0.set_ylabel("Amplitude (dB)")
                ax1.set_ylabel("Phase (rad)")
            else:
                fig = plt.figure(fig_num)
                ax0, ax1 = fig.get_axes()

            if sam_meas.n_sweeps != 1:
                r_amp_db, r_phi = 20 * np.log10(np.abs(sam_meas.r)), np.angle(sam_meas.r)

                print(sam_meas.system)
                print(f"Frequencies: {sam_meas.freq}")
                print(f"ref mean phase: {ref_meas.phase_avg}±{np.std(ref_meas.phase, axis=0)}")
                print(f"sample mean phase: {sam_meas.phase_avg}±{np.std(sam_meas.phase, axis=0)}")
                print(f"r direct mean Amp. {np.mean(r_amp_db, axis=0)}±{np.std(r_amp_db, axis=0)}")
                print(f"r direct mean phase: {np.mean(r_phi, axis=0)}±{np.std(r_phi, axis=0)}\n")
                for i, freq in enumerate(sam_meas.freq):
                    if freq_idx and (i != freq_idx):
                        continue
                    ax0.plot(r_amp_db[:, i], label=f"{np.round(freq, 2)} THz")
                    ax1.plot(r_phi[:, i], label=f"{np.round(freq, 2)} THz")

        mod_meas = ModelMeasurement(self.options["selected_sample"])
        if mod_meas:
            legend_label = str(mod_meas.system.name)
            r_mod = mod_meas.r_avg
            mod_amp_line, = ax0_r.plot(mod_meas.freq, 20 * np.log10(np.abs(r_mod)), label=legend_label, c="black")
            mod_phi_line, = ax1_r.plot(mod_meas.freq, np.angle(r_mod), label=legend_label, c="black")

            def update(val):
                new_ref_idx = []
                for layer_idx_ in range(layer_cnt):
                    n_min = n_sliders[layer_idx_].val[0] - 1j * k_sliders[layer_idx_].val[0]
                    n_max = n_sliders[layer_idx_].val[1] - 1j * k_sliders[layer_idx_].val[1]
                    new_ref_idx.append((n_min, n_max))

                sample.set_ref_idx(new_ref_idx)

                new_thicknesses = [d_slider_.val for d_slider_ in d_sliders]
                sample.set_thicknesses(new_thicknesses)

                mod_meas.simulate_sam_measurement(fast=True)
                freqs = mod_meas.freq
                new_amp, new_phi = 20 * np.log10(np.abs(mod_meas.r_avg)), np.angle(mod_meas.r_avg)

                mod_amp_line.set_data(freqs, new_amp)
                mod_phi_line.set_data(freqs, new_phi)
                fig_r.canvas.draw_idle()

            for slider in n_sliders + k_sliders + d_sliders:
                slider.on_changed(update)

        plt_show(mpl, en_save=self.options["en_save"])

    def plot_noise(self):
        selected_sample = self.options["selected_sample"]
        sam_meas_ = [meas for meas in self.measurements["all"] if meas.sample == selected_sample][0]
        # ref = find_nearest_meas(sam_meas_, self.measurements["refs"])
        # sam_meas_ = ref

        freqs = sam_meas_.freq

        f_idx_sel = [np.argmin(np.abs(f - freqs)) for f in self.options["freq_selection"]]

        min_amp, max_amp = np.min(sam_meas_.amp, axis=0), np.max(sam_meas_.amp, axis=0)
        min_phase, max_phase = np.min(sam_meas_.phase, axis=0), np.max(sam_meas_.phase, axis=0)

        r = sam_meas_.r

        real_weights = 1 / np.std(r.real, axis=0)
        imag_weights = 1 / np.std(r.imag, axis=0)

        r_amp_db, r_phi = 20 * np.log10(np.abs(r)), np.angle(r)

        np.savez(f"noise_{selected_sample.name}",
                 freqs=freqs, std_r=np.std(r.real), std_i=r.imag)
        _, ax_weights = plt.subplots(num="Weights (std noise)")
        ax_weights.set_title(f"Number of sweeps: {sam_meas_.n_sweeps} {sam_meas_}")
        ax_weights.set_xlabel("Frequency (THz)")
        ax_weights.set_ylabel("1/std(sweeps)")
        range_ = (0.05 < freqs) * (freqs < 1.6)
        ax_weights.plot(freqs[range_], real_weights[range_], label=f"Real weights")
        ax_weights.plot(freqs[range_], imag_weights[range_], label=f"Imaginary weights")
        ax_weights.scatter(freqs[f_idx_sel], real_weights[f_idx_sel],
                           label=f"Real (@sel freq)", c="red", s=30, zorder=3)
        ax_weights.scatter(freqs[f_idx_sel], imag_weights[f_idx_sel],
                           label=f"Imag (@sel freq)", c="black", s=30, zorder=2)

        _, (ax_00, ax_01) = plt.subplots(2, 1, num="Refl. coef. extrema (dB, phase)")
        ax_00.set_title(f"Number of sweeps: {sam_meas_.n_sweeps} {sam_meas_}")
        ax_00.set_xlabel("Frequency (THz)")
        ax_00.set_ylabel("Reflection coefficient amplitude (dB)")
        range_ = (0.05 < freqs) * (freqs < 1.6)
        ax_00.plot(freqs[range_], np.min(r_amp_db, axis=0)[range_], label=f"Min")
        ax_00.plot(freqs[range_], np.max(r_amp_db, axis=0)[range_], label=f"Max")
        ax_00.scatter(freqs[f_idx_sel], np.min(r_amp_db, axis=0)[f_idx_sel],
                      label=f"Min (@sel freq)", c="red", s=30, zorder=3)
        ax_00.scatter(freqs[f_idx_sel], np.max(r_amp_db, axis=0)[f_idx_sel],
                      label=f"Max (@sel freq)", c="black", s=30, zorder=2)

        ax_01.set_xlabel("Frequency (THz)")
        ax_01.set_ylabel("Reflection coefficient phase (rad)")
        range_ = (0.05 < freqs) * (freqs < 1.6)
        ax_01.plot(freqs[range_], np.min(r_phi, axis=0)[range_], label=f"Min")
        ax_01.plot(freqs[range_], np.max(r_phi, axis=0)[range_], label=f"Max")
        ax_01.scatter(freqs[f_idx_sel], np.min(r_phi, axis=0)[f_idx_sel],
                      label=f"Min (@sel freq)", c="red", s=30, zorder=3)
        ax_01.scatter(freqs[f_idx_sel], np.max(r_phi, axis=0)[f_idx_sel],
                      label=f"Max (@sel freq)", c="black", s=30, zorder=2)

        _, (ax_00, ax_01) = plt.subplots(2, 1, num="Refl. coef. extrema")
        ax_00.set_title(f"Number of sweeps: {sam_meas_.n_sweeps} {sam_meas_}")
        ax_00.set_xlabel("Frequency (THz)")
        ax_00.set_ylabel("Reflection coefficient (real)")
        range_ = (0.05 < freqs) * (freqs < 1.6)
        ax_00.plot(freqs[range_], np.min(r.real, axis=0)[range_], label=f"Min. real")
        ax_00.plot(freqs[range_], np.max(r.real, axis=0)[range_], label=f"Max. real")
        ax_00.scatter(freqs[f_idx_sel], np.min(r.real, axis=0)[f_idx_sel],
                     label=f"Real (@sel freq)", c="red", s=30, zorder=3)
        ax_00.scatter(freqs[f_idx_sel], np.max(r.real, axis=0)[f_idx_sel],
                     label=f"Imag (@sel freq)", c="black", s=30, zorder=2)

        ax_01.set_xlabel("Frequency (THz)")
        ax_01.set_ylabel("Reflection coefficient (imag)")
        range_ = (0.05 < freqs) * (freqs < 1.6)
        ax_01.plot(freqs[range_], np.min(r.imag, axis=0)[range_], label=f"Min. imag.")
        ax_01.plot(freqs[range_], np.max(r.imag, axis=0)[range_], label=f"Max. imag.")
        ax_01.scatter(freqs[f_idx_sel], np.min(r.imag, axis=0)[f_idx_sel],
                      label=f"Real (@sel freq)", c="red", s=30, zorder=3)
        ax_01.scatter(freqs[f_idx_sel], np.max(r.imag, axis=0)[f_idx_sel],
                      label=f"Imag (@sel freq)", c="black", s=30, zorder=2)


        _, ax_0 = plt.subplots(num="Refl. coef. std.")
        ax_0.set_title(f"Number of sweeps: {sam_meas_.n_sweeps} {sam_meas_}")
        ax_0.set_xlabel("Frequency (THz)")
        ax_0.set_ylabel("Reflection coefficient std.")
        range_ = (0.05 < freqs) * (freqs < 1.6)
        ax_0.plot(freqs[range_], 1/real_weights[range_], label=f"Real")
        ax_0.plot(freqs[range_], 1/imag_weights[range_], label=f"Imaginary")
        ax_0.scatter(freqs[f_idx_sel], 1/real_weights[f_idx_sel],
                           label=f"Real (@sel freq)", c="red", s=30, zorder=3)
        ax_0.scatter(freqs[f_idx_sel], 1/imag_weights[f_idx_sel],
                           label=f"Imag (@sel freq)", c="black", s=30, zorder=2)

        _, ax_1 = plt.subplots(num="Amplitude noise")
        ax_1.set_title(f"Number of sweeps: {sam_meas_.n_sweeps} {sam_meas_}")
        ax_1.set_xlabel("Frequency (THz)")
        ax_1.set_ylabel("Amplitude extrema")
        range_ = (0.05 < freqs) * (freqs < 1.6)
        ax_1.plot(freqs[range_], min_amp[range_], label=f"Min. amplitude")
        ax_1.plot(freqs[range_], max_amp[range_], label=f"Max. amplitude")
        ax_1.scatter(freqs[f_idx_sel], min_amp[f_idx_sel],
                           label=f"Min. amp. (@sel freq)", c="red", s=30, zorder=3)
        ax_1.scatter(freqs[f_idx_sel], max_amp[f_idx_sel],
                           label=f"Max. amp. (@sel freq)", c="black", s=30, zorder=2)

        _, ax_2 = plt.subplots(num="Phase noise")
        ax_2.set_title(f"Number of sweeps: {sam_meas_.n_sweeps} {sam_meas_}")
        ax_2.set_xlabel("Frequency (THz)")
        ax_2.set_ylabel("Phase extrema")
        range_ = (0.05 < freqs) * (freqs < 1.6)
        ax_2.plot(freqs[range_], min_phase[range_], label=f"Min. phase")
        ax_2.plot(freqs[range_], sam_meas_.phase_avg[range_], label=f"Mean phase", c="grey")
        ax_2.plot(freqs[range_], max_phase[range_], label=f"Max. phase")
        ax_2.scatter(freqs[f_idx_sel], min_phase[f_idx_sel],
                     label=f"Min. phase (@sel freq)", c="red", s=30, zorder=3)
        ax_2.scatter(freqs[f_idx_sel], max_phase[f_idx_sel],
                     label=f"Max. phase (@sel freq)", c="black", s=30, zorder=2)

    def integrated_std_err_plot(self, std_err_=None):
        selected_sample_ = self.options["selected_sample"]
        dt = 0.960  # ms
        n_sweeps_ = 2000
        layer_cnt = selected_sample_.value.layers
        std_err_dict = {SamplesEnum.ampelMannLeft: [0.02, 0.02, 0.06],
                        SamplesEnum.ampelMannRight: [0.12, 0.10, 0.07]}
        if std_err_ is None:
            try:
                std_err_ = np.array(std_err_dict[selected_sample_], dtype=float)
            except KeyError:
                std_err_ = np.zeros(layer_cnt)

        sigma = std_err_ * np.sqrt(n_sweeps_)
        n = np.arange(1, 20 + 1, 1)

        plt.figure(f"Integrated_standard_error_{selected_sample_.name}")
        plt.title("Standarderror as a function of measurement time")
        plt.xlabel("Measurement time (ms)")
        plt.ylabel(r"Standarderror ($\mu$m)")

        for layer_idx in range(layer_cnt):
            std_err_ = sigma[layer_idx] / np.sqrt(n)
            label = fr"Layer {layer_idx + 1} ($\sigma =${np.round(sigma[layer_idx], 2)} $\mu$m)"
            plt.plot(n * dt, std_err_, label=label, lw=3.0, markersize=5)


def change_freq_selection(options_, rnd_sel_size=8):
    freq_sel = np.array(options_["freq_selection"], dtype=float)
    if len(freq_sel) > 1000:
        return options_

    all_freqs = np.arange(0.08, 1.5, 0.001)

    changed_selection = np.sort([*freq_sel,
                                          #*(freq_sel + 0.001),
                                          #*(freq_sel + 0.002),
                                          #*(freq_sel + 0.003),
                                          #*(freq_sel[:4] - 0.001),
                                          ])
    changed_selection = np.sort([*freq_sel,
                                          #*(freq_sel + 0.001),
                                          #*(freq_sel - 0.001),
                                          #*(freq_sel + 0.002),
                                          #*(freq_sel[:4] - 0.002),
                                          ])

    # options_["freq_selection"] = np.sort([*freq_sel, *random.sample(list(all_freqs), 114)])
    # options_["freq_selection"] = np.sort([*np.random.choice(all_freqs, 6, replace=False)])

    rnd_sel = [*np.random.choice(freq_sel, rnd_sel_size, replace=False)]
    changed_selection = np.sort(rnd_sel)

    if len(changed_selection) < 10:
        f_sel = changed_selection
        logging.warning(f"Used frequencies: {f_sel} THz")

    ret_options = deepcopy(options_)
    ret_options["freq_selection"] = changed_selection

    return ret_options

def compile_and_run(options_, run_id_str=None):
    make_dir = options_["c_proj_path"]

    fsel_list = list(options_["freq_selection"])
    freq_cnt = len(fsel_list)

    if run_id_str is None:
        run_id_str = "_".join(f"{f_}" for f_ in fsel_list)

    print(f"Running make with ID={run_id_str}, FCNT={freq_cnt}")
    args = ["make", f"ID={run_id_str}", f"FCNT={freq_cnt}", "simple"]
    result = subprocess.run(args,
                            cwd=make_dir,
                            capture_output=True,
                            text=True,
                            )

    if result.returncode == 0:
        print(f"✅ Success")
    else:
        print(f"❌ Error")
        print(result.stderr)


def setup_meas(options_):

    if options_["sim_meas"]:
        meas_ = simulated_measurements(options_)
    else:
        jl = JumpingLaserEval(options_)
        jl.options["en_print"] = 1
        sam_meas_list = jl.calc_sample_refl_coe()
        if options_["use_r_meas"] and len(sam_meas_list) > 1:
            meas_ = sam_meas_list[1]
        else:
            meas_ = sam_meas_list[0]
        print(f"Using measurement {meas_}")

    return meas_

def eval_impl(options_):

    if options_["sim_meas"]:
        measurements = setup_meas(options_)
        for meas in measurements:
            calculate_coe(meas, options_)
            compile_and_run(options_)
    else:
        meas = setup_meas(options_)
        calculate_coe(meas, options_)
        compile_and_run(options_)


def freq_var(options_):
    max_run_cnt = 100
    all_freqs = np.array(options_["freq_selection"], dtype=float)

    def gen_unique_f_sel(size=8):
        sel_ret = []
        max_selection_cnt = comb(len(all_freqs), size, exact=True)

        while len(sel_ret) < min(max_run_cnt, max_selection_cnt):
            rnd_sel = np.random.choice(all_freqs, size, replace=False)
            rnd_sel_sorted = np.sort(rnd_sel)

            if not any(np.array_equal(rnd_sel_sorted, arr) for arr in sel_ret):
                sel_ret.append(rnd_sel_sorted)

        return sel_ret

    new_options = deepcopy(options_)
    for sel_size in range(8, 9):
        np.random.seed(sel_size)
        selections = gen_unique_f_sel(sel_size)
        for freq_sel in selections:
            new_options["freq_selection"] = freq_sel
            if options_["sim_meas"]:
                simulated_measurements(new_options)
            else:
                calculate_coe(new_options, sweep_cnt=5000)

            compile_and_run(new_options)

if __name__ == '__main__':

    options = {"selected_system": SystemEnum.WaveSource,
               "selected_sample": SamplesEnum.ampelMannLeft,
               "en_save": 0,
               "selected_sweep": 10,  # selected_sweep: int, None(=average) or "random"
               "less_plots": 1,
               "debug_info": 0,
               "c_proj_path": Path(""), # TODO set correct path

               "use_r_meas": 0,  # r_dataset exists only for TSweeper
               "use_avg_ref": 1,
               "mean_start_idx": 0,
               "freq_selection": [0.05, 0.15, 0.20, 0.60, 0.75, 0.80, 1.00, 1.5],  # Wavesource, links
               "calc_f_axis_shift": 0.0,
               "sim_meas": False,
               }

    freq_var(options)
