import numpy as np
from enum import Enum
from tmm import interface_r, list_snell

class Sample:
    thickness = None
    layers = None
    tot_thickness = None
    ref_idx = None
    name = None

    def __init__(self, thicknesses: list, ref_idx=None, core=False):
        # list thicknesses given in mm then converted to um. (Dicke 1 [mm] in case of single layer samples)
        self.thicknesses = np.array(thicknesses) * 1e3  # treat "OP" samples as single layers (ignoring iron core)
        self.tot_thickness = np.sum(self.thicknesses)
        self.layers = len(thicknesses)
        if ref_idx is not None:
            self.ref_idx = np.array(ref_idx, dtype=complex)
        else:
            self.ref_idx = np.array(1.5 * np.ones_like(self.thicknesses), dtype=complex)
        self.has_iron_core = core

    def __repr__(self):
        ri_s = "".join([f"   Layer {i + 1}: {n_}\n" for i, n_ in enumerate(self.ref_idx)])
        return f"Thicknesses: {self.thicknesses}\nRefractive index:\n{ri_s}"

    def set_thicknesses(self, new_thicknesses):
        # new_thicknesses should be in um
        self.thicknesses = np.array(new_thicknesses, dtype=float)
        self.tot_thickness = np.sum(new_thicknesses)

    def set_ref_idx(self, ref_idx):
        self.ref_idx = np.array(ref_idx, dtype=complex)

    def get_ref_idx(self, selected_freqs=None, shift=None):
        if shift is None:
            shift = self.layers * [0.0]
        freq_axis = np.arange(-0.200, 5.000, 0.001)
        one = np.ones_like(freq_axis)
        sample_ref_idx = np.zeros((self.layers, len(freq_axis)), dtype=complex)
        n0 = self.ref_idx

        fa_idx, fe_idx = np.argmin(np.abs(freq_axis - -0.11)), np.argmin(np.abs(freq_axis - 2.000))
        for i in range(self.layers):
            n_min, n_max = n0[i][0], n0[i][1]

            n_r = np.linspace(n_min.real, n_max.real, fe_idx - fa_idx)
            n_i = np.linspace(n_min.imag, n_max.imag, fe_idx - fa_idx)
            sample_ref_idx[i, :fa_idx] = np.ones(fa_idx)
            sample_ref_idx[i, fa_idx:fe_idx] = n_r + 1j * n_i
            sample_ref_idx[i, fe_idx:] = np.ones(len(freq_axis) - fe_idx)

        if self.has_iron_core:
            n_fe = (500 - 500j) * one
            n = np.array([one, *sample_ref_idx, n_fe, one], dtype=complex).T
        else:
            n = np.array([one, *sample_ref_idx, one], dtype=complex).T

        if selected_freqs is not None:
            sel_freq_idx = [np.argmin(np.abs(freq_axis - sel_freq)) for sel_freq in selected_freqs]
            n = n[sel_freq_idx, :]

        for i in range(self.layers):
            n[:, i+1] += shift[i]

        return n

    def get_r(self, options_):
        freqs = options_["freqs"]
        th_0 = options_["th_0"]
        pol = options_["pol"]

        num_layers = self.layers + 2
        n_list = self.get_ref_idx(selected_freqs=freqs)

        r_list = np.zeros((len(freqs), num_layers, num_layers), dtype=complex)
        for f_idx in range(len(freqs)):
            th_list = list_snell(n_list[f_idx], th_0)

            for i in range(num_layers - 1):
                r_list[f_idx, i, i + 1] = interface_r(pol, n_list[f_idx, i], n_list[f_idx, i + 1],
                                                      th_list[i], th_list[i + 1])

        return r_list


class SamplesEnum(Enum):
    empty = Sample([0.0])

    # 1 layer
    blueCube = Sample([30.000], [(1.54 - 0.005j, 1.54 - 0.005j)])  # probably ifft -> window -> fft
    fpSample2 = Sample([4.000], [(1.683 - 0.0192j, 1.699 - 0.024j)])  # WORKS
    fpSample3 = Sample([1.150], [(1.676 - 0.0092j, 1.68 - 0.0945j)])  # WORKS
    fpSample5Plastic = Sample([5.200], [(1.2 - 0.0001j, 1.9 - 0.0080j)])  # doesnt work
    fpSample5ceramic = Sample([1.600], [(2.307 - 0.00334j, 2.330 - 0.012j)])  # WORKS best
    fpSample6 = Sample([0.600], [(1.34 - 0.028j, 1.370 - 0.15j)])  # kriege ich nicht gut hin ...

    # 1 layer + mirror core
    opBluePos1 = Sample([0.210], [(1.747 - 0.0001j, 1.980 - 0.065j)], True)  # works
    opBluePos2 = Sample([0.295], [(2.25, 2.25)], True)
    opBlackPos1 = Sample([0.172], [(1.597 - 0.0001j, 1.73 - 0.065j)], True)  # works Lauri d1: 0.145
    opBlackPos2 = Sample([0.210], [(1.93, 1.93)], True)
    opRedPos1 = Sample([0.235], [(1.93, 1.93)], True)  # not done
    opRedPos2 = Sample([0.335], [(1.93, 1.93)], True)
    opDarkRedPos1 = Sample([0.285], [(1.93, 1.93)], True)
    opDarkRedPos2 = Sample([0.385], [(1.1 - 0.0001j, 2.73 - 0.065j)], True)
    opToolRedPos1 = Sample([0.235], [(1.93, 1.93)], True)
    opToolRedPos2 = Sample([0.335], [(1.93, 1.93)], True)
    opToolBluePos1 = Sample([0.210], [(1.93, 1.93)], True)
    opToolBluePos2 = Sample([0.295], [(2.25, 2.25)], True)

    # 2 layer
    # WORKS? [(2.911 - 0.001j, 2.950 - 0.059j), (2.685 - 0.001j, 2.722 - 0.0636j)]
    bwCeramicWhiteUp = Sample([0.500, 0.140],
                              [(2.911 - 0.001j, 2.950 - 0.0236j),
                               (2.685 - 0.001j, 2.737 - 0.0236j)])
    # WORKS
    bwCeramicBlackUp = Sample([0.140, 0.500],
                              [(2.685 - 0.001j, 2.737 - 0.0236j),
                               (2.911 - 0.001j, 2.950 - 0.0236j),
                               ])

    # 3 layer
    ampelMannRight = Sample([0.045, 0.660, 0.076],
                            [  # (1.504, 1.54),
                                (1.527 - 0.000j, 1.532 - 0.11j),
                                (2.78 - 0.000j, 2.85 - 0.025j),
                                #(1.504, 1.54)
                                (1.527 - 0.000j, 1.532 - 0.11j)])  # WORKS


    ampelMannLeft = Sample([0.073, 0.660, 0.042],
                           [
                               (1.59 - 0.000j, 1.6 - 0.001j),
                               (2.81 - 0.00j, 2.83 - 0.001j),
                               (1.59 - 0.000j, 1.6 - 0.001j),
                           ])
    ampelMannOld = Sample([0.046, 0.660, 0.073], [(1.52, 1.521),
                                                  (2.78 - 0.000j, 2.78 - 0.015j),
                                                  (1.52, 1.521)])


if __name__ == '__main__':
    print(SamplesEnum.blueCube.name)
