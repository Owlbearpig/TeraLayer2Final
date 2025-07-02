from numpy import (pi, array, round, sqrt, sign, cos, sin, exp, array,
                   arcsin, conj, sum, outer, ones, inf, zeros)
from numpy.random import uniform
from scipy.special import factorial
from pathlib import Path
import numpy as np
import os
from scipy.constants import c as c0

c_thz = c0 * 10**-6  # um / ps

cur_os = os.name

settings = {
    'data_range_idx': (234, -2)
}

Omega_, Delta_, sigma_, mu_, epsilon_, degree_ = '\u03BC', '\u0394', '\u03C3', '\u03BC', '\u03B5', '\u00B0'

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
if 'posix' in cur_os:
    data_dir = Path(r"/home/ftpuser/ftp/Data/TeraLayer2")
else:
    data_dir = Path(r"E:\measurementdata\TeraLayer2")
    try:
        os.scandir(data_dir)
    except FileNotFoundError:
        data_dir = Path(r"C:\Users\Laptop\Desktop")

matlab_data_dir = Path(ROOT_DIR / 'matlab_enrique' / 'Data')
optimization_results_dir = Path(ROOT_DIR / 'measurementComparisonResults')
hhi_data_dir = Path(ROOT_DIR / "data" / "T-Sweeper_and_TeraFlash" /
                    "Lackierte Keramik" / "CW (T-Sweeper)" / "Kopf_Ahmad_3")

if 'posix' in cur_os:
    base_dir = Path(r"/home/ftpuser/ftp/Data/TeraLayer2")
    data_dir = base_dir / "Foil_glue" / "Img0"
    result_dir = Path(r"/home/alex/MEGA/AG/Projects/TeraLayer/Publication/2")
    # result_dir = Path(r"/home/alex/MEGA/AG/Projects/TeraLayer/Endreport/Figures")
else:
    data_dir = Path(r"")
    result_dir = Path(r"E:\Mega\AG\Projects\TeraLayer\Publication\2")
    base_dir = Path(r"E:\measurementdata\TeraLayer2")
    # result_dir = Path(r"E:\Mega\AG\Projects\TeraLayer\Endreport\Figures")

if os.name != "posix":
    op_besteck_dir = Path(r"E:\measurementdata\TeraLayer2\OP-Besteck\Blaues Teil")
else:
    op_besteck_dir = Path(r"/home/alex/Data/TeraLayer2/OP-Besteck/Blaues Teil")

data_file_cnt = 100

rad = 180 / pi
# thea = 8 * pi / 180
thea = 8 * pi / 180
a = 1

"""
20 1/cm POM 0.0477
0.25 1/cm PTFE 0.0006
@1 THz
5 1/cm POM 0.0120
~0.25 1/cm PTFE 0.0006
@0.5 THz
"""

#n = [1, 1.50, 2.8, 1.50, 1]
#n = [1, 1.35+0.0006*1j, 1.68+0.0120*1j, 1, 1]
#n = [1, 1.35, 1.68, 1, 1]
n = [1, 1.50, 2.80, 1.50, 1]

# n0 = array([1.56, 1.57, 1.60, 1.61, 1.62, 1.62])
# n1 = array([2.88, 2.89, 2.89, 2.90, 2.88, 2.89])
# n2 = array([1.56, 1.57, 1.60, 1.61, 1.62, 1.62])

# matches "original" frequency set
n0 = array([1.513, 1.515, 1.520, 1.521, 1.522, 1.524])
n1 = array([2.782, 2.782, 2.784, 2.785, 2.786, 2.787])
n2 = array([1.513, 1.515, 1.520, 1.521, 1.522, 1.524])
n_ = array([n0, n1, n2]).T

# wavesource ("manual") set (8 frequencies)
n0 = array([1.513, 1.515, 1.520, 1.521, 1.522, 1.524, 1.522, 1.524])
n1 = array([2.782, 2.782, 2.784, 2.785, 2.786, 2.787, 2.786, 2.787])
n2 = array([1.513, 1.515, 1.520, 1.521, 1.522, 1.524, 1.522, 1.524])
n_ = array([n0, n1, n2]).T

nm = 10 ** 9
um = 10 ** 6
MHz = 10 ** 6
GHz = 10 ** 9
THz = 10 ** 12

um_to_m = 1 / um

# ni, nf, nn = 400, 640, 40  # 0-4500 # original data indices
ni, nf, nn = 400, 640, 40  # 0-4500

default_mask = np.arange(ni, nf, nn)
default_mask_hr = np.arange(ni, nf, 1)  # default mask high rez

wide_mask = np.arange(250, 1000, 40)
all_freqs_lowend = np.arange(0, 1000, 1)
full_range_mask = np.arange(250, 1000, 1)
full_range_mask_new = np.arange(420, 1000, 1)  # based on plot of reference and background, big water line at 1 THz
full_range_mask_new_low_rez = np.arange(450, 1000, 100)
custom_mask_420 = array([420, 520, 650, 800, 850, 950])
high_freq_mask = np.arange(250, 600, 1)
high_freq_mask_low_rez = np.arange(760, 1000, 40)
high_freq_mask_low_rez2 = np.arange(640, 920, 40)
new_mask = array([299, 350, 499, 599, 799, 950])

plot_range = slice(25, 200)
# plot_range = slice(25, 1000)
plot_range1 = slice(1, 500)
# plot_range1 = slice(1, 1000)
plot_range_OP = slice(25, 350)
# plot_range_sub = slice(25, 1000)
# eval_point = (10, 10)#(20, 9)

shgo_iters = 6

f_offset = 0.0296  # 0.0296 with screenshot works

# selected_freqs = array([0.420, 0.520, 0.650, 0.800, 0.850, 0.950], dtype=float)
# selected_freqs = array([0.100, 0.250, 0.650, 0.800, 0.910, 1.050], dtype=float)
# selected_freqs = round(array([0.130, 0.280, 0.680, 0.830, 0.940, 1.080], dtype=float) - f_offset, 3) # original set
selected_freqs = round(array([0.130, 0.280, 0.680, 0.830, 0.940, 1.080], dtype=float) - f_offset, 3)

