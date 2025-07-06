from numpy import pi
from pathlib import Path
import os
from scipy.constants import c as c0

c_thz = c0 * 10**-6  # um / ps
thea = 8 * pi / 180

cur_os = os.name

Omega_, Delta_, sigma_, mu_, epsilon_, degree_ = '\u03BC', '\u0394', '\u03C3', '\u03BC', '\u03B5', '\u00B0'

if 'posix' in cur_os:
    data_root = r"/home/ftpuser/ftp/Data/TeraLayer2"
    c_proj_path = r"/home/alex/PycharmProjects/TeraLayer2/C_implementation/nelder-mead"
else:
    data_root = r"C:\Users\alexj\Data\TeraLayer2"
    c_proj_path = r"C:\Users\alexj\Projects\TL2_engine"

msys2_bash_path = Path(r"C:\msys64\usr\bin\bash.exe")

data_root = Path(data_root)
c_proj_path = Path(c_proj_path)
