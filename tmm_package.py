from helpers import is_iterable
from scipy.constants import c as c0
from tmm import coh_tmm, list_snell, EPSILON, is_forward_angle, interface_r, interface_t, make_2x2_array
import numpy as np
from numpy import inf, pi, exp, zeros, cos, seterr


def check_ri(n_lst, bound_n):
    if bound_n is None:
        bound_n = [1, 1]

    if not is_iterable(n_lst):
        return np.array([1, n_lst, 1])

    n_lst = np.array(n_lst)
    if not np.isclose(n_lst[0], bound_n[0]):
        n_lst = np.array([bound_n[0], *n_lst])
    if not np.isclose(n_lst[-1], bound_n[-1]):
        n_lst = np.array([*n_lst, bound_n[-1]])

    return n_lst


def tmm_package_wrapper(freqs, d_list, n, geom="r", bound_n=None, angle=8):
    # freq should be in THz ("between 0 and 10 THz"), d in um (wl in um)
    # n[freq_idx, n_idx]
    if d_list[0] != inf:
        d_list = [inf, *d_list]
    if d_list[-1] != inf:
        d_list = [*d_list, inf]

    angle_in = angle * pi / 180

    if (n.ndim == 1) and not is_iterable(freqs):
        lambda_vac = (c0 / freqs) * 10 ** -6
        n_list = n
        n_list = check_ri(n_list, bound_n)
        r = coh_tmm("s", n_list, d_list, angle_in, lambda_vac)[geom] * -(geom == "r")
        ret = np.array([freqs, r])
    else:
        lambda_vacs = (c0 / freqs) * 10 ** -6
        r_list = []
        for i, lambda_vac in enumerate(lambda_vacs):
            n_list = n[i]
            n_list = check_ri(n_list, bound_n)
            r_list.append(coh_tmm("s", n_list, d_list, angle_in, lambda_vac)[geom])
        r_arr = np.array(r_list) * -(geom == "r")
        ret = np.array([freqs, r_arr]).T

    ret = np.nan_to_num(ret)

    return ret


def coh_tmm_slim(pol, n_list, d_list, th_0, lam_vac, en_debug=False):
    """
    see coh_tmm of tmm package. This is a slimmed down version. (only r)
    """
    # Convert lists to numpy np.arrays if they're not already.
    n_list = np.array(n_list)
    d_list = np.array(d_list, dtype=float)

    # Input tests
    if ((hasattr(lam_vac, 'size') and lam_vac.size > 1)
            or (hasattr(th_0, 'size') and th_0.size > 1)):
        raise ValueError('This function is not vectorized; you need to run one '
                         'calculation at a time (1 wavelength, 1 angle, etc.)')
    if (n_list.ndim != 1) or (d_list.ndim != 1) or (n_list.size != d_list.size):
        raise ValueError("Problem with n_list or d_list!")
    assert d_list[0] == d_list[-1] == inf, 'd_list must start and end with inf!'
    assert abs((n_list[0] * np.sin(th_0)).imag) < 100 * EPSILON, 'Error in n0 or th0!'
    assert is_forward_angle(n_list[0], th_0), 'Error in n0 or th0!'
    num_layers = n_list.size

    # th_list is a list with, for each layer, the angle that the light travels
    # through the layer. Computed with Snell's law. Note that the "angles" may be
    # complex!
    th_list = list_snell(n_list, th_0)

    # kz is the z-component of (complex) angular wavevector for forward-moving
    # wave. Positive imaginary part means decaying.
    kz_list = 2 * np.pi * n_list * cos(th_list) / lam_vac

    # delta is the total phase accrued by traveling through a given layer.
    # Ignore warning about inf multiplication
    olderr = seterr(invalid='ignore')
    delta = kz_list * d_list
    seterr(**olderr)

    # For a very opaque layer, reset delta to avoid divide-by-0 and similar
    # errors. The criterion imag(delta) > 35 corresponds to single-pass
    # transmission < 1e-30 --- small enough that the exact value doesn't
    # matter.
    for i in range(1, num_layers - 1):
        if delta[i].imag > 35:
            delta[i] = delta[i].real + 35j
            if 'opacity_warning' not in globals():
                global opacity_warning
                opacity_warning = True
                print("Warning: Layers that are almost perfectly opaque "
                      "are modified to be slightly transmissive, "
                      "allowing 1 photon in 10^30 to pass through. It's "
                      "for numerical stability. This warning will not "
                      "be shown again.")

    # t_list[i,j] and r_list[i,j] are transmission and reflection amplitudes,
    # respectively, coming from i, going to j. Only need to calculate this when
    # j=i+1. (2D np.array is overkill but helps avoid confusion.)
    t_list = zeros((num_layers, num_layers), dtype=complex)
    r_list = zeros((num_layers, num_layers), dtype=complex)
    for i in range(num_layers - 1):
        t_list[i, i + 1] = interface_t(pol, n_list[i], n_list[i + 1],
                                       th_list[i], th_list[i + 1])
        r_list[i, i + 1] = interface_r(pol, n_list[i], n_list[i + 1],
                                       th_list[i], th_list[i + 1])
    # At the interface between the (n-1)st and nth material, let v_n be the
    # amplitude of the wave on the nth side heading forwards (away from the
    # boundary), and let w_n be the amplitude on the nth side heading backwards
    # (towards the boundary). Then (v_n,w_n) = M_n (v_{n+1},w_{n+1}). M_n is
    # M_list[n]. M_0 and M_{num_layers-1} are not defined.
    # My M is a bit different than Sernelius's, but Mtilde is the same.
    M_list = zeros((num_layers, 2, 2), dtype=complex)
    for i in range(1, num_layers - 1):
        M_list[i] = (1 / t_list[i, i + 1]) * np.dot(
            make_2x2_array(exp(-1j * delta[i]), 0, 0, exp(1j * delta[i]),
                           dtype=complex),
            make_2x2_array(1, r_list[i, i + 1], r_list[i, i + 1], 1, dtype=complex))
    Mtilde = make_2x2_array(1, 0, 0, 1, dtype=complex)
    for i in range(1, num_layers - 1):
        Mtilde = np.dot(Mtilde, M_list[i])
    Mtilde = np.dot(make_2x2_array(1, r_list[0, 1], r_list[0, 1], 1,
                                   dtype=complex) / t_list[0, 1], Mtilde)

    # Net complex transmission and reflection amplitudes
    # r = Mtilde[1, 0] / Mtilde[0, 0]

    r = Mtilde[0, 1] / Mtilde[1, 1]
    if en_debug:
        print("delta", delta)
        print("num:", Mtilde[0, 1])
        print("den:", Mtilde[1, 1])

    return r


def coh_tmm_slim_no_checks(pol, n_list, d_list, th_0, lam_vac):
    """
    see coh_tmm of tmm package. This is a slimmed down version without input checks.
    removed transmission coefficient calculation
    """
    # Convert lists to numpy np.arrays if they're not already.
    n_list = np.array(n_list)
    d_list = np.array(d_list, dtype=float)
    num_layers = n_list.size

    # th_list is a list with, for each layer, the angle that the light travels
    # through the layer. Computed with Snell's law. Note that the "angles" may be
    # complex!
    th_list = list_snell(n_list, th_0)

    # kz is the z-component of (complex) angular wavevector for forward-moving
    # wave. Positive imaginary part means decaying.
    kz_list = 2 * np.pi * n_list * cos(th_list) / lam_vac

    # delta is the total phase accrued by traveling through a given layer.
    # Ignore warning about inf multiplication
    olderr = seterr(invalid='ignore')
    delta = kz_list * d_list
    seterr(**olderr)

    r_list = zeros((num_layers, num_layers), dtype=complex)
    for i in range(num_layers - 1):
        r_list[i, i + 1] = interface_r(pol, n_list[i], n_list[i + 1],
                                       th_list[i], th_list[i + 1])
    # At the interface between the (n-1)st and nth material, let v_n be the
    # amplitude of the wave on the nth side heading forwards (away from the
    # boundary), and let w_n be the amplitude on the nth side heading backwards
    # (towards the boundary). Then (v_n,w_n) = M_n (v_{n+1},w_{n+1}). M_n is
    # M_list[n]. M_0 and M_{num_layers-1} are not defined.
    # My M is a bit different than Snelius's, but Mtilde is the same.
    M_list = zeros((num_layers, 2, 2), dtype=complex)
    for i in range(1, num_layers - 1):
        M_list[i] = np.dot(
            make_2x2_array(exp(-1j * delta[i]), 0, 0, exp(1j * delta[i]),
                           dtype=complex),
            make_2x2_array(1, r_list[i, i + 1], r_list[i, i + 1], 1, dtype=complex))
    Mtilde = make_2x2_array(1, 0, 0, 1, dtype=complex)
    for i in range(1, num_layers - 1):
        Mtilde = np.dot(Mtilde, M_list[i])
    Mtilde = np.dot(make_2x2_array(1, r_list[0, 1], r_list[0, 1], 1, dtype=complex), Mtilde)

    # Net complex transmission and reflection amplitudes
    r = Mtilde[0, 1] / Mtilde[1, 1]

    return r
