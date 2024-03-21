r"""
This module contains functionality for modelling the vicious cycle

The vicious cycle is modelled by the ODE
.. math:: \dot{v} = \dot{v}(\dot{\alpha})
where $\alpha$ is the damage distribution.

Assuming swelling is proportional to damage:
.. math:: \dot{v} = K \dot{\alpha}

Since the rate of damage accumulation depends on the current voicing conditions,
.. math:: \dot{v} = \dot{\alpha}(p_\mathrm{sub}, E, v, \ellipsis)
where $p_\mathrm{sub}$ is the subglottal pressure, $E$ is the vocal fold
stiffness, and $\ellipsis$ represents any number of additional voicing
conditions (stiffness, poisson's ratio, etc.).

Because swelling changes voice outputs, it is likely that compensatory changes
in voicing conditions ($p_\mathrm{sub}$, $E$ for muscle stiffness) will occur
to maintain some 'normal' voice output.
This results in a dependence of some voicing parameters on $v$
.. math:: \dot{v} = \dot{\alpha}(p_\mathrm{sub}(v), E(v), v, \ellipsis)
This is the non-linear ODE we assume governs the rate at which swelling occurs.
"""

from typing import Any
from numpy.typing import NDArray

import itertools as itls

import numpy as np
import dolfin as dfn

from femvf import forward, statefile as sf
from femvf.postprocess.base import TimeSeriesStats
from femvf.postprocess import solid as slsig
from femvf.models.transient import coupled
from blockarray import blockvec as bv

from scipy import signal
from vfsig import clinical, fftutils, modal

from nonlineq import newton_solve

Model = coupled.BaseTransientFSIModel

def proc_time(f):
        return f.get_times()

def proc_glottal_flow_rate(f):
    qs = [
        np.sum([f.get_state(ii)[f'fluid{n}.q'][0] for n in range(len(f.model.fluids))])
        for ii in range(f.size)
    ]
    return np.array(qs)

def proc_prms(t, q):
    """
    Return the RMS radiated pressure at 1m using a piston-in-baffle approximation
    """
    # t = SIGNALS[f'{case_name}/time']
    # q = SIGNALS[f'{case_name}/q']
    dt = t[1]-t[0]

    # Assume a 2cm vocal fold length
    # This is needed to get a true flow rate since the 1D model flow rate
    # is over one dimension
    piston_params = {
        'r': 100.0,
        'theta': 0.0,
        'a': 1.0,
        'rho': 0.001225,
        'c': 343*100
    }
    VF_LENGTH = 2
    win = signal.windows.tukey(q.size, alpha=0.15)
    wq = win*q*VF_LENGTH
    fwq = np.fft.rfft(wq)
    freq = np.fft.rfftfreq(wq.size, d=dt)

    fwp = clinical.prad_piston(fwq, f=freq*2*np.pi, piston_params=piston_params)

    # This computes the squared sound pressure
    psqr = fftutils.power_from_rfft(fwp, fwp, n=fwq.size)

    # The rms pressure in units of Pa
    prms = np.sqrt(psqr/fwp.size)

    return prms

def proc_voice_output(f: sf.StateFile):
    t = proc_time(f)
    q = proc_glottal_flow_rate(f)

    dt = t[1] - t[0]
    fund_freq, fund_phase, dfreq, dphase, info = \
        modal.fundamental_mode_from_peaks(q, dt, height=np.max(q)*0.8)

    prms = proc_prms(t, q)/10

    return prms, fund_freq

def proc_compensatory_input(
        model: Model, x0: NDArray,
        const_control: bv.BlockVector, const_prop: bv.BlockVector
    ):
    """
    Return model parameters from compensatory inputs

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    x0: sf.StateFile
        The statefile corresponding to the linearization point
    """
    # `x0[0]` corresponds to an increment in subglottal pressure
    control = const_control.copy()
    nfluid = len(model.fluids)
    for n in range(nfluid):
        control[f'fluid{n}.psub'] = const_control[f'fluid{n}.psub'] + x0[0]

    # `x0[1]` corresponds to changes in elastic modulus
    # It increments stiffness from the constant distribution
    # A value of 1 will add a maximum of 1 cgs pressure unit (0.1 Pa) to the
    # stiffness
    dprop = const_prop.copy()
    dprop[:] = 0
    demod = const_prop.sub['emod']/np.max(const_prop.sub['emod'])
    dprop['emod'][:] = x0[1]*demod
    prop = const_prop + dprop

    # The initial state is just a static state
    ini_state = model.state0
    ini_state[:] = 0

    return ini_state, control, prop

def voice_output_jac(
        model: Model, x0: NDArray, f0: sf.StateFile,
        const_control, const_prop
    ):
    """
    Return model parameters needed to achieve target voice outputs

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    f0: sf.StateFile
        The statefile corresponding to the linearization point
    """
    # Calculate voice output sensitivity to phonation parameters
    # with a FD approximation
    times = f0.get_times()
    voice_output0 = proc_voice_output(f0)

    # Populate the voice output sensitivity matrix column-by-column
    # Use a subglottal pressure change of 5 Pa
    # Use a maximum stiffness change of 500 Pa
    dinputs = (5 * 10, 500 * 10)
    voice_output_jac = np.zeros((len(voice_output0), len(x0)))
    for j, dx in enumerate(dinputs):
        # Run a simulation with the incremented comp. inputs to find the change
        # in voice outputs
        dinput = np.zeros(len(x0))
        dinput[j] = dx
        ini_state, control1, prop1 = proc_compensatory_input(
            x0+dinput, const_control, const_prop
        )
        with sf.StateFile(model, 'tmp.h5', mode='w') as f:
            _, solver_info = forward.integrate(
                model, f, ini_state, [control1], prop1, times
            )
            voice_output1 = proc_voice_output(f)

        voice_output_jac[:, j] = (voice_output1-voice_output0)/dinputs[j]

    return voice_output0, voice_output_jac

def compensate(
        model: Model,
        targets: Any,
        x0: NDArray,
        f0: sf.StateFile,
        const_control, const_prop
    ) -> bv.BlockVector:
    """
    Return model parameters needed to achieve target voice outputs

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    targets:
        The target voice outputs
    f0:
        The statefile corresponding to a model run with the supplied initial guess
    """
    # To find the appropriate compensatory adjustments, use an iterative Newton
    # method

    def lin_subproblem(x):
        # Calculate voice output sensitivity to phonation parameters
        # with a FD approximation
        y0, jac = voice_output_jac(model, x0, f0, const_control, const_prop)
        y1 = targets

        def assem_res(x):
            return y1 - y0

        def assem_jac(x):
            return -jac

    x, info = newton_solve(x0, lin_subproblem, lambda x: np.linalg.norm(x))

    return x

def damage_rate(
        model: Model,
        x: NDArray,
        f: sf.StateFile,
        const_control, const_prop
    ) -> NDArray:
    """
    Return the damage accumulation rate

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    targets:
        The target voice outputs
    prop, control: bv.BlockVector
        The set of model properties (phonation conditions)
    times:
        The integration times
    """
    # TODO: You could/should make `measure` a parameter that's passed in
    dx = model.solid.residual.measure('dx')
    mesh = model.solid.residual.mesh()
    fspace = dfn.FunctionSpace(mesh, 'DG', 0)
    state_measure = slsig.ViscousDissipationField(
        model, dx=dx, fspace=fspace
    )
    def measure(f):
        mean = TimeSeriesStats(state_measure).mean(f, range(f.size//2, f.size))
        return mean

    damage = measure(f)
    return damage

def swelling_rate(
        damage_rate: NDArray
    ) -> NDArray:
    """
    Return the swelling rate

    Parameters
    ----------
    damage_rate:
        The damage accumulation rate
    """
    K = 1.0
    return K * damage_rate
