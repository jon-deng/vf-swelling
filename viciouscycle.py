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

def voice_outputs(f: sf.StateFile):
    t = proc_time(f)
    q = proc_glottal_flow_rate(f)

    dt = t[1] - t[0]
    fund_freq, fund_phase, dfreq, dphase, info = \
        modal.fundamental_mode_from_peaks(q, dt, height=np.max(q)*0.8)

    prms = proc_prms(t, q)/10

    return prms, fund_freq


def compensate(
        model: Model,
        targets: Any,
        prop0: bv.BlockVector, control0: bv.BlockVector,
        times: NDArray,
        f0: sf.StateFile
    ) -> bv.BlockVector:
    """
    Return model parameters needed to achieve target voice outputs

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    targets:
        The target voice outputs
    prop0, control0: bv.BlockVector
        A set of model properties that are constants or initial guesses

        Model parameters that are variable for achieving target outputs will
        be changed starting from the initial values given in these arguments.
    times:
        The integration times
    f0:
        The statefile corresponding to a model run with the supplied initial guess
    """
    # To find the appropriate compensatory adjustments, use an iterative Newton
    # method
    # Calculate voice output sensitivity to phonation parameters
    # with a FD approximation
    voice_output0 = voice_outputs(f0)

    # Populate the voice output sensitivity matrix column-by-column
    # Use a subglottal pressure change of 5 Pa
    # Use a maximum stiffness change of 500 Pa
    input_labels = ('psub', 'emod')
    dinputs = (5 * 10, 500 * 10)
    voice_output_jac = np.zeros((len(voice_output0), len(input_labels)))
    for j, input_label in enumerate(input_labels):
        dcontrol = control0.copy()
        dcontrol[:] = 0
        dprop = prop0.copy()
        dprop[:] = 0


        if input_label == 'psub':
            dcontrol['psub'] = dinputs[0]
            dprop[:] = 0
        elif input_label == 'emod':
            dcontrol[:] = 0
            dprop[:] = 0
            scaled_emod = prop0.sub['emod']/np.max(prop0.sub['emod'])
            dprop['emod'] = dinputs[1] * scaled_emod

        # Run a simulation with the incremented control/prop to find
        # the change in voice outputs
        control1 = control0 + dcontrol
        prop1 = prop0 + dprop
        ini_state = model.state0
        ini_state[:] = 0
        with sf.StateFile(model, 'tmp.h5', mode='w') as f:
            _, solver_info = forward.integrate(
                model, f, ini_state, [control1], prop1, times
            )
            voice_output1 = voice_outputs(f)

        voice_output_jac[:, j] = (voice_output1-voice_output0)/dinputs

    return voice_output_jac

def damage_rate(
        model: Model,
        prop: bv.BlockVector, control: bv.BlockVector,
        times: NDArray,
        fname: str
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

    # Run the model to figure out the damage rate
    ini_state = model.state0
    ini_state[:] = 0
    with sf.StateFile(model, fname, mode='w', driver='') as f:
        _, solver_info = forward.integrate(
            model, f, ini_state, [control], prop, times
        )
        damage = measure(f)

    return f

def swelling_rate(
        model: Model,
        damage_rate: NDArray,
        prop: bv.BlockVector, control: bv.BlockVector
    ) -> NDArray:
    """
    Return the swelling rate

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    damage_rate:
        The damage accumulation rate
    prop, control: bv.BlockVector
        The set of model properties (phonation conditions)
    """
    K = 1

    return K * damage_rate