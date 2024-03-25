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

from tqdm import tqdm

import numpy as np
import dolfin as dfn
import h5py

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

    # dt = t[1] - t[0]
    # fund_freq, fund_phase, dfreq, dphase, info = \
    #     modal.fundamental_mode_from_peaks(q, dt, height=np.max(q)*0.8)

    prms = proc_prms(t, q)/10

    # voice_output = np.array([prms, fund_freq])
    voice_output = np.array([prms])
    return voice_output


def make_model_param_from_comp_input(
        model: Model, x0: NDArray,
        const_control: bv.BlockVector, const_prop: bv.BlockVector,
        ini_state=None
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
        control[f'fluid{n}.psub'] = const_control.sub[f'fluid{n}.psub'][0] + x0[0]

    dprop = const_prop.copy()
    dprop[:] = 0
    if len(x0) > 1:
        # `x0[1]` corresponds to changes in elastic modulus
        # It increments stiffness from the constant distribution
        # A value of 1 will add a maximum of 1 cgs pressure unit (0.1 Pa) to the
        # stiffness
        demod = np.array(const_prop.sub['emod'][:])/np.max(const_prop.sub['emod'][:])
        dprop['emod'][:] = x0[1]*demod
    prop = const_prop + dprop

    # The initial state is a swollen static state
    if ini_state is None:
        ini_state = model.state0.copy()
        ini_state[:] = 0
        model.set_control(control)
        model.set_prop(prop)
        sl_control = model.solid.control.copy()
        sl_prop = model.solid.prop.copy()
        sl_control['p'][:] = 0
        # Use 1 loading step for each 5% swelling
        vmax = np.max(sl_prop['v_swelling'])
        nload = int(round(1/.05 * vmax))
        static_state, static_solve_info = main.solve_static_swollen_config(
            model.solid, sl_control, sl_prop, nload=nload
        )
        ini_state[['u', 'v', 'a']] = static_state
    else:
        static_solve_info = {}

    solve_status = static_solve_info.get('status')
    if solve_status != 0 and solve_status is not None:
        raise RuntimeError("Uh oh I couldn't find a static swollen state")

    return ini_state, control, prop, static_solve_info

def voice_output_jac(
        model: Model, x0: NDArray, f0: sf.StateFile,
        const_control, const_prop, times: NDArray,
        ini_state=None
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
    # Compute the swollen initial state once and re-use it
    ini_state, *_ = make_model_param_from_comp_input(
        model, x0, const_control, const_prop, ini_state=ini_state
    )
    # Calculate voice output sensitivity to phonation parameters
    # with a FD approximation
    # times = f0.get_times()
    voice_output0 = proc_voice_output(f0)

    # Populate the voice output sensitivity matrix column-by-column
    # Use a subglottal pressure change of 5 Pa
    # Use a maximum stiffness change of 500 Pa
    dinputs = (5 * 10, 500 * 10)
    dinputs = (5 * 10,)
    voice_output_jac = np.zeros((len(voice_output0), len(x0)))
    for j, dx in enumerate(dinputs):
        # Run a simulation with the incremented comp. inputs to find the change
        # in voice outputs
        dinput = np.zeros(len(x0))
        dinput[j] = dx
        with sf.StateFile(model, 'tmp.h5', mode='w') as f:
            _, solver_info = integrate(
                model, f, x0+dinput, const_control, const_prop, times,
                ini_state=ini_state
            )
            voice_output1 = proc_voice_output(f)

        voice_output_jac[:, j] = (voice_output1-voice_output0)/dinputs[j]

    return voice_output0, voice_output_jac

def compensate(
        model: Model,
        target: Any,
        x_0: NDArray,
        fname: str,
        const_control, const_prop,
        times: NDArray,
        ini_state=None
    ) -> bv.BlockVector:
    """
    Return model parameters needed to achieve target voice outputs

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    target:
        The target voice outputs
    fname:
        The name of the state file to write
    """
    # Compute the swollen initial state once so you can reuse it for each
    # simulation
    # ; you only have to compute it once because the swelling magnitude remains
    # constant
    ini_state, *_ = make_model_param_from_comp_input(
        model, x_0, const_control, const_prop, ini_state=ini_state
    )

    # Use an iterative Newton method to find the appropriate compensatory
    # adjustments
    def lin_subproblem(x):
        # Calculate voice output sensitivity to phonation parameters
        # with a FD approximation
        with sf.StateFile(model, fname, mode='w') as f:
            integrate(
                model, f, x, const_control, const_prop, times,
                ini_state=ini_state
            )
            y, jac = voice_output_jac(
                model, x, f, const_control, const_prop, times,
                ini_state=ini_state
            )
        y1 = target

        def assem_res():
            return y1 - y

        def assem_jac(res):
            return np.linalg.solve(jac, -res)

        return assem_res, assem_jac

    x, info = newton_solve(x_0, lin_subproblem, lambda x: np.linalg.norm(x))

    return x, info

def proc_damage_rate(
        model: Model,
        f: sf.StateFile
    ) -> NDArray:
    """
    Return the damage accumulation rate

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    f: sf.StateFile
        The model time history
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

def proc_swelling_rate(
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

def integrate(
        model: Model, f: sf.StateFile,
        compensatory_input: NDArray,
        const_control: bv.BlockVector, const_prop: bv.BlockVector,
        times: NDArray,
        ini_state: bv.BlockVector=None
    ):
    ini_state, control_0, prop_0, _ = make_model_param_from_comp_input(
        model, compensatory_input,
        const_control, const_prop,
        ini_state=ini_state
    )
    return forward.integrate(
        model, f, ini_state, [control_0], prop_0, times, use_tqdm=True
    )

def postprocess_xdmf(
        model, fstate, xdmf_path: str,
        overwrite: bool=False
    ):
    """
    Write an XDMF file
    """
    from femvf.vis import xdmfutils
    xdfm_data_dir, xdmf_basename = path.split(xdmf_path)
    xdmf_data_basename = f'{path.splitext(xdmf_basename)[0]}.h5'
    xdmf_data_path = path.join(xdfm_data_dir, xdmf_data_basename)

    # Export mesh values
    _labels = ['mesh/solid', 'time']
    labels = _labels
    datasets = [
        fstate[label] for label in _labels
    ]
    formats = [None, None]

    mesh = model.solid.residual.mesh()
    function_space = dfn.VectorFunctionSpace(mesh, 'CG', 1)
    _labels = ['state/u', 'state/v', 'state/a']
    labels += _labels
    datasets += [fstate[label] for label in _labels]
    formats += len(_labels)*[function_space]

    function_space = dfn.FunctionSpace(mesh, 'DG', 0)
    _labels = ['properties/v_swelling']
    labels += _labels
    datasets += [fstate[f'{label}'] for label in _labels]
    formats += [function_space]

    with h5py.File(xdmf_data_path, mode='w') as fxdmf:
        xdmfutils.export_mesh_values(
            datasets, formats, fxdmf, output_names=labels
        )

        # Annotate the mesh values with an XDMF file
        static_dataset_descrs = [
            (fxdmf['state/u'], 'vector', 'node'),
            (fxdmf['properties/v_swelling'], 'scalar', 'Cell')
        ]
        static_idxs = [
            (0, ...), (slice(None),),
            (slice(None),)
        ]

        temporal_dataset_descrs = [
            (fxdmf['state/u'], 'vector', 'node'),
            (fxdmf['state/v'], 'vector', 'node'),
            (fxdmf['state/a'], 'vector', 'node')
        ]
        temporal_idxs = len(temporal_dataset_descrs)*[
            (slice(None),)
        ]

        xdmfutils.write_xdmf(
            fxdmf['mesh/solid'],
            static_dataset_descrs, static_idxs,
            fxdmf['time'],
            temporal_dataset_descrs, temporal_idxs,
            xdmf_path
        )


def integrate_vicious_cycle(
        model: Model,
        v_0: NDArray, x_0: NDArray,
        const_control: bv.BlockVector, const_prop: bv.BlockVector,
        times: NDArray,
        n_step: int=1,
        v_step: float=0.05
    ):
    """
    Integrate the vicious cycle
    """
    ## Run the zero swelling simulation to establish the compensatory target
    const_prop['v_swelling'] = v_0
    ini_state, *_, solve_info = make_model_param_from_comp_input(
        model, x_0, const_control, const_prop, ini_state=None
    )

    with sf.StateFile(model, f'SwellingStep{0}.h5', mode='w') as f:
        # Run the simulation for the initial compensatory inputs
        _, solve_info = integrate(
            model, f, x_0, const_control, const_prop, times, ini_state=ini_state
        )
        voice_target = proc_voice_output(f)
        damage_rate = proc_damage_rate(model, f)
        vd_0 = proc_swelling_rate(damage_rate)

    # Loops through steps of the vicious cycle (VC)
    for n in tqdm(range(1, n_step)):
        dv = v_step*vd_0/vd_0.max()
        v_1 = v_0 + dv
        const_prop['v_swelling'] = v_1
        ini_state, *_, solve_info = make_model_param_from_comp_input(
            model, x_0, const_control, const_prop, ini_state=None
        )

        # For the current swelling, determine the compensatory change in
        # inputs required
        x_1, info = compensate(
            model, voice_target, x_0,
            f'SwellingStep{n}.h5',
            const_control, const_prop,
            times
        )

        with sf.StateFile(model, f'SwellingStep{n}.h5', mode='r') as f:
            voice_output = proc_voice_output(f)
            damage_rate = proc_damage_rate(model, f)
            vd_1 = proc_swelling_rate(damage_rate)

        # Update variables for next step
        (x_0, v_0, vd_0) = x_1, v_1, vd_1

if __name__ == '__main__':
    from os import path

    import main
    import cases

    POSTPROCESS_XDMF = True

    param = main.ExpParam({
        'MeshName': cases.MESH_BASE_NAME, 'clscale': 0.75,
        'GA': 3,
        'DZ': 1.5, 'NZ': 15,
        'Ecov': cases.ECOV, 'Ebod': cases.EBOD,
        'vcov': 1, 'mcov': 0.0,
        'psub': 600*10,
        'dt': 5e-5, 'tf': 0.50,
        'ModifyEffect': '',
        'SwellingDistribution': 'uniform',
        'SwellingModel': 'power'
    })
    model = main.setup_model(param)

    N = 6
    fpaths = [f'SwellingStep{n}.h5' for n in range(N)]
    if not all(path.isfile(fpath) for fpath in fpaths):
        ini_state, const_controls, const_prop = main.setup_state_control_props(param, model)

        # This is how long to integrate the 'voicing' simulations for, which
        # are used to determine damage rates, swelling fields, etc.
        times = 5e-5*np.arange(2**4)
        times = 5e-5*np.arange(2**12)

        # `v0` and `x0` are the initial swelling field and compensatory inputs
        v_0 = np.ones(const_prop['v_swelling'].shape)
        # x_0 = np.array([0, 0])
        x_0 = np.array([0])
        integrate_vicious_cycle(
            model, v_0, x_0, const_controls[0], const_prop, times,
            n_step=N, v_step=0.1
        )

    if POSTPROCESS_XDMF:
        for fpath in fpaths:
            with h5py.File(fpath, mode='r') as fstate:
                h5_head, h5_suffix = path.splitext(fpath)
                xdmf_path = f'{h5_head}--export.xdmf'
                postprocess_xdmf(
                    model, fstate, xdmf_path
                )
