r"""
This module contains functionality for modelling the vicious cycle

The vicious cycle is modelled by the ODE
.. math:: \dot{v} = \dot{v}(\dot{\alpha}) ,
where $\alpha$ is the damage distribution.

Assuming swelling is proportional to damage, results in:
.. math:: \dot{v} = K \dot{\alpha},
where $K$ is a constant.

Since the rate of damage accumulation depends on the current voicing conditions,
.. math:: \dot{v} \propto \dot{\alpha}(p_\mathrm{sub}, E, v, \ellipsis)
where $p_\mathrm{sub}$ is the subglottal pressure, $E$ is the vocal fold
stiffness, and $\ellipsis$ represents any number of additional voicing
conditions (stiffness, poisson's ratio, etc.).

Because swelling changes voice outputs, it is likely that compensatory changes
in voicing conditions ($p_\mathrm{sub}$, $E$ for muscle stiffness) will occur
to maintain some 'normal' voice output.
This results in a dependence of some voicing parameters on $v$ and a
target voice condition $f_\mathrm{target}$.
.. math:: \dot{v} \propto \dot{\alpha}(p_\mathrm{sub}(v, f_\mathrm{target}), E(v, f_\mathrm{target}), v, \ellipsis).
"""

from typing import Any, Mapping, Tuple, Optional, Union, List
from numpy.typing import NDArray

from tqdm import tqdm
from argparse import ArgumentParser
import warnings
import functools

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

from exputils import postprocutils

import main
from main import proc_glottal_flow_rate

Model = coupled.BaseTransientFSIModel
SolverInfo = Mapping[str, Any]


def proc_time(f: sf.StateFile) -> NDArray:
    """
    Return the simulation time vector
    """
    return f.get_times()


def proc_voice_output(f: sf.StateFile, n: int) -> Tuple[NDArray, Mapping[str, NDArray]]:
    """
    Return voice outputs (RMS pressure, fundamental frequency)
    """
    t = proc_time(f)
    q = proc_glottal_flow_rate(f)

    # Truncate part of the flow rate signal to a steady state portion
    idx_trunc = slice(t.size // 2, t.size)
    t_trunc = t[idx_trunc]
    q_trunc = q[idx_trunc]

    dt = t[1] - t[0]
    fund_freq, fund_phase, dfreq, dphase, info = modal.fundamental_mode_from_peaks(
        q, dt, height=np.max(q) * 0.8
    )
    if len(info['peaks']) < 2:
        fund_freq = 0.0

    prms = calc_prms(t_trunc, q_trunc) / 10

    _voice_output = (prms, fund_freq)
    # voice_output = np.array([prms, fund_freq])
    voice_output = np.array(_voice_output[:n])
    return voice_output, {'t': t, 'q': q}


def calc_prms(t: NDArray, q: NDArray) -> float:
    """
    Return the RMS radiated pressure at 1 m using a piston-in-baffle approximation
    """
    # t = SIGNALS[f'{case_name}/time']
    # q = SIGNALS[f'{case_name}/q']
    dt = t[1] - t[0]

    # Assume a 2cm vocal fold length
    # This is needed to get a true flow rate since the 1D model flow rate
    # is over one dimension
    piston_params = {
        'r': 100.0,
        'theta': 0.0,
        'a': 1.0,
        'rho': 0.001225,
        'c': 343 * 100,
    }
    VF_LENGTH = 2
    win = signal.windows.tukey(q.size, alpha=0.15)
    wq = win * q * VF_LENGTH
    fwq = np.fft.rfft(wq)
    freq = np.fft.rfftfreq(wq.size, d=dt)

    fwp = clinical.prad_piston(fwq, f=freq * 2 * np.pi, piston_params=piston_params)

    # This computes the squared sound pressure
    psqr = fftutils.power_from_rfft(fwp, fwp, n=fwq.size)

    # The rms pressure in units of Pa
    prms = np.sqrt(psqr / fwp.size)

    return prms


def proc_damage_rate(
    model: Model,
    f: sf.StateFile,
    damage_measure: str = 'field.tavg_viscous_dissipation',
) -> NDArray:
    """
    Return the damage rate

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
    if damage_measure == 'field.tavg_viscous_dissipation':
        state_measure = slsig.ViscousDissipationField(model, dx=dx, fspace=fspace)

        def measure(f):
            time_mean = TimeSeriesStats(state_measure).mean(
                f, range(f.size // 2, f.size)
            )
            return time_mean

    elif damage_measure == 'field.tmax_strain_energy':
        state_measure = slsig.StrainEnergy(model, dx=dx, fspace=fspace)

        def measure(f):
            time_max = TimeSeriesStats(state_measure).max(f, range(f.size // 2, f.size))
            return time_max

    else:
        raise ValueError(f"Unknown damage measure '{damage_measure}'")

    damage = measure(f)
    return damage


def proc_swelling_rate(
    model: Model,
    fpath: str,
    damage_measure: str = 'viscous_dissipation',
    swelling_dmg_growth_rate: float = 1.0,
) -> NDArray:
    """
    Return the swelling rate for the given conditions

    Note that `v`, `comp_input`, and `const_control` and `const_prop` define
    all inputs to the swelling/damage rate functions.

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    fpath: str
        The path to write the voicing simulation to

        If the path already exists, it's assumed that this is the result of
        running the voicing simulation
    v: NDArray
        The initial swelling field
    comp_input: NDArray
        The compensatory input
    const_ini_state, const_control, const_prop: bv.BlockVector
        Any constant values for the model inputs
    voicing_time: NDArray
        A voicing time vector
    """
    with sf.StateFile(model, fpath, mode='r') as f:
        dmg_rate = proc_damage_rate(model, f, damage_measure=damage_measure)

    swelling_rate = swelling_dmg_growth_rate * dmg_rate
    return swelling_rate, dmg_rate


def map_vc_input_to_model_input(
    model: Model,
    v: NDArray,
    comp_input: NDArray,
    const_model_args: Tuple[bv.BlockVector, bv.BlockVector, bv.BlockVector]
) -> Tuple[bv.BlockVector, bv.BlockVector, bv.BlockVector, SolverInfo]:
    """
    Return model inputs from vicious cycle inputs

    Notably, this will ensure the swelling field is set properly and compute a
    static, swollen initial state.

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    v: NDArray
        The swelling field
    comp_input: NDArray
        The compensatory input vector

        This is assumed to be in the order:
        (subglottal pressure, modulus increase).
    const_ini_state, const_control, const_prop: bv.BlockVector
        Any constant values of the model inputs
    """
    const_ini_state, const_control, const_prop = const_model_args
    # `comp_input[0]` corresponds to an increment in subglottal pressure
    control = const_control.copy()
    for n in range(len(model.fluids)):
        control[f'fluid{n}.psub'] = (
            const_control.sub[f'fluid{n}.psub'][0] + comp_input[0]
        )

    # `comp_input[1]` corresponds to changes in elastic modulus
    # It increments stiffness from the constant distribution
    # A value of 1 will add a maximum of 1 stiffness unit to the stiffness
    dprop = const_prop.copy()
    dprop[:] = 0
    if len(comp_input) > 1:
        demod = const_prop['emod'] / np.max(const_prop['emod'])
        dprop['emod'][:] = comp_input[1] * demod
    prop = const_prop + dprop

    prop['v_swelling'][:] = v

    # Compute `ini_state` as a swollen static state without external loading
    ini_state = const_ini_state.copy()
    # ini_state[:] = 0

    model.set_control(control)
    model.set_prop(prop)
    sl_control = model.solid.control.copy()
    sl_prop = model.solid.prop.copy()
    # Ensure there's no external loading
    sl_control['p'][:] = 0

    # Compute the swollen static state
    # Use 1 loading step for each 5% swelling increase
    vmax = np.max(sl_prop['v_swelling'])
    nload = int(round(1 / 0.01 * (vmax-1)))
    static_state, static_solve_info = main.solve_static_swollen_config(
        model.solid,
        sl_control,
        sl_prop,
        nload=nload,
        static_state_0=const_ini_state[['u', 'v', 'a']],
    )

    num_loading_steps = static_solve_info.get('num_loading_steps', None)
    if num_loading_steps is not None:
        final_solve_info = static_solve_info[f'LoadingStep{num_loading_steps}']
    else:
        final_solve_info = static_solve_info

    if final_solve_info['status'] != 0:
        raise RuntimeError(
            "Static state couldn't be solved with solver info: " f"{static_solve_info}"
        )

    ini_state[['u', 'v', 'a']] = static_state

    return ini_state, control, prop, static_solve_info


def make_voice_output_jac(
    model: Model,
    voice_output_0: NDArray,
    v: NDArray,
    comp_input: NDArray,
    const_model_args: Tuple[bv.BlockVector, bv.BlockVector, bv.BlockVector],
    voicing_time: NDArray,
    fpath: str = 'tmp.h5',
) -> NDArray:
    """
    Return model parameters needed to achieve target voice outputs

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    voice_output_0: NDArray
        The voice otuput at the linearization point
    v: NDArray
        The swelling field
    comp_input: NDArray
        The compensatory input vector (linearization point)

        This is assumed to be in the order:
        (subglottal pressure, modulus increase).
    const_ini_state, const_control, const_prop: bv.BlockVector
        Any constant values of the model inputs
    voicing_time: NDArray
        The voicing simulation time vector
    ini_state: Optional[bv.BlockVector]
        An optional initial state to return

        If an initial state is not provided, it is calculated such that the
        model is static under the given swelling field and with no external
        loads.
    fpath: str
        The path to write simulation files to

        These files are needed to compute the jacobian sensitivity.
    """
    # Compute the swollen initial state once and re-use it
    ini_state, *_ = map_vc_input_to_model_input(
        model, v, comp_input, const_model_args
    )
    _model_args = (ini_state,) + const_model_args[1:]

    # Calculate voice output sensitivity to phonation parameters
    # with a FD approximation
    # Populate the voice output sensitivity matrix column-by-column
    # For the FD increments:
    # Use a subglottal pressure change of 10 Pa
    # Use a maximum stiffness change of 10 kPa
    _dinputs = (10 * 10, 10e3 * 10)
    dinputs = list(np.diag(_dinputs))[: len(comp_input)]

    def form_jac_column(dinput):
        with sf.StateFile(model, fpath, mode='w') as f:
            _, solver_info = integrate(
                model,
                f,
                v,
                comp_input + dinput,
                _model_args,
                voicing_time,
            )
            voice_output1 = proc_voice_output(f, len(comp_input))[0]

        return (voice_output1 - voice_output_0) / np.linalg.norm(dinput)

    voice_output_jac = np.array([form_jac_column(dinput) for dinput in dinputs])

    # print(voice_output_jac)

    return voice_output_jac


def solve_comp_input(
    model: Model,
    fpath: str,
    v: NDArray,
    voice_target: NDArray,
    const_model_args: Tuple[bv.BlockVector, bv.BlockVector, bv.BlockVector],
    voicing_time: NDArray,
    comp_input_0: Optional[NDArray] = None,
) -> Tuple[str, bv.BlockVector, SolverInfo]:
    """
    Return compensatory adjustments needed to achieve target voice outputs

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    fpath: str
        The path to a file to write the compensatory simulations

        This path will be written/rewritten iteratively as the function
        tries to find compensatory inputs that give the target voice output.
    v: NDArray
        The swelling field
    voice_target: NDArray
        The target voice otuputs
    const_ini_state, const_control, const_prop: bv.BlockVector
        Any constant values of the model inputs
    voicing_time: NDArray
        A simulation time vector
    comp_input_0: NDArray
        The compensatory input vector (linearization point)

        This is assumed to be in the order:
        (subglottal pressure, modulus increase).
    use_existing_fpath_on_first: bool
        Whether to use an existing voicing simulation as the first iterative guess
    """
    # Compute the swollen initial state once so you can reuse it for each
    # simulation
    # ; you only have to compute it once because the swelling magnitude remains
    # constant
    const_ini_state, *_ = map_vc_input_to_model_input(
        model, v, comp_input_0, const_model_args
    )

    # Use an iterative Newton method to find the appropriate compensatory
    # adjustments
    const_model_args = (const_ini_state,) + const_model_args[1:]
    def lin_subproblem(x):
        # Get model parameters for current guess `x`
        ini_state, control, prop, _ = map_vc_input_to_model_input(
            model, v, x, const_model_args
        )
        _model_args = (ini_state, control, prop)

        # Check whether an existing voicing simulation exists for the current guess
        existing_voicing_sim = False
        if path.isfile(fpath):
            with sf.StateFile(model, fpath, mode='r') as f:

                ext_ini_state = f.get_state(0)
                ext_control = f.get_control(0)
                ext_prop = f.get_prop()

                ini_state_err = (ini_state - ext_ini_state).norm()
                control_err = (control - ext_control).norm()
                prop_err = (prop - ext_prop).norm()

                if all(np.isclose([ini_state_err, control_err, prop_err], 0)):
                    existing_voicing_sim = True

        # Calculate voice output sensitivity to phonation parameters
        # with a FD approximation
        if not existing_voicing_sim:
            with sf.StateFile(model, fpath, mode='w') as f:
                forward.integrate(model, f, ini_state, [control], prop, voicing_time)

        with sf.StateFile(model, fpath, mode='r') as f:
            y = proc_voice_output(f, len(comp_input_0))[0]

        def assem_res():
            y1 = voice_target
            return y1 - y

        def assem_jac(res):
            path_head, path_tail = path.splitext(fpath)
            jac = make_voice_output_jac(
                model,
                y,
                v,
                x,
                const_model_args,
                voicing_time,
                fpath=f'{path_head}--tmp{path_tail}',
            )
            return np.linalg.solve(jac, -res)

        return assem_res, assem_jac

    comp_input, info = newton_solve(
        comp_input_0,
        lin_subproblem,
        lambda x: np.linalg.norm(x),
        params={
            'absolute_tolerance': 1e-4,
            'relative_tolerance': 1e-8,
            'maximum_iterations': 10,
        },
    )

    if info['status'] != 0:
        raise RuntimeError(
            "Couldn't find compensatory input!" "Solver failed with info: " f"{info}"
        )

    return fpath, comp_input, info


def integrate(
    model: Model,
    f: sf.StateFile,
    v: NDArray,
    comp_input: NDArray,
    const_model_args: Tuple[bv.BlockVector, bv.BlockVector, bv.BlockVector],
    voicing_time: NDArray,
) -> Tuple[bv.BlockVector, SolverInfo]:
    """
    Integrate the model over time

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    v: NDArray
        The swelling field
    comp_input: NDArray
        The compensatory input vector

        This is assumed to be in the order:
        (subglottal pressure, modulus increase).
    const_ini_state, const_control, const_prop: bv.BlockVector
        Any constant values of the model inputs
    voicing_time: NDArray
        A simulation time vector
    """
    ini_state, control, prop, _ = map_vc_input_to_model_input(
        model, v, comp_input, const_model_args
    )
    return forward.integrate(
        model, f, ini_state, [control], prop, voicing_time, use_tqdm=True
    )


def postprocess_xdmf(
    model: Model,
    fstate: h5py.File,
    fpost: h5py.File,
    xdmf_path: str,
    overwrite: bool = False,
):
    """
    Post-process results to an XDMF file

    Parameters
    ----------
    model: Model
        The model
    fstate: h5py.File
        The model time history
    xdmf_path: str
        The XDMF file path to write
    overwrite: bool
        Whether to overwrite previously written files
    """
    from femvf.vis import xdmfutils

    xdfm_data_dir, xdmf_basename = path.split(xdmf_path)
    xdmf_data_basename = f'{path.splitext(xdmf_basename)[0]}.h5'
    xdmf_data_path = path.join(xdfm_data_dir, xdmf_data_basename)

    # Export mesh values
    _labels = ['mesh/solid', 'time']
    labels = _labels
    datasets = [fstate[label] for label in _labels]
    formats = [None, None]

    mesh = model.solid.residual.mesh()
    function_space = dfn.VectorFunctionSpace(mesh, 'CG', 1)
    _labels = ['state/u', 'state/v', 'state/a']
    labels += _labels
    datasets += [fstate[label] for label in _labels]
    formats += len(_labels) * [function_space]

    function_space = dfn.FunctionSpace(mesh, 'DG', 0)
    _labels = ['properties/v_swelling']
    labels += _labels
    datasets += [fstate[f'{label}'] for label in _labels]
    formats += [function_space]

    _labels = [
        'field.tmax_strain_energy',
        'field.tavg_strain_energy',
        'field.tavg_viscous_rate',
    ]
    labels += _labels
    datasets += [fpost[f'{label}'] for label in _labels]
    formats += len(_labels) * [function_space]

    with h5py.File(xdmf_data_path, mode='w') as fxdmf:
        xdmfutils.export_mesh_values(datasets, formats, fxdmf, output_names=labels)

        # Annotate the mesh values with an XDMF file
        static_dataset_descrs = [
            (fxdmf['state/u'], 'vector', 'node'),
            (fxdmf['properties/v_swelling'], 'scalar', 'Cell'),
            (fxdmf['field.tavg_strain_energy'], 'scalar', 'Cell'),
            (fxdmf['field.tmax_strain_energy'], 'scalar', 'Cell'),
            (fxdmf['field.tavg_viscous_rate'], 'scalar', 'Cell'),
        ]
        static_idxs = [
            (0, ...),
            (slice(None),),
            (slice(None),),
            (slice(None),),
            (slice(None),),
        ]

        temporal_dataset_descrs = [
            (fxdmf['state/u'], 'vector', 'node'),
            (fxdmf['state/v'], 'vector', 'node'),
            (fxdmf['state/a'], 'vector', 'node'),
        ]
        temporal_idxs = len(temporal_dataset_descrs) * [(slice(None),)]

        xdmfutils.write_xdmf(
            fxdmf['mesh/solid'],
            static_dataset_descrs,
            static_idxs,
            fxdmf['time'],
            temporal_dataset_descrs,
            temporal_idxs,
            xdmf_path,
        )


def integrate_vc(
    model: Model,
    v_0: NDArray,
    voice_target: Union[NDArray, None],
    const_model_args: Tuple[bv.BlockVector, bv.BlockVector, bv.BlockVector],
    voicing_time: NDArray,
    n_start: int = 0,
    n_stop: int = 1,
    dv_max: float = 0.05,
    t_0: float = 0.0,
    dt: float = 1.0,
    comp_input_0: Optional[NDArray] = None,
    swelling_dmg_growth_rate=1.0,
    swelling_healing_rate=1.0,
    damage_measure: str = 'viscous_dissipation',
    output_dir: str = 'out',
    base_fname: str = 'SwellingStep',
):
    """
    Integrate the vicious cycle

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    v_0: NDArray
        The initial swelling field
    voice_target: Union[NDArray, None]
        The target voice output

        If no target voice output is given, it is assumed that the target
        voice output is that for the initial condition.
    const_model_args: bv.BlockVector
        Constant values for the model initial state, control, and property vectors
    voicing_time: NDArray
        A simulation time vector
    n_start: int
        The initial state index of the vicious cycle
    n_stop: int
        The final state index of the vicious cycle
    dv_max: float
        The maximum increment in swelling for a vicious cycle step
    comp_input_0: Optional[NDArray]
        A guess for the initial compensatory input

        If no target voice output is given, this is ignored and set to be zero.
    output_dir: str
        The directory to write results to
    base_fname: str
        The base filename
    """

    if comp_input_0 is None and voice_target is None:
        comp_input_0 = np.zeros(1)
    # elif comp_input_0 is None and voice_target is not None:
        comp_input_0 = np.zeros(voice_target.shape)

    # Establish the voice target if none is provided
    if voice_target is None:
        state_fpath_0 = f'{output_dir}/{base_fname}{n_start}.h5'

        ini_state, control, prop, _ = map_vc_input_to_model_input(
            model, v_0, comp_input_0, const_model_args
        )
        with sf.StateFile(model, state_fpath_0, mode='w') as f:
            _, solve_info = forward.integrate(
                model, f, ini_state, [control], prop, voicing_time, use_tqdm=True
            )
            voice_target = proc_voice_output(f, len(comp_input_0))[0]

    ## Loop through steps of the vicious cycle (VC)
    integrate_vc_steps(
        model,
        v_0,
        voice_target,
        const_model_args,
        voicing_time,
        n_start=n_start,
        n_stop=n_stop,
        dv_max=dv_max,
        t_0=t_0,
        dt=dt,
        swelling_dmg_growth_rate=swelling_dmg_growth_rate,
        swelling_healing_rate=swelling_healing_rate,
        damage_measure=damage_measure,
        comp_input_0=comp_input_0,
        output_dir=output_dir,
        base_fname=base_fname,
    )


def resume_integrate_vc(
    model: Model,
    n_start: int,
    n_stop: int,
    dv_max: float = 0.05,
    damage_measure: str = 'viscous_dissipation',
    dt=1.0,
    swelling_dmg_growth_rate=1.0,
    swelling_healing_rate=1.0,
    output_dir: str = 'out',
    base_fname: str = 'SwellingStep',
):
    """
    Integrate the vicious cycle starting from a previous voicing simulation

    Parameters
    ----------
    model: Model
    n_start: int
        The initial state index of the vicious cycle
    n_stop: int
        The final state index of the vicious cycle
    v_step: float
        The increment in swelling to take for each step of the vicious cycle
    output_dir: str
        The directory to write results to
    base_fname: str
        The base filename
    """
    state_fpath_0 = f'{output_dir}/{base_fname}{n_start}.h5'
    with sf.StateFile(model, state_fpath_0, mode='r') as f:
        voice_target = proc_voice_output(f, 1)[0]
        const_ini_state = f.get_state(0)
        const_control = f.get_control(0)
        const_prop = f.get_prop()
        voicing_time = f.get_times()
        t_0 = f.file['ViciousCycle/time'][()]
    v_0 = np.array(const_prop.sub['v_swelling'])
    const_model_args = (const_ini_state, const_control, const_prop)

    ## Loop through steps of the vicious cycle (VC)
    # comp_input_n = comp_input_0
    # NOTE: Ideally you should be able to figure out what the compensatory input
    # was from the initial state file

    comp_input_0 = np.array([0])
    integrate_vc_steps(
        model,
        v_0,
        voice_target,
        const_model_args,
        voicing_time,
        n_start=n_start,
        n_stop=n_stop,
        dv_max=dv_max,
        damage_measure=damage_measure,
        t_0=t_0,
        dt=dt,
        swelling_dmg_growth_rate=swelling_dmg_growth_rate,
        swelling_healing_rate=swelling_healing_rate,
        comp_input_0=comp_input_0,
        output_dir=output_dir,
        base_fname=base_fname,
    )


def integrate_vc_steps(
    model: Model,
    v_0: NDArray,
    voice_target: Union[NDArray, None],
    const_model_args: Tuple[bv.BlockVector, bv.BlockVector, bv.BlockVector],
    voicing_time: NDArray,
    n_start: int = 0,
    n_stop: int = 1,
    dv_max: float = 0.05,
    t_0=0.0,
    dt=1.0,
    comp_input_0: Optional[NDArray] = None,
    swelling_dmg_growth_rate=1.0,
    swelling_healing_rate=1.0,
    damage_measure: str = 'viscous_dissipation',
    output_dir: str = 'out',
    base_fname: str = 'SwellingStep',
):
    for n in tqdm(range(n_start, n_stop), desc='Vicious cycle integration'):

        state_fpath_n = f'{output_dir}/{base_fname}{n}.h5'
        v_1, t_1, comp_input_0 = integrate_vc_step(
            model,
            state_fpath_n,
            v_0,
            voice_target,
            const_model_args,
            voicing_time,
            dv_max=dv_max,
            damage_measure=damage_measure,
            comp_input_n=comp_input_0,
            t_0=t_0,
            dt=dt,
            swelling_dmg_growth_rate=swelling_dmg_growth_rate,
            swelling_healing_rate=swelling_healing_rate,
        )
        v_0 = v_1
        t_0 = t_1


def integrate_vc_step(
    model: Model,
    state_fpath_n: str,
    v_n: NDArray,
    voice_target: NDArray,
    const_model_args: Tuple[bv.BlockVector, bv.BlockVector, bv.BlockVector],
    voicing_time: NDArray,
    t_0: float = 0,
    dt: float = 0.05,
    dv_max: float = 0.05,
    comp_input_n: Optional[NDArray] = None,
    damage_measure: str = 'viscous_dissipation',
    swelling_dmg_growth_rate: float = 1.0,
    swelling_healing_rate: float = 1.0,
):
    """
    Integrate the vicious cycle over a single step

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    state_fpath_n: str
        The filepath to write the step's simulation to
    v_n: NDArray
        The initial swelling field
    voice_target: Union[NDArray, None]
        The target voice output

        If no target voice output is given, it is assumed that the target
        voice output is that for the initial condition.
    const_model_args: bv.BlockVector
        Constant values for the model initial state, control, and property vectors
    voicing_time: NDArray
        A simulation time vector
    comp_input_n: Optional[NDArray]
        A guess for the initial compensatory input
    v_step: float
        The maximum step in swelling to take
    """
    if comp_input_n is None:
        comp_input_n = np.zeros(voice_target.shape)

    state_fpath_n, comp_input_n, compensation_solver_info = solve_comp_input(
        model,
        state_fpath_n,
        v_n,
        voice_target,
        const_model_args,
        voicing_time,
        comp_input_0=comp_input_n,
    )

    dv_dt_n_swell, dmg_rate = proc_swelling_rate(
        model,
        state_fpath_n,
        damage_measure=damage_measure,
        swelling_dmg_growth_rate=swelling_dmg_growth_rate,
    )

    dv_dt_n_heal = -swelling_healing_rate * (v_n - 1.0)

    with sf.StateFile(model, state_fpath_n, mode='r') as f:
        voice_output, info = proc_voice_output(f, len(voice_target))

    dv_dt_n = dv_dt_n_swell + dv_dt_n_heal
    max_dv_dt = np.max(np.abs(dv_dt_n))
    dt_max = dv_max / max_dv_dt

    dt = min(dt, dt_max)
    dv = dt * dv_dt_n
    v_1 = v_n + dv
    t_1 = t_0 + dt

    with h5py.File(state_fpath_n, mode='a') as f:
        group = f.file.require_group('ViciousCycle')
        values = {
            'damage_rate': dmg_rate,
            'time': t_0,
            'dv_dt_healing': dv_dt_n_heal,
            'dv_dt_damage': dv_dt_n_swell,
            'voice_target': voice_target
        }
        for key, value in values.items():
            if key not in group:
                group[key] = value

    print("-- Found compensatory input for current swelling --")
    print(f"Voice target: {voice_target}")
    print(f"Compensatory input: {comp_input_n}")
    print(f"Post compensation voice output: {voice_output}")
    print(f"Compensation solver stats: {compensation_solver_info}")
    print(
        f"(avg/max/min) swelling is ({np.mean(v_1):.4e}, {np.max(v_1):.4e}, {np.min(v_1):.4e})"
    )
    print(
        "(avg/max/min) Healing-induced swelling rate: "
        f"({np.mean(dv_dt_n_heal):.4e}, {np.max(dv_dt_n_heal):.4e}, {np.min(dv_dt_n_heal):.4e})"
    )
    print(
        "(avg/max/min) Damage-induced swelling rate: "
        f"({np.mean(dv_dt_n_swell):.4e}, {np.max(dv_dt_n_swell):.4e}, {np.min(dv_dt_n_swell):.4e})"
    )
    print("q:", info['q'])
    print(f"dt: {dt:.2e}")
    return v_1, t_1, comp_input_n


def postprocess(
    out_fpath: str,
    in_fpaths: List[str],
    overwrite_results: Optional[List[str]] = None,
    num_proc: int = 1,
):
    """
    Postprocess key signals from the simulations
    """
    # signal_to_proc = get_result_to_proc(model)
    # result_names = list(signal_to_proc.keys())

    with h5py.File(out_fpath, mode='a') as f:
        for in_fpath in in_fpaths:
            case_name = path.splitext(in_fpath.split('/')[-1])[0]
            print(case_name)
            postprocutils.postprocess_parallel(
                f.require_group(case_name),
                in_fpath,
                lambda name: model,
                main.get_result_name_to_postprocess,
                num_proc=num_proc,
                overwrite_results=overwrite_results,
            )


if __name__ == '__main__':
    from os import path

    import cases

    parser = ArgumentParser()
    # Whether to run different parts of processing
    parser.add_argument("--run-vc-sim", action='store_true')
    parser.add_argument("--export-xdmf", action='store_true')
    parser.add_argument("--postprocess", action='store_true')

    # Where/how to write results
    parser.add_argument("--output-dir", type=str, default='out/vicious_cycle')
    parser.add_argument("--overwrite-results", type=str, action='extend', nargs='+')

    # Control which damage measure is used for swelling
    parser.add_argument(
        "--damage-measure", type=str, default='field.tavg_viscous_dissipation'
    )

    # Voicing time parameters
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--nt", type=int, default=2**13)

    # Vicious cycle integration parameters
    parser.add_argument("--dv", type=float, default=0.05)
    parser.add_argument("--nstart", type=int, default=0)
    parser.add_argument("--nstop", type=int, default=1)

    # Swelling rate parameters
    # The time unit of the vicious cycle is in hours
    # for '--swelling-heal-rate' `0.5` represents a 50% reduction while `5` represents 5 hours
    parser.add_argument("--swelling-dmg-rate", type=float, default=0.0)
    parser.add_argument("--swelling-heal-rate", type=float, default=-np.log(0.5) / (5))

    cmd_args = parser.parse_args()

    param = cases.VCExpParam(
        {
            'MeshName': cases.MESH_BASE_NAME,
            'clscale': 0.75,
            'GA': 3,
            'DZ': 1.5,
            'NZ': 15,
            'Ecov': cases.ECOV,
            'Ebod': cases.EBOD,
            'vcov': 1.0,
            'mcov': 0.0,
            'psub': 400 * 10,
            'dt': 5e-5,
            'tf': 0.50,
            'ModifyEffect': '',
            'SwellingDistribution': 'uniform',
            'SwellingModel': 'power',
            'SwellHealRate': cmd_args.swelling_heal_rate,
            'SwellDamageRate': cmd_args.swelling_dmg_rate,
        }
    )
    # param = main.ExpParam(
    #     {
    #         'MeshName': cases.MESH_BASE_NAME,
    #         'clscale': 0.94,
    #         'GA': 3,
    #         'DZ': 1.5,
    #         'NZ': 12,
    #         'Ecov': cases.ECOV,
    #         'Ebod': cases.EBOD,
    #         'vcov': 1,
    #         'mcov': 0.0,
    #         'psub': 400 * 10,
    #         'dt': 5e-5,
    #         'tf': 0.50,
    #         'ModifyEffect': '',
    #         'SwellingDistribution': 'uniform',
    #         'SwellingModel': 'power',
    #     }
    # )
    model = main.setup_model(param)

    dv = cmd_args.dv
    n_start = cmd_args.nstart
    n_stop = cmd_args.nstop

    healing_rate = param['SwellHealRate']
    damage_rate = param['SwellDamageRate']

    base_fname = f'DamageMeasure{cmd_args.damage_measure}--DamageRate{damage_rate:.4e}--HealRate{healing_rate:.4e}--Step'
    fpaths = [
        f'{cmd_args.output_dir}/{base_fname}{n:d}.h5' for n in range(n_start, n_stop)
    ]

    if cmd_args.run_vc_sim:
        # Check that you won't overwrite existing files, excluding the initial state
        if any(path.isfile(fpath) for fpath in fpaths[1:]):
            raise RuntimeError(
                f"Some existing files, {fpaths[1:]}, would be overwritten."
            )

        const_ini_state, const_controls, const_prop = main.setup_state_control_prop(
            param, model
        )

        # This is how long to integrate the 'voicing' simulations for, which
        # are used to determine damage rates, swelling fields, etc.
        voicing_time = cmd_args.dt * np.arange(cmd_args.nt)

        # `v0` and `x0` are the initial swelling field and compensatory inputs
        v_0 = np.array(const_prop['v_swelling'][:])
        x_0 = np.array([0, 0])
        x_0 = np.array([0])

        fpath_0 = fpaths[0]
        if not path.isfile(fpath_0):
            const_model_args = (const_ini_state, const_controls[0], const_prop)
            integrate_vc(
                model,
                v_0,
                None,
                const_model_args,
                voicing_time,
                n_start=n_start,
                n_stop=n_stop,
                dv_max=dv,
                output_dir=cmd_args.output_dir,
                base_fname=base_fname,
                comp_input_0=x_0,
                dt=1.0,
                swelling_dmg_growth_rate=damage_rate,
                swelling_healing_rate=healing_rate,
                damage_measure=cmd_args.damage_measure,
            )
        else:
            resume_integrate_vc(
                model,
                n_start,
                n_stop,
                dv_max=dv,
                output_dir=cmd_args.output_dir,
                dt=1.0,
                swelling_dmg_growth_rate=damage_rate,
                swelling_healing_rate=healing_rate,
                base_fname=base_fname,
                damage_measure=cmd_args.damage_measure,
            )

    if cmd_args.postprocess:
        _postprocess = functools.partial(
            postprocess, overwrite_results=cmd_args.overwrite_results
        )

        out_fpath = f'{cmd_args.output_dir}/postprocess.h5'
        _postprocess(out_fpath, fpaths, num_proc=1)

    if cmd_args.export_xdmf:
        with h5py.File(f'{cmd_args.output_dir}/postprocess.h5', mode='r') as fpost:
            for fpath in fpaths:
                with h5py.File(fpath, mode='r') as fstate:
                    fdir, fname = path.split(fpath)
                    fname_head, fname_suffix = path.splitext(fname)
                    xdmf_path = f'{fdir}/{fname_head}--export.xdmf'
                    postprocess_xdmf(model, fstate, fpost[fname_head], xdmf_path)
