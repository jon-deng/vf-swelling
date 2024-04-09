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

from typing import Any, Mapping, Tuple, Optional, Union
from numpy.typing import NDArray

from tqdm import tqdm
from argparse import ArgumentParser
import warnings

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
SolverInfo = Mapping[str, Any]

def proc_time(f: sf.StateFile) -> NDArray:
    """
    Return the simulation times vector
    """
    return f.get_times()

def proc_glottal_flow_rate(f: sf.StateFile) -> NDArray:
    """
    Return the glottal flow rate vector
    """
    num_fluid = len(f.model.fluids)

    # Compute q as a weighted average over all coronal cross-sections
    qs = np.array(
        [f.file[f'state/fluid{n}.q'] for n in range(num_fluid)]
    )

    # Assign full weights to all coronal sections with neighbours and
    # half-weights to the anterior/posterior coronal sections
    weights = np.ones(num_fluid)
    weights[[0, -1]] = 0.5
    q = np.sum(np.array(qs)[..., 0] * weights[:, None], axis=0)/np.sum(weights)
    return np.array(q)

def proc_voice_output(f: sf.StateFile, n: int) -> NDArray:
    """
    Return voice outputs (RMS pressure, fundamental frequency)
    """
    t = proc_time(f)
    q = proc_glottal_flow_rate(f)

     # Truncate part of the flow rate signal to a steady state portion
    idx_trunc = slice(t.size//2, t.size)
    t_trunc = t[idx_trunc]
    q_trunc = q[idx_trunc]

    dt = t[1] - t[0]
    fund_freq, fund_phase, dfreq, dphase, info = \
        modal.fundamental_mode_from_peaks(q, dt, height=np.max(q)*0.8)
    if len(info['peaks']) < 2:
        fund_freq = 0.0

    prms = calc_prms(t_trunc, q_trunc)/10

    _voice_output = (prms, fund_freq)
    # voice_output = np.array([prms, fund_freq])
    voice_output = np.array(_voice_output[:n])
    return voice_output

def calc_prms(t: NDArray, q: NDArray) -> float:
    """
    Return the RMS radiated pressure at 1 m using a piston-in-baffle approximation
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


def map_vc_input_to_model_input(
        model: Model, v: NDArray, comp_input: NDArray,
        const_ini_state: bv.BlockVector,
        const_control: bv.BlockVector,
        const_prop: bv.BlockVector,
        ini_state_solved: bool=False
    ) -> Tuple[bv.BlockVector, bv.BlockVector, bv.BlockVector, SolverInfo]:
    """
    Return model parameters from compensatory inputs

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    comp_input: NDArray
        The compensatory input vector

        This is assumed to be in the order:
        (subglottal pressure, modulus increase).
    const_control, const_prop: bv.BlockVector
        Any constant values of the model control and property vectors
    ini_state: Optional[bv.BlockVector]
        An optional initial state to return

        If an initial state is not provided, it is calculated such that the
        model is static under the given swelling field and with no external
        loads.
    """
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
        demod = const_prop['emod']/np.max(const_prop['emod'])
        dprop['emod'][:] = comp_input[1]*demod
    prop = const_prop + dprop

    prop['v_swelling'][:] = v

    # Compute `ini_state` as a swollen static state without external loading
    ini_state = const_ini_state.copy()
    if not ini_state_solved:
        ini_state[:] = 0

        model.set_control(control)
        model.set_prop(prop)
        sl_control = model.solid.control.copy()
        sl_prop = model.solid.prop.copy()
        # Ensure there's no external loading
        sl_control['p'][:] = 0

        # Compute the swollen static state
        # Use 1 loading step for each 5% swelling increase
        vmax = np.max(sl_prop['v_swelling'])
        nload = int(round(1/.05 * vmax))
        static_state, static_solve_info = main.solve_static_swollen_config(
            model.solid, sl_control, sl_prop, nload=nload
        )
        ini_state[['u', 'v', 'a']] = static_state

        solve_status = static_solve_info.get('status')
        if solve_status != 0 and solve_status is not None:
            raise RuntimeError("A static swollen state could not be solved")
    else:
        static_solve_info = {}

    return ini_state, control, prop, static_solve_info

def make_voice_output_jac(
        model: Model,
        voice_output_0: NDArray,
        v: NDArray, comp_input: NDArray,
        const_ini_state: bv.BlockVector, const_control: bv.BlockVector, const_prop: bv.BlockVector,
        times: NDArray,
        fpath: str='tmp.h5'
    ) -> NDArray:
    """
    Return model parameters needed to achieve target voice outputs

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    comp_input: NDArray
        The compensatory input vector (linearization point)

        This is assumed to be in the order:
        (subglottal pressure, modulus increase).
    voice_output: NDArray
        The voice otuput at the linearization point
    const_control, const_prop: bv.BlockVector
        Any constant values of the model control and property vectors
    times: NDArray
        A simulation time vector
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
        model,
        v, comp_input,
        const_ini_state, const_control, const_prop,
        ini_state_solved=False
    )

    # Calculate voice output sensitivity to phonation parameters
    # with a FD approximation
    # Populate the voice output sensitivity matrix column-by-column
    # For the FD increments:
    # Use a subglottal pressure change of 10 Pa
    # Use a maximum stiffness change of 10 kPa
    _dinputs = (100 * 10, 10e3 * 10)
    dinputs = list(np.diag(_dinputs))[:len(comp_input)]

    def form_jac_column(dinput):
        with sf.StateFile(model, fpath, mode='w') as f:
            _, solver_info = integrate(
                model, f,
                v, comp_input+dinput,
                ini_state, const_control, const_prop, times,
                ini_state_solved=True
            )
            voice_output1 = proc_voice_output(f, len(comp_input))

        return (voice_output1-voice_output_0)/np.linalg.norm(dinput)

    voice_output_jac = np.array([
        form_jac_column(dinput) for dinput in dinputs
    ])

    # print(voice_output_jac)

    return voice_output_jac

def solve_comp_input(
        model: Model,
        fpath: str,
        v: NDArray, voice_output_target: NDArray,
        const_ini_state: bv.BlockVector, const_control: bv.BlockVector, const_prop: bv.BlockVector,
        times: NDArray,
        comp_input_0: Optional[NDArray]=None
    ) -> Tuple[bv.BlockVector, SolverInfo]:
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
    voice_output_target: NDArray
        The target voice otuputs
    const_ini_state, const_control, const_prop: bv.BlockVector
        Any constant values of the model control and property vectors
    times: NDArray
        A simulation time vector
    comp_input_0: NDArray
        The compensatory input vector (linearization point)

        This is assumed to be in the order:
        (subglottal pressure, modulus increase).
    """
    # Compute the swollen initial state once so you can reuse it for each
    # simulation
    # ; you only have to compute it once because the swelling magnitude remains
    # constant
    ini_state, *_ = map_vc_input_to_model_input(
        model,
        v, comp_input_0,
        const_ini_state, const_control, const_prop, ini_state_solved=True
    )

    # Use an iterative Newton method to find the appropriate compensatory
    # adjustments
    def lin_subproblem(x):
        # Calculate voice output sensitivity to phonation parameters
        # with a FD approximation
        with sf.StateFile(model, fpath, mode='w') as f:
            integrate(
                model, f,
                v, x, ini_state, const_control, const_prop, times,
                ini_state_solved=True
            )
            y = proc_voice_output(f, len(comp_input_0))
            path_head, path_tail = path.splitext(fpath)
            jac = make_voice_output_jac(
                model, y,
                v, x,
                const_ini_state, const_control, const_prop, times,
                fpath=f'{path_head}--tmp{path_tail}'
            )
        y1 = voice_output_target

        def assem_res():
            return y1 - y

        def assem_jac(res):
            return np.linalg.solve(jac, -res)

        return assem_res, assem_jac

    comp_input, info = newton_solve(
        comp_input_0, lin_subproblem, lambda x: np.linalg.norm(x),
        params={
            'absolute_tolerance': 1e-4,
            'relative_tolerance': 1e-8,
            'maximum_iterations': 10
        }
    )

    if info['status'] != 0:
        raise RuntimeError(
            "Couldn't find compensatory input!"
            "Solver failed with info: "
            f"{info}"
        )

    return comp_input, info

def proc_damage_rate(
        model: Model,
        f: sf.StateFile
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
    state_measure = slsig.ViscousDissipationField(
        model, dx=dx, fspace=fspace
    )
    def measure(f):
        mean = TimeSeriesStats(state_measure).mean(f, range(f.size//2, f.size))
        return mean

    damage = measure(f)
    return damage

def calc_swelling_rate(
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
        v: NDArray, comp_input: NDArray,
        const_ini_state: bv.BlockVector,
        const_control: bv.BlockVector,
        const_prop: bv.BlockVector,
        times: NDArray,
        ini_state_solved=False
    ) -> Tuple[bv.BlockVector, SolverInfo]:
    """
    Integrate the model over time

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    comp_input: NDArray
        The compensatory input vector

        This is assumed to be in the order:
        (subglottal pressure, modulus increase).
    const_control, const_prop: bv.BlockVector
        Any constant values of the model control and property vectors
    ini_state: Optional[bv.BlockVector]
        An optional initial state

        If an initial state is not provided, it is calculated such that the
        model is static under the given swelling field and with no external
        loads.
    """
    ini_state, control, prop, _ = map_vc_input_to_model_input(
        model, v, comp_input,
        const_ini_state, const_control, const_prop,
        ini_state_solved=ini_state_solved
    )
    return forward.integrate(
        model, f, ini_state, [control], prop, times, use_tqdm=True
    )

def postprocess_xdmf(
        model: Model, fstate: h5py.File, xdmf_path: str,
        overwrite: bool=False
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


def integrate_vc(
        model: Model,
        v_0: NDArray, voice_target: Union[NDArray, None],
        const_ini_state: bv.BlockVector,
        const_control: bv.BlockVector,
        const_prop: bv.BlockVector,
        voicing_time: NDArray,
        n_start: int=0, n_stop: int=1, v_step: float=0.05,
        comp_input_0: Optional[NDArray]=None,
        output_dir: str='out',
        base_fname: str='SwellingStep'
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
    const_ini_state, const_control, const_prop: bv.BlockVector
        Constant values for the model initial state, control, and property vectors
    voicing_time: NDArray
        A simulation time vector
    n_start: int
        The initial state index of the vicious cycle
    n_stop: int
        The final state index of the vicious cycle
    v_step: float
        The increment in swelling to take for each step of the vicious cycle
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
    elif comp_input_0 is None and voice_target is not None:
        comp_input_0 = np.zeros(voice_target.shape)

    # Establish the voice target if none is provided
    if voice_target is None:
        state_fpath_0 = f'{output_dir}/{base_fname}{n_start}.h5'
        with sf.StateFile(model, state_fpath_0, mode='w') as f:
            _, solve_info = integrate(
                model, f, v_0, comp_input_0,
                const_ini_state, const_control, const_prop, voicing_time,
                ini_state_solved=False
            )
            voice_target = proc_voice_output(f, len(comp_input_0))

    ## Loop through steps of the vicious cycle (VC)
    # comp_input_n = comp_input_0
    for n in tqdm(
            range(n_start, n_stop+1),
            desc='Vicious cycle integration'
        ):

        state_fpath_n = f'{output_dir}/{base_fname}{n-1}.h5'
        v_1, comp_input_0 = integrate_vc_step(
            model, state_fpath_n,
            v_0, voice_target,
            const_ini_state, const_control, const_prop, voicing_time,
            v_step=v_step,
            comp_input_n=comp_input_0
        )
        v_0 = v_1

def solve_swelling_rate(
        model: Model,
        fpath: str,
        v: NDArray, comp_input: NDArray,
        const_ini_state: bv.BlockVector,
        const_control: bv.BlockVector,
        const_prop: bv.BlockVector,
        voicing_time: NDArray
    ):
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
    const_control, const_prop: bv.BlockVector
        Any constant values for the model control and property vectors
    voicing_time: NDArray
        A voicing time vector
    """
    if not path.isfile(fpath):
        with sf.StateFile(model, fpath, mode='w') as f:
            _, solve_info = integrate(
                model, f,
                v, comp_input,
                const_ini_state, const_control, const_prop, voicing_time
            )
            voice_target = proc_voice_output(f, len(voice_target))

    with sf.StateFile(model, fpath, mode='r') as f:
        dmg_rate = proc_damage_rate(model, f)
        swelling_rate = calc_swelling_rate(dmg_rate)

    return swelling_rate

def integrate_vc_step(
        model: Model,
        state_fpath_n: str,
        v_n: NDArray, voice_target: NDArray,
        const_ini_state: bv.BlockVector,
        const_control: bv.BlockVector,
        const_prop: bv.BlockVector,
        voicing_time: NDArray,
        comp_input_n: Optional[NDArray]=None,
        v_step: float=0.05
    ):
    if comp_input_n is None:
        comp_input_n = np.zeros(voice_target.shape)

    # input `v_n` `comp_input_n` `voice_target` `state_fpath`
    const_prop_n = const_prop.copy()
    const_prop_n['v_swelling'] = v_n

    ini_state, *_, solve_info = map_vc_input_to_model_input(
        model,
        v_n, comp_input_n,
        const_ini_state, const_control, const_prop_n
    )

    comp_input_n, compensation_solver_info = solve_comp_input(
        model, state_fpath_n,
        v_n, voice_target,
        const_ini_state, const_control, const_prop_n, voicing_time,
        comp_input_0=comp_input_n
    )

    vd_0 = solve_swelling_rate(
        model, state_fpath_n,
        v_n, comp_input_n,
        const_ini_state, const_control, const_prop_n, voicing_time
    )

    with sf.StateFile(model, state_fpath_n, mode='r') as f:
        voice_output = proc_voice_output(f, len(voice_target))

    print("-- Found compensatory input for current swelling --")
    print(f"Voice target: {voice_target}")
    print(f"Compensatory input: {comp_input_n}")
    print(f"Post compensation voice output: {voice_output}")
    print(f"Compensation solver stats: {compensation_solver_info}")

    dv = v_step*vd_0/vd_0.max()
    v_1 = v_n + dv
    return v_1, comp_input_n

def resume_integrate_vc(
        n_start: int, n_stop: int,
        v_step: float=0.05,
        output_dir: str='out',
        base_fname: str='SwellingStep'
    ):
    state_fpath_0 = f'{output_dir}/{base_fname}{n_start}.h5'
    with sf.StateFile(model, state_fpath_0, mode='r') as f:
        voice_target = proc_voice_output(state_fpath_0, 1)
        const_ini_state = f.get_state(0)
        const_control = f.get_control(0)
        const_prop = f.get_prop()
        voicing_time = f.get_times()

    ## Loop through steps of the vicious cycle (VC)
    # comp_input_n = comp_input_0
    # NOTE: Ideally you should be able to figure out what the compensatory input
    # was from the initial state file
    comp_input_0 = np.array([0])
    for n in tqdm(
            range(n_start, n_stop+1),
            desc='Vicious cycle integration'
        ):

        state_fpath_n = f'{output_dir}/{base_fname}{n}.h5'
        v_1, comp_input_0 = integrate_vc_step(
            model, state_fpath_n,
            v_0, voice_target,
            const_ini_state, const_control, const_prop, voicing_time,
            v_step=v_step,
            comp_input_n=comp_input_0
        )
        v_0 = v_1

if __name__ == '__main__':
    from os import path

    import main
    import cases

    parser = ArgumentParser()
    parser.add_argument('--export-xdmf', action='store_true')
    parser.add_argument('--output-dir', type=str, default='out')
    cmd_args = parser.parse_args()

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
    param = main.ExpParam({
        'MeshName': cases.MESH_BASE_NAME, 'clscale': 0.94,
        'GA': 3,
        'DZ': 1.5, 'NZ': 12,
        'Ecov': cases.ECOV, 'Ebod': cases.EBOD,
        'vcov': 1, 'mcov': 0.0,
        'psub': 600*10,
        'dt': 5e-5, 'tf': 0.50,
        'ModifyEffect': '',
        'SwellingDistribution': 'uniform',
        'SwellingModel': 'power'
    })
    model = main.setup_model(param)

    N_START = 0
    N_STOP = 4
    fpaths = [
        f'{cmd_args.output_dir}/SwellingStep{n}.h5'
        for n in range(N_START, N_STOP+1)
    ]
    # Check that you won't overwrite existing files, excluding the initial state
    if any(path.isfile(fpath) for fpath in fpaths[1:]):
        raise RuntimeError(
            f"Some existing files, {fpaths[1:]}, would be overwritten."
        )
    else:
        const_ini_state, const_controls, const_prop = main.setup_state_control_props(param, model)

        # This is how long to integrate the 'voicing' simulations for, which
        # are used to determine damage rates, swelling fields, etc.
        voicing_time = 5e-5*np.arange(2**3)
        # times = 5e-5*np.arange(2**6)
        # voicing_time = 5e-5*np.arange(2**8)
        # times = 5e-5*np.arange(2**10)
        # times = 5e-5*np.arange(2**13)
        # times = 5e-5*np.arange(10000+1)

        # `v0` and `x0` are the initial swelling field and compensatory inputs
        v_0 = np.ones(const_prop['v_swelling'].shape)
        x_0 = np.array([0, 0])
        x_0 = np.array([0])
        integrate_vc(
            model,
            v_0, None,
            const_ini_state, const_controls[0], const_prop, voicing_time,
            n_start=N_START, n_stop=N_STOP,
            v_step=0.05, output_dir=cmd_args.output_dir,
            comp_input_0=x_0
        )

    if cmd_args.export_xdmf:
        for fpath in fpaths:
            with h5py.File(fpath, mode='r') as fstate:
                h5_head, h5_suffix = path.splitext(fpath)
                xdmf_path = f'{h5_head}--export.xdmf'
                postprocess_xdmf(
                    model, fstate, xdmf_path
                )
