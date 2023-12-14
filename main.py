"""
Run a sequence of vocal fold simulations with swelling
"""

from typing import List, Tuple, Mapping, Optional
from numpy.typing import NDArray

from os import path
import argparse as ap
import multiprocessing as mp
import itertools as it
import functools

import numpy as np
import dolfin as dfn
import h5py

from femvf import forward, static, statefile as sf, meshutils
from femvf.models.transient import solid, fluid, base as trabase, coupled
from femvf.models.dynamical import base as dynbase
from femvf.postprocess.base import TimeSeries, TimeSeriesStats
from femvf.postprocess import solid as slsig
from femvf.load import load_transient_fsi_model
from exputils import postprocutils, exputils

from blockarray import blockvec as bv

dfn.set_log_level(50)

## Defaults for 'nominal' parameter values
MESH_BASE_NAME = 'M5_BC'

CLSCALE = 1

POISSONS_RATIO = 0.4

PSUB = 300 * 10

VCOVERS = np.array([1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3])
MCOVERS = np.array([0.0, -0.4, -0.8, -1.2, -1.6])

ECOV = 2.5e4
EBOD = 5e4

PARAM_SPEC = {
    'MeshName': str,
    'GA': float,
    'DZ': float,
    'NZ': int,
    'clscale': float,
    'Ecov': float,
    'Ebod': float,
    'vcov': float,
    'mcov': float,
    'psub': float,
    'dt': float,
    'tf': float,
    'ModifyEffect': str
}
ExpParam = exputils.make_parameters(PARAM_SPEC)

Model = coupled.BaseTransientFSIModel

def setup_mesh_name(params: ExpParam) -> str:
    """
    Return the name of the mesh
    """
    base_name = params['MeshName']
    ga = params['GA']
    clscale = params['clscale']
    dz = params['DZ']
    nz = params['NZ']
    return f'{base_name}--GA{ga:.2f}--DZ{dz:.2f}--NZ{nz:d}--clscale{clscale:.2e}'

def setup_model(params: ExpParam) -> Model:
    """
    Return the model
    """
    mesh_path = f"mesh/{setup_mesh_name(params)}.msh"

    if params['DZ'] == 0.0:
        zs = None
    elif params['DZ'] > 0.0:
        zs = np.linspace(0, params['DZ'], params['NZ']+1)
    else:
        raise ValueError("Parameter 'DZ' must be >= 0")

    model = load_transient_fsi_model(
        mesh_path, None,
        SolidType=solid.SwellingKelvinVoigtWEpitheliumNoShape,
        FluidType=fluid.BernoulliAreaRatioSep,
        zs=zs
    )
    return model

def setup_state_control_props(
        params: ExpParam, model: Model
    ) -> Tuple[bv.BlockVector, bv.BlockVector, bv.BlockVector]:
    """
    Return a (state, controls, prop) tuple defining a transient run
    """
    ## Set 'basic' model properties
    # These properties don't include the glottal gap since you may/may not
    # want to modify the glottal gap based on the swelling level
    prop = setup_basic_props(params, model)
    model.set_prop(prop)

    ## Set the initial state
    # The initial state is based on the post-swelling static configuration
    state0 = setup_ini_state(params, model)

    # Set the glottal gap based on the post-swelling static configuration
    ndim = model.solid.residual.mesh().topology().dim()
    if (params['ModifyEffect'] == 'const_pregap'
        or params['ModifyEffect'] == 'const_mass_pregap'
        ):
        # Using the `ndim` to space things ensures you get the y-coordinate
        # for both 2D and 3D meshes
        ymax = (model.solid.XREF + state0.sub['u'])[1::ndim].max()
    else:
        ymax = (model.solid.XREF)[1::ndim].max()
    ygap = 0.03 # 0.3 mm half-gap -> 0.6 mm glottal gap
    ycoll_offset = 1/10*ygap

    prop['ycontact'] = ymax + ygap - ycoll_offset
    prop['ymid'] = ymax + ygap
    for n in range(len(model.fluids)):
        prop[f'fluid{n}.area_lb'] = 2*ycoll_offset

    model.set_prop(prop)

    controls = setup_controls(params, model)
    return state0, controls, prop

def setup_basic_props(params: ExpParam, model: Model) -> bv.BlockVector:
    """
    Set the properties vector
    """
    mesh = model.solid.residual.mesh()
    forms = model.solid.residual.form
    mf = model.solid.residual.mesh_function('cell')
    mf_label_to_value = model.solid.residual.mesh_function_label_to_value('cell')
    cellregion_to_sdof = meshutils.process_meshlabel_to_dofs(
        mesh, mf, mf_label_to_value,
        forms['coeff.prop.emod'].function_space().dofmap()
    )

    prop = model.prop.copy()
    # prop[:] = 0
    ## Solid constant properties
    prop['rho'] = 1.0
    prop['eta'] = 5.0
    prop['kcontact'] = 1e15
    # prop['ncontact'] = [0, 1]

    ## Fluid constant properties
    for n in range(len(model.fluids)):
        prop[f'fluid{n}.r_sep'] = 1.2
        prop[f'fluid{n}.rho_air'] = 1.2e-3

    ## Swelling specific properties
    if (
            params['ModifyEffect'] == 'const_pregap'
            or params['ModifyEffect'] == ''
        ):
        # This one is controlled in the initial state
        modify_kwargs = {}
    elif (
            params['ModifyEffect'] == 'const_mass'
            or params['ModifyEffect'] == 'const_mass_pregap'
        ):
        modify_kwargs = {'modify_density': False}
    else:
        raise ValueError(f"Unkown 'ModifyEffect' parameter {params['ModifyEffect']}")

    prop = _set_swelling_props(
        prop, params['vcov'], params['mcov'], cellregion_to_sdof,
        **modify_kwargs
    )

    ## Set VF layer properties
    emods = {
        'cover': params['Ecov'],
        'body': params['Ebod']
    }
    prop = _set_layer_props(prop, emods, cellregion_to_sdof)

    return prop

def setup_controls(params: ExpParam, model: Model) -> bv.BlockVector:
    """
    Set the controls
    """
    control = model.control.copy()
    control[:] = 0

    for n in range(len(model.fluids)):
        control[f'fluid{n}.psub'] = params['psub']
        control[f'fluid{n}.psup'] = 0.0

    return [control]

def setup_ini_state(params: ExpParam, model: Model) -> bv.BlockVector:
    """
    Set the initial state vector
    """
    state0 = model.state0.copy()
    state0[:] = 0.0
    model.solid.control[:] = 0.0

    vcov = params['vcov']
    nload = max(int(round((vcov-1)/0.025)), 1)

    prop = setup_basic_props(params, model)
    model.set_prop(prop)

    static_state, _ = solve_static_swollen_config(
        model.solid, model.solid.control, model.solid.prop, nload
    )

    state0[['u', 'v', 'a']] = static_state
    return state0

def _set_swelling_props(
        prop: bv.BlockVector,
        v: float, m: float,
        cellregion_to_sdof: Mapping[str, NDArray],
        modify_density=True,
        modify_geometry=True
    ) -> bv.BlockVector:
    """
    Set properties related to the level of swelling

    Parameters
    ----------
    v :
        swelling field
    """
    RHO_VF = 1.0
    RHO_SWELL = 1.0
    # KSWELL_FACTOR = 20

    # Apply swelling parameters to the cover layer only
    # dofs_cov = np.unique(
    #     np.concatenate(
    #         [cellregion_to_sdof[label] for label in ['medial', 'inferior', 'superior']]
    #     )
    # )
    dofs_cov = np.unique(
        np.concatenate(
            [cellregion_to_sdof[label] for label in ['cover']]
        )
    )
    dofs_bod = cellregion_to_sdof['body']
    # dofs_sha = np.intersect1d(dofs_cov, dofs_bod)

    # prop['k_swelling'][0] = KSWELL_FACTOR * 10e3*10
    prop['m_swelling'][dofs_cov] = m

    prop['v_swelling'][:] = 1.0
    if modify_geometry:
        prop['v_swelling'][dofs_cov] = v
        prop['v_swelling'][dofs_bod] = 1.0

    prop['rho'][:] = RHO_VF
    if modify_density:
        _v = np.array(prop['v_swelling'][:])
        prop['rho'][:] = RHO_VF + (_v-1)*RHO_SWELL

    return prop

def _set_layer_props(
        prop: bv.BlockVector,
        emods: Mapping[str, float],
        cellregion_to_sdof: Mapping[str, NDArray]
    ) -> bv.BlockVector:
    """
    Set properties for each layer of a model

    Parameters
    ----------
    emod_vec :
        The vector of nodal values of elastic moduli
    emods : dict
        A mapping from named regions to modulus values
    cellregion_to_sdof:
        A mapping from names regions to mesh function values
    """
    # dofs_cov = np.unique(
    #     np.concatenate(
    #         [cellregion_to_sdof[label] for label in ['medial', 'inferior', 'superior']]
    #     )
    # )
    dofs_cov = np.unique(
        np.concatenate(
            [cellregion_to_sdof[label] for label in ['cover']]
        )
    )
    dofs_bod = cellregion_to_sdof['body']
    prop['emod'][dofs_bod] = emods['body']
    prop['emod'][dofs_cov] = emods['cover']

    prop['nu'][:] = POISSONS_RATIO

    # membrane/epithelium properties
    prop['emod_membrane'][:] = 50e3 * 10
    # prop['emod_membrane'][:] = 0.0
    prop['th_membrane'][:] = 0.005
    prop['nu_membrane'][:] = POISSONS_RATIO
    return prop


def solve_static_swollen_config(
        model: Model,
        control: bv.BlockVector,
        prop: bv.BlockVector,
        nload: int=1
    ):
    """
    Solve for the static swollen configuration

    This uses `nload` loading steps of the swelling field
    """
    if isinstance(model, trabase.BaseTransientModel):
        static_state_n = model.state0.copy()
        static_state_n[:] = 0
    elif isinstance(model, dynbase.BaseDynamicalModel):
        static_state_n = model.state.copy()
        static_state_n[:] = 0

    v_final = prop['v_swelling'][:].copy()
    v_0 = v_final.copy()
    v_0[:] = 1
    dv = (v_final-v_0)/nload

    props_n = prop.copy()
    props_n['v_swelling'][:] = v_0

    info = {}
    for n in range(nload+1):
        props_n['v_swelling'][:] = v_0 + n*dv
        static_state_n, info = static.static_solid_configuration(
            model, control, props_n, state=static_state_n
        )
    return static_state_n, info


def make_exp_params(study_name: str) -> List[ExpParam]:
    if study_name == 'none':
        paramss = []
    elif study_name == 'test':
        vcovs = [1.0, 1.1, 1.2, 1.3]
        # vcovs = [1.0]
        paramss = [
            ExpParam({
                'MeshName': MESH_BASE_NAME, 'clscale': 0.5,
                'GA': 3,
                'DZ': 1.50, 'NZ': 10,
                'Ecov': ECOV, 'Ebod': EBOD,
                'vcov': vcov,
                'mcov': -0.8,
                'psub': 300*10,
                'dt': 1.25e-5, 'tf': 0.1,
                'ModifyEffect': ''
            })
            for vcov in vcovs
        ]
    elif study_name == 'test_3d_onset':
        psubs = 10*np.arange(300, 1001, 100)
        paramss = [
            ExpParam({
                'MeshName': MESH_BASE_NAME, 'clscale': 0.5,
                'GA': 3,
                'DZ': 1.50, 'NZ': 10,
                'Ecov': ECOV, 'Ebod': EBOD,
                'vcov': 1.0,
                'mcov': -0.8,
                'psub': psub,
                'dt': 1.25e-5, 'tf': 0.1,
                'ModifyEffect': ''
            })
            for psub in psubs
        ]
    elif study_name == 'debug_time_psub':
        vcovs = [1.0, 1.3]
        fdts = [1, 2, 4, 8]
        paramss = [
            ExpParam({
                'MeshName': MESH_BASE_NAME, 'clscale': 1,
                'Ecov': ECOV, 'Ebod': EBOD,
                'vcov': vcov,
                'mcov': -0.8,
                'psub': 300*10,
                'dt': 1.25e-5/fdt, 'tf': 0.2,
                'ModifyEffect': ''
            })
            for fdt in fdts
            for vcov in vcovs
        ]
    elif study_name == 'mesh_time_independence':
        def make_params(clscale, dt):
            return ExpParam({
                'MeshName': MESH_BASE_NAME, 'clscale': clscale,
                'Ecov': ECOV, 'Ebod': EBOD,
                'vcov': 1.3,
                'mcov': -1.6,
                'psub': PSUB,
                'dt': dt, 'tf': 0.5,
                'ModifyEffect': ''
            })

        clscales = 1 * 2.0**np.arange(1, -2, -1)
        dts = 5e-5 * 2.0**np.arange(0, -5, -1)
        paramss = [
            make_params(clscale, dt) for clscale in clscales for dt in dts
        ]
    elif study_name == 'main':
        def make_params(elayers, vcov, mcov):
            return ExpParam({
                'MeshName': MESH_BASE_NAME, 'clscale': CLSCALE,
                'DZ': 0.00, 'NZ': 1,
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov,
                'mcov': mcov,
                'psub': PSUB,
                'dt': DT, 'tf': TF,
                'ModifyEffect': ''
            })

        paramss = [
            make_params(*args) for args in it.product(EMODS, VCOVERS, MCOVERS)
        ]
    elif study_name == 'main_3D':
        def make_params(elayers, vcov, mcov):
            return ExpParam({
                'MeshName': MESH_BASE_NAME, 'clscale': CLSCALE,
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'DZ': 1.5, 'NZ': 10,
                'vcov': vcov,
                'mcov': mcov,
                'psub': PSUB,
                'dt': DT, 'tf': TF,
                'ModifyEffect': ''
            })

        paramss = [
            make_params(*args) for args in it.product(EMODS, VCOVERS, MCOVERS)
        ]
    elif study_name == 'main_coarse':
        def make_params(elayers, vcov, mcov):
            return ExpParam({
                'MeshName': MESH_BASE_NAME, 'clscale': CLSCALE,
                'DZ': 0.00, 'NZ': 1,
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov,
                'mcov': mcov,
                'psub': PSUB,
                'dt': 2e-4, 'tf': 0.5,
                'ModifyEffect': ''
            })

        paramss = [
            make_params(*args) for args in it.product(EMODS, VCOVERS, MCOVERS)
        ]
    elif study_name == 'main_coarse_3D':
        def make_params(elayers, vcov, mcov):
            return ExpParam({
                'MeshName': MESH_BASE_NAME, 'clscale': CLSCALE,
                'DZ': 1.5, 'NZ': 10,
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov,
                'mcov': mcov,
                'psub': PSUB,
                'dt': 2e-4, 'tf': 0.5,
                'ModifyEffect': ''
            })

        paramss = [
            make_params(*args) for args in it.product(EMODS, VCOVERS, MCOVERS)
        ]
    elif study_name == 'const_pregap':
        def make_params(elayers, vcov, mcov):
            return ExpParam({
                'MeshName': MESH_BASE_NAME, 'clscale': CLSCALE,
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov,
                'mcov': mcov,
                'psub': PSUB,
                'dt': DT, 'tf': TF,
                'ModifyEffect': 'const_pregap'
            })

        paramss = [
            make_params(*args) for args in it.product(EMODS, VCOVERS, MCOVERS)
        ]
    elif study_name == 'const_mass':
        def make_params(elayers, vcov, mcov):
            return ExpParam({
                'MeshName': MESH_BASE_NAME, 'clscale': CLSCALE,
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov,
                'mcov': mcov,
                'psub': PSUB,
                'dt': DT, 'tf': TF,
                'ModifyEffect': 'const_mass'
            })

        paramss = [
            make_params(*args) for args in it.product(EMODS, VCOVERS, MCOVERS)
        ]
    elif study_name == 'const_mass_pregap':
        def make_params(elayers, vcov, mcov):
            return ExpParam({
                'MeshName': MESH_BASE_NAME, 'clscale': CLSCALE,
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov,
                'mcov': mcov,
                'psub': PSUB,
                'dt': DT, 'tf': TF,
                'ModifyEffect': 'const_mass_pregap'
            })

        paramss = [
            make_params(*args) for args in it.product(EMODS, VCOVERS, MCOVERS)
        ]
    else:
        raise ValueError(f"Unknown `--study-name` {study_name}")

    return paramss


## Main functions for running/postprocessing simulations
def run(
        params: dict,
        out_dir: str
    ):
    """
    Run the transient simulation
    """
    # Convert `params` from a dict to the parameters object
    # This is needed because you can't call `run` in parallel if `params`
    # is not pickleable (`ExpParams` instances can't be pickled)
    params = ExpParam(params)
    model = setup_model(params)
    state0, controls, prop = setup_state_control_props(params, model)

    dt = params['dt']
    tf = params['tf']
    times = dt*np.arange(0, int(round(tf/dt))+1)

    out_path = f'{out_dir}/{params.to_str()}.h5'
    if not path.isfile(out_path):
        with sf.StateFile(model, out_path, mode='a') as f:
            forward.integrate(
                model, f, state0, controls, prop, times, use_tqdm=True
            )
    else:
        print(f"Skipping {out_path} because the file already exists")

    return out_path

def postprocess(
        out_fpath: str, in_fpaths: List[str],
        overwrite_results: Optional[List[str]]=None,
        num_proc: int=1
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
                in_fpath, get_model, get_result_to_proc,
                num_proc=num_proc, overwrite_results=overwrite_results
            )

def get_result_to_proc(model: trabase.BaseTransientModel):
    """Return the mapping of results to post-processing functions"""
    proc_gw = slsig.MeanGlottalWidth(model)

    cell_label_to_id = model.solid.residual.mesh_function_label_to_value('cell')
    facet_label_to_id = model.solid.residual.mesh_function_label_to_value('facet')
    dx = model.solid.residual.measure('dx')
    # dx_cover = (
    #     dx(int(cell_label_to_id['medial']))
    #     + dx(int(cell_label_to_id['inferior']))
    #     + dx(int(cell_label_to_id['superior']))
    # )
    dx_cover = dx(int(cell_label_to_id['cover']))
    # dx_medial = dx(int(cell_label_to_id['medial']))
    ds_medial = model.solid.residual.measure('ds')(int(facet_label_to_id['pressure']))
    proc_visc_rate = slsig.ViscousDissipationRate(model, dx=dx_cover)

    ## Project field variables onto a DG0 space
    fspace_dg0 = model.solid.residual.form['coeff.fsi.p1'].function_space()
    proc_hydro_field = slsig.StressHydrostaticField(model)
    proc_vm_field = slsig.StressVonMisesField(model)
    proc_visc_diss_rate_field = slsig.ViscousDissipationField(
        model, dx=dx, fspace=fspace_dg0
    )

    proc_contact_area_density_field = slsig.ContactAreaDensityField(
        model, dx=ds_medial, fspace=fspace_dg0
    )
    proc_contact_pressure_field = slsig.ContactPressureField(
        model, dx=ds_medial, fspace=fspace_dg0
    )

    ## Project field variables onto a CG1 space
    proc_ymom_field = slsig.YMomentum(
        model, dx=dx, fspace=model.solid.residual.form['coeff.fsi.p1'].function_space()
    )

    ## Compute spatial statistics of field variables
    def make_carea_field_stats():
        return slsig.FieldStats(
            proc_contact_area_density_field
        )
    def make_cpressure_field_stats():
        return slsig.FieldStats(
            proc_contact_pressure_field
        )
    def make_wvisc_field_stats(dx=dx):
        return slsig.FieldStats(
            slsig.ViscousDissipationField(model, dx=dx, fspace=fspace_dg0)
        )
    def make_svm_field_stats(dx=dx):
        return slsig.FieldStats(
            slsig.StressVonMisesField(model, dx=dx, fspace=fspace_dg0)
        )

    def make_ymom_field_stats():
        return slsig.FieldStats(proc_ymom_field)

    def proc_time(f):
        return f.get_times()

    def proc_q(f):
        qs = [
            np.sum([f.get_state(ii)[f'fluid{n}.q'][0] for n in range(len(f.model.fluids))])
            for ii in range(f.size)
        ]
        return np.array(qs)

    signal_to_proc = {
        'q': proc_q,
        'gw': TimeSeries(proc_gw),
        'time': proc_time,

        'signal_savg_viscous_rate': TimeSeries(proc_visc_rate),

        'field_tavg_viscous_rate': lambda f: TimeSeriesStats(proc_visc_diss_rate_field).mean(f, range(f.size//2, f.size)),
        'field_tavg_vm': lambda f: TimeSeriesStats(proc_vm_field).mean(f, range(f.size//2, f.size)),
        'field_tavg_hydrostatic': lambda f: TimeSeriesStats(proc_hydro_field).mean(f, range(f.size//2, f.size)),
        'field_tavg_pc': lambda f: TimeSeriesStats(proc_contact_pressure_field).mean(f, range(f.size//2, f.size)),
        'field_tini_hydrostatic': lambda f: proc_hydro_field(f.get_state(0), f.get_control(0), f.get_prop()),
        'field_tini_vm': lambda f: proc_vm_field(f.get_state(0), f.get_control(0), f.get_prop()),

        'signal_spatial_stats_con_p': TimeSeries(make_cpressure_field_stats()),
        'signal_spatial_stats_con_a': TimeSeries(make_carea_field_stats()),
        'signal_spatial_stats_viscous': TimeSeries(make_wvisc_field_stats(dx_cover)),
        # 'signal_spatial_stats_viscous_medial': TimeSeries(make_wvisc_field_stats(dx_medial)),
        'signal_spatial_stats_vm': TimeSeries(make_svm_field_stats(dx_cover)),
        # 'signal_spatial_stats_vm_medial': TimeSeries(make_svm_field_stats(dx_medial)),
        'signal_spatial_state_ymom': TimeSeries(make_ymom_field_stats())
    }
    return signal_to_proc

def get_model(in_fpath: str) -> Model:
    """Return the model"""
    in_fname = path.splitext(path.split(in_fpath)[-1])[0]
    params = ExpParam(in_fname)
    return setup_model(params)

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--study-name", type=str, default='none')
    parser.add_argument("--output-dir", type=str, default='out')
    parser.add_argument("--overwrite-results", type=str, action='extend', nargs='+')
    parser.add_argument("--default-dt", type=float, default=1.25e-5)
    parser.add_argument("--default-tf", type=float, default=0.5)
    clargs = parser.parse_args()

    TF = clargs.default_tf
    DT = clargs.default_dt

    postprocess = functools.partial(postprocess, overwrite_results=clargs.overwrite_results)

    # Pack up the emod arguments to a dict format
    _emods = np.array([[2.5, 5.0]]) * 1e3 * 10
    layer_labels = ['cover', 'body']
    EMODS = [
        {label: value for label, value in zip(layer_labels, layer_values)}
        for layer_values in _emods
    ]

    ## Run and postprocess simulations
    out_dir = clargs.output_dir
    paramss = make_exp_params(clargs.study_name)
    paramss_dict = [params.data for params in paramss]
    if clargs.num_proc > 1:
        _run = functools.partial(run, out_dir=out_dir)
        with mp.Pool(processes=clargs.num_proc) as pool:
            print(f"Pool running with {clargs.num_proc:d} processors")
            in_fpaths = pool.map(_run, paramss_dict, chunksize=1)
    else:
        in_fpaths = [run(params, out_dir) for params in paramss_dict]

    out_fpath = f'{out_dir}/postprocess.h5'
    postprocess(out_fpath, in_fpaths, num_proc=clargs.num_proc)
