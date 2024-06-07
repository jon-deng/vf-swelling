"""
Run a sequence of vocal fold simulations with swelling
"""

from typing import List, Tuple, Mapping, Optional, Callable
from numpy.typing import NDArray

from os import path
import argparse as ap
import multiprocessing as mp
import functools
from typing import List, Mapping

import numpy as np
import dolfin as dfn
import h5py

from femvf import forward, static, statefile as sf, meshutils
from femvf.models.transient import solid, fluid, base as trabase, coupled
from femvf.models.dynamical import base as dynbase
from femvf.postprocess.base import TimeSeries, TimeSeriesStats
from femvf.postprocess import solid as slsig
from femvf.load import load_transient_fsi_model

from blockarray import blockvec as bv

from exputils import postprocutils, exputils

from cases import ExpParam, make_exp_params

dfn.set_log_level(50)

POISSONS_RATIO = 0.4

Model = coupled.BaseTransientFSIModel


def setup_mesh_name(param: ExpParam) -> str:
    """
    Return the name of the mesh
    """
    base_name = param['MeshName']
    ga = param['GA']
    clscale = param['clscale']
    dz = param['DZ']
    nz = param['NZ']
    return f'{base_name}--GA{ga:.2f}--DZ{dz:.2e}--NZ{nz:d}--CL{clscale:.2e}'


def setup_model(param: ExpParam) -> Model:
    """
    Return the model
    """
    mesh_path = f"mesh/{setup_mesh_name(param)}.msh"

    if param['DZ'] == 0.0:
        zs = None
    elif param['DZ'] > 0.0:
        zs = np.linspace(0, param['DZ'], param['NZ'] + 1)
    else:
        raise ValueError("Parameter 'DZ' must be >= 0")

    swell_model_key = param['SwellingModel']
    if param['SwellingModel'] == 'linear':
        SolidType = solid.SwellingKelvinVoigtWEpitheliumNoShape
    elif param['SwellingModel'] == 'power':
        SolidType = solid.SwellingPowerLawKelvinVoigtWEpitheliumNoShape
    else:
        raise ValueError("Parameter 'SwellingModel' must be 'linear' or 'power'")

    model = load_transient_fsi_model(
        mesh_path,
        None,
        SolidType=SolidType,
        FluidType=fluid.BernoulliAreaRatioSep,
        zs=zs,
    )
    return model


def setup_state_control_prop(
    param: ExpParam, model: Model, dv: float = 0.01
) -> Tuple[bv.BlockVector, bv.BlockVector, bv.BlockVector]:
    """
    Return a (state, controls, prop) tuple defining a transient run
    """
    ## Set 'basic' model properties
    # These properties don't include the glottal gap since you may/may not
    # want to modify the glottal gap based on the swelling level
    prop = setup_basic_prop(param, model)
    model.set_prop(prop)

    ## Set the initial state
    # The initial state is based on the post-swelling static configuration
    state0 = setup_ini_state(param, model, dv=dv)

    # Set the glottal gap based on the post-swelling static configuration
    ndim = model.solid.residual.mesh().topology().dim()
    if (
        param['ModifyEffect'] == 'const_pregap'
        or param['ModifyEffect'] == 'const_mass_pregap'
    ):
        # Using the `ndim` to space things ensures you get the y-coordinate
        # for both 2D and 3D meshes
        ymax = (model.solid.XREF + state0.sub['u'])[1::ndim].max()
    else:
        ymax = (model.solid.XREF)[1::ndim].max()
    ygap = 0.03  # 0.3 mm half-gap -> 0.6 mm glottal gap
    ycoll_offset = 1 / 10 * ygap

    prop['ycontact'] = ymax + ygap - ycoll_offset
    prop['ymid'] = ymax + ygap
    for n in range(len(model.fluids)):
        prop[f'fluid{n}.area_lb'] = 2 * ycoll_offset

    model.set_prop(prop)

    controls = setup_controls(param, model)
    return state0, controls, prop


def setup_basic_prop(param: ExpParam, model: Model) -> bv.BlockVector:
    """
    Set the properties vector
    """
    mesh = model.solid.residual.mesh()
    forms = model.solid.residual.form
    mf = model.solid.residual.mesh_function('cell')
    mf_label_to_value = model.solid.residual.mesh_function_label_to_value('cell')
    cellregion_to_sdof = meshutils.process_meshlabel_to_dofs(
        mesh, mf, mf_label_to_value, forms['coeff.prop.emod'].function_space().dofmap()
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
    if param['ModifyEffect'] == 'const_pregap' or param['ModifyEffect'] == '':
        # This one is controlled in the initial state
        modify_kwargs = {}
    elif (
        param['ModifyEffect'] == 'const_mass'
        or param['ModifyEffect'] == 'const_mass_pregap'
    ):
        modify_kwargs = {'modify_density': False}
    else:
        raise ValueError(f"Unkown 'ModifyEffect' parameter {param['ModifyEffect']}")

    prop = _set_swelling_prop(param, model, prop, cellregion_to_sdof, **modify_kwargs)

    ## Set VF layer properties
    emods = {'cover': param['Ecov'], 'body': param['Ebod']}
    prop = _set_layer_prop(prop, emods, cellregion_to_sdof)

    return prop


def setup_controls(param: ExpParam, model: Model) -> bv.BlockVector:
    """
    Set the controls
    """
    control = model.control.copy()
    control[:] = 0

    for n in range(len(model.fluids)):
        control[f'fluid{n}.psub'] = param['psub']
        control[f'fluid{n}.psup'] = 0.0

    return [control]


def setup_ini_state(param: ExpParam, model: Model, dv: float = 0.01) -> bv.BlockVector:
    """
    Set the initial state vector
    """
    state0 = model.state0.copy()
    state0[:] = 0.0
    model.solid.control[:] = 0.0

    prop = setup_basic_prop(param, model)
    model.set_prop(prop)

    vcov = param['vcov']
    nload = max(int(round((vcov - 1) / dv)), 1)

    static_state, solve_info = solve_static_swollen_config(
        model.solid, model.solid.control, model.solid.prop, nload
    )

    num_loading_steps = solve_info.get('num_loading_steps', None)
    if num_loading_steps is not None:
        final_solve_info = solve_info[f'LoadingStep{num_loading_steps}']
    else:
        final_solve_info = solve_info

    if final_solve_info['status'] != 0:
        raise RuntimeError(
            "Static state couldn't be solved with solver info: " f"{solve_info}"
        )

    state0[['u', 'v', 'a']] = static_state
    return state0


def _set_swelling_prop(
    param: ExpParam,
    model: Model,
    prop: bv.BlockVector,
    cellregion_to_sdof: Mapping[str, NDArray],
    modify_density=True,
    modify_geometry=True,
    out_dir='out',
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
        np.concatenate([cellregion_to_sdof[label] for label in ['cover']])
    )
    dofs_bod = cellregion_to_sdof['body']
    # dofs_sha = np.intersect1d(dofs_cov, dofs_bod)

    # prop['k_swelling'][0] = KSWELL_FACTOR * 10e3*10
    prop['m_swelling'][dofs_cov] = param['mcov']

    prop['v_swelling'][:] = 1.0
    if modify_geometry:
        prop['v_swelling'][dofs_cov] = param['vcov']
        prop['v_swelling'][dofs_bod] = 1.0

    prop['rho'][:] = RHO_VF

    if modify_density:
        _v = np.array(prop['v_swelling'][:])
        prop['rho'][:] = RHO_VF + (_v - 1) * RHO_SWELL

    ## TODO : Fix this ad-hoc thing to do the swelling based on damage
    if param['SwellingDistribution'] != 'uniform' and param['vcov'] != 1.0:
        param_unswollen = param.substitute({'vcov': 1.0})
        with h5py.File(f'out/postprocess.h5', mode='a') as f:
            # damage_key = 'field.tavg_viscous_rate'
            damage_key = param['SwellingDistribution']
            group_name = param_unswollen.to_str()
            dataset_name = f'{group_name}/{damage_key}'
            # Check if the post-processed damage measure exists;
            # if not, post-process the damage measure.
            if dataset_name not in f:
                state_fpath = path.join(out_dir, f'{group_name}.h5')
                # If the simulation hasn't been run, then run it first
                if not path.isfile(state_fpath):
                    run(param_unswollen, out_dir)

                with sf.StateFile(model, state_fpath, mode='r') as fstate:
                    postprocess = get_result_name_to_postprocess(model)[damage_key]
                    group = f.require_group(group_name)
                    group.create_dataset(damage_key, data=postprocess(fstate))
            _damage = f[dataset_name][:]

        residual = model.solid.residual
        damage = residual.form['coeff.prop.v_swelling'].copy()
        damage.vector()[:] = _damage
        v_swelling = damage.copy()

        cell_label_to_id = residual.mesh_function_label_to_value('cell')
        dx = residual.measure('dx')
        dx_cover = dx(int(cell_label_to_id['cover']))
        # Make the increase in volume (i.e. `v-1`) proportional to the damage
        # measure and scale the resulting swelling field so that the total
        # prescribed volume increase is `param['vcov']`
        original_vol = dfn.assemble(1 * dx_cover)
        swollen_vol_incr = dfn.assemble(damage * dx_cover)
        v_factor = (param['vcov'] * original_vol - original_vol) / swollen_vol_incr
        v_swelling.vector()[:] = 1 + v_factor * damage.vector()[:]
        # print(dfn.assemble(v_swell*dx_cover)/dfn.assemble(1*dx_cover))
        prop['v_swelling'][:] = v_swelling.vector()[:]
        prop['v_swelling'][dofs_bod] = 1.0

    return prop


def _set_layer_prop(
    prop: bv.BlockVector,
    emods: Mapping[str, float],
    cellregion_to_sdof: Mapping[str, NDArray],
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
        np.concatenate([cellregion_to_sdof[label] for label in ['cover']])
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
    nload: int = 1,
    static_state_0: Optional[bv.BlockVector] = None,
):
    """
    Solve for the static swollen configuration

    This uses `nload` loading steps of the swelling field
    """
    # First, try to directly solve for the static swollen config from
    # the supplied initial guess, `static_state_0`
    # If this doesn't work run incremental changes in the swelling field to find
    # the static swollen configuration
    if static_state_0 is not None:
        static_state_n, info = static.static_solid_configuration(
            model, control, prop, state=static_state_0
        )
        solve_success = info['status']
    else:
        solve_success = -1

    if solve_success != 0:
        static_state_n, info = solve_static_swollen_config_stepped(
            model, control, prop, nload=nload
        )
    return static_state_n, info


def solve_static_swollen_config_stepped(
    model: Model, control: bv.BlockVector, prop: bv.BlockVector, nload: int = 1
):
    """
    Solve for the static swollen configuration

    This uses `nload` loading steps of the swelling field
    """
    if isinstance(model, trabase.BaseTransientModel):
        static_state_n = model.state0.copy()
        static_state_n[:] = 0
        model.dt = 1.0
    elif isinstance(model, dynbase.BaseDynamicalModel):
        static_state_n = model.state.copy()
        static_state_n[:] = 0

    v_final = prop['v_swelling'][:].copy()
    dv = (v_final - 1.0) / nload

    prop_n = prop.copy()
    prop_n['v_swelling'][:] = 1.0

    info = {'num_loading_steps': nload}
    for n in range(nload + 1):
        prop_n['v_swelling'][:] = 1.0 + n * dv
        static_state_n, solve_info_n = static.static_solid_configuration(
            model, control, prop_n, state=static_state_n
        )
        info[f'LoadingStep{n}'] = solve_info_n
    return static_state_n, info


## Main functions for running/postprocessing simulations
def run(param: dict, out_dir: str):
    """
    Run the transient simulation
    """
    # We need to convert `param` from a dict to the `ExpParam` object
    # because you can't call `run` in parallel if `param`
    # is not pickleable (`ExpParam` instances can't be pickled)
    param = ExpParam(param)
    out_path = f'{out_dir}/{param.to_str()}.h5'
    if not path.isfile(out_path):
        model = setup_model(param)
        state0, controls, prop = setup_state_control_prop(param, model)
        # breakpoint()

        dt = param['dt']
        tf = param['tf']
        times = dt * np.arange(0, int(round(tf / dt)) + 1)

        with sf.StateFile(model, out_path, mode='a') as f:
            forward.integrate(model, f, state0, controls, prop, times, use_tqdm=True)
    else:
        print(f"Skipping {out_path} because the file already exists")

    return out_path


def proc_glottal_flow_rate(f: sf.StateFile) -> NDArray:
    """
    Return the glottal flow rate vector
    """
    num_fluid = len(f.model.fluids)

    # Compute `q` as a weighted average over all coronal cross-sections
    if num_fluid > 1:
        qs = np.array([f.get_state()[f'state/fluid{n}.q'][0] for n in range(num_fluid)])
    else:
        qs = f.file[f'state/fluid0.q']

    # Assign full weights to all coronal sections with neighbours and
    # half-weights to the anterior/posterior coronal sections
    weights = np.ones(qs.shape[-1:])
    weights[[0, -1]] = 0.5
    q = np.sum(qs * weights, axis=-1) / np.sum(weights)
    return np.array(q)


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
                get_model,
                get_result_name_to_postprocess,
                num_proc=num_proc,
                overwrite_results=overwrite_results,
            )


from femvf.vis import xdmfutils


def postprocess_xdmf(model, param: ExpParam, xdmf_path: str, overwrite: bool = False):
    """
    Write an XDMF file
    """
    xdfm_data_dir, xdmf_basename = path.split(xdmf_path)
    xdmf_data_basename = f'{path.splitext(xdmf_basename)[0]}.h5'
    xdmf_data_path = path.join(xdfm_data_dir, xdmf_data_basename)
    with (
        h5py.File(f'out/{param.to_str()}.h5', mode='r') as fstate,
        h5py.File(f'out/postprocess.h5', mode='r') as fpost,
        h5py.File(xdmf_data_path, mode='w') as fxdmf,
    ):
        # breakpoint()
        # Export mesh values
        export_labels = ['mesh/solid', 'time']
        labels = export_labels
        datasets = [fstate[label] for label in export_labels]
        formats = [None, None]

        mesh = model.solid.residual.mesh()
        function_space = dfn.VectorFunctionSpace(mesh, 'CG', 1)
        export_labels = ['state/u', 'state/v', 'state/a']
        labels += export_labels
        datasets += [fstate[label] for label in export_labels]
        formats += len(export_labels) * [function_space]

        function_space = dfn.FunctionSpace(mesh, 'CG', 1)
        export_labels = ['time.field.p']
        labels += export_labels
        datasets += [fpost[f'{param.to_str()}/{label}'] for label in export_labels]
        formats += len(export_labels) * [function_space]

        function_space = dfn.FunctionSpace(mesh, 'DG', 0)
        export_labels = [
            'field.tavg_viscous_rate',
            'field.tavg_strain_energy',
            'field.growth_rate',
        ]
        # Account for the missing 'field.growth_rate' key for some measures
        export_labels = [
            label for label in export_labels if f'{param.to_str()}/{label}' in fpost
        ]
        labels += export_labels
        datasets += [fpost[f'{param.to_str()}/{label}'] for label in export_labels]
        formats += len(export_labels) * [function_space]

        xdmfutils.export_mesh_values(datasets, formats, fxdmf, output_names=labels)

        # Annotate the mesh values with an XDMF file
        xdmf_DG0_labels = [
            'field.tavg_viscous_rate',
            'field.tavg_strain_energy',
            'field.growth_rate',
        ]
        xdmf_DG0_labels = [label for label in export_labels if label in fxdmf]
        xdmf_DG0_descrs = [
            (fxdmf[label], 'scalar', 'Cell') for label in xdmf_DG0_labels
        ]

        # print(f"Exporting case: {param.to_str()}")
        # print(f"Exporting post-processed labels: {xdmf_DG0_labels}")

        static_dataset_descrs = [
            (fxdmf['state/u'], 'vector', 'node'),
        ] + xdmf_DG0_descrs
        static_idxs = [(0, ...)] + len(xdmf_DG0_descrs) * [
            (slice(None),),
        ]

        temporal_dataset_descrs = [
            (fxdmf['state/u'], 'vector', 'node'),
            (fxdmf['state/v'], 'vector', 'node'),
            (fxdmf['state/a'], 'vector', 'node'),
            (fxdmf['time.field.p'], 'scalar', 'node'),
        ]
        temporal_idxs = len(temporal_dataset_descrs) * [(slice(None),)]
        # breakpoint()
        xdmfutils.write_xdmf(
            fxdmf['mesh/solid'],
            static_dataset_descrs,
            static_idxs,
            fxdmf['time'],
            temporal_dataset_descrs,
            temporal_idxs,
            xdmf_path,
        )


def get_result_name_to_postprocess(
    model: trabase.BaseTransientModel,
) -> Mapping[str, Callable[[sf.StateFile], np.ndarray]]:
    """
    Return a mapping of result names to post-processing functions

    Returns
    -------
    Mapping[str, Callable]
        A mapping from post-processed result names to functions

        The post-processed result names have a format where each axis of the
        data is given a name separated by dots. For example, the name
        'time.vertex.u' could indicate a 2-axis array of 'u' displacements where
        the first axis represents time while the second axis represents
        different vertices.
    """
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
    # fspace_dg0 = model.solid.residual.form['coeff.fsi.p1'].function_space()
    fspace_dg0 = model.solid.residual.form['coeff.prop.eta'].function_space()
    proc_hydro_field = slsig.StressHydrostaticField(model)
    proc_vm_field = slsig.StressVonMisesField(model)
    proc_visc_diss_rate_field = slsig.ViscousDissipationField(
        model, dx=dx, fspace=fspace_dg0
    )

    proc_strain_energy = slsig.StrainEnergy(model, dx=dx, fspace=fspace_dg0)

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
        return slsig.FieldStats(proc_contact_area_density_field)

    def make_cpressure_field_stats():
        return slsig.FieldStats(proc_contact_pressure_field)

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

    result_name_to_postprocess = {
        'time.t': proc_time,
        'time.q': proc_glottal_flow_rate,
        'time.gw': TimeSeries(proc_gw),
        'time.field.p': TimeSeries(slsig.FSIPressure(model)),
        'time.savg_viscous_rate': TimeSeries(proc_visc_rate),
        'field.tavg_viscous_rate': lambda f: TimeSeriesStats(
            proc_visc_diss_rate_field
        ).mean(f, range(f.size // 2, f.size)),
        'field.tavg_vm': lambda f: TimeSeriesStats(proc_vm_field).mean(
            f, range(f.size // 2, f.size)
        ),
        'field.tavg_hydrostatic': lambda f: TimeSeriesStats(proc_hydro_field).mean(
            f, range(f.size // 2, f.size)
        ),
        'field.tavg_pc': lambda f: TimeSeriesStats(proc_contact_pressure_field).mean(
            f, range(f.size // 2, f.size)
        ),
        'field.tavg_strain_energy': lambda f: TimeSeriesStats(proc_strain_energy).mean(
            f, range(f.size // 2, f.size)
        ),
        'field.tmax_strain_energy': lambda f: TimeSeriesStats(proc_strain_energy).max(
            f, range(f.size // 2, f.size)
        ),
        'field.tini_hydrostatic': lambda f: proc_hydro_field(
            f.get_state(0), f.get_control(0), f.get_prop()
        ),
        'field.tini_vm': lambda f: proc_vm_field(
            f.get_state(0), f.get_control(0), f.get_prop()
        ),
        'field.vswell': lambda f: np.array(f.get_prop().sub['v_swelling']),
        'time.spatial_stats_con_p': TimeSeries(make_cpressure_field_stats()),
        'time.spatial_stats_con_a': TimeSeries(make_carea_field_stats()),
        'time.spatial_stats_viscous': TimeSeries(make_wvisc_field_stats(dx_cover)),
        # 'time.spatial_stats_viscous_medial': TimeSeries(make_wvisc_field_stats(dx_medial)),
        'time.spatial_stats_vm': TimeSeries(make_svm_field_stats(dx_cover)),
        # 'time.spatial_stats_vm_medial': TimeSeries(make_svm_field_stats(dx_medial)),
        'time.spatial_state_ymom': TimeSeries(make_ymom_field_stats()),
    }
    return result_name_to_postprocess


def get_model(in_fpath: str) -> Model:
    """Return the model"""
    in_fname = path.splitext(path.split(in_fpath)[-1])[0]
    param = ExpParam(in_fname)
    return setup_model(param)


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--study-name", type=str, default='none')
    parser.add_argument("--output-dir", type=str, default='out')
    parser.add_argument("--overwrite-results", type=str, action='extend', nargs='+')
    # parser.add_argument("--default-dt", type=float, default=1.25e-5)
    # parser.add_argument("--default-tf", type=float, default=0.5)
    parser.add_argument("--run-sim", action='store_true', default=False)
    parser.add_argument("--postprocess", action='store_true', default=False)
    parser.add_argument("--export-xdmf", action='store_true', default=False)
    clargs = parser.parse_args()

    ## Run and postprocess simulations
    out_dir = clargs.output_dir
    params = make_exp_params(clargs.study_name)
    param_dicts = [param.data for param in params]
    if clargs.num_proc > 1:
        _run = functools.partial(run, out_dir=out_dir)
        with mp.Pool(processes=clargs.num_proc) as pool:
            print(f"Pool running with {clargs.num_proc:d} processors")
            in_fpaths = pool.map(_run, param_dicts, chunksize=1)
    else:
        in_fpaths = [run(params, out_dir) for params in param_dicts]

    if clargs.postprocess:
        postprocess = functools.partial(
            postprocess, overwrite_results=clargs.overwrite_results
        )

        out_fpath = f'{out_dir}/postprocess.h5'
        postprocess(out_fpath, in_fpaths, num_proc=clargs.num_proc)

    if clargs.export_xdmf:
        for param in params:
            in_fpath = f'{out_dir}/{param.to_str()}.h5'
            xdmf_path = (
                "vis"
                f"--vcov{param['vcov']:.4e}"
                f"--mcov{param['mcov']:.2e}"
                f"--psub{param['psub']:.2e}"
                f"--distribution{param['SwellingDistribution']:s}.xdmf"
            )
            # xdmf_path = 'temp.xdmf'

            model = setup_model(param)
            postprocess_xdmf(model, param, xdmf_path)
            # if not path.isfile(out_fpath):
            # else:
            #     print(f"Skipping XDMF export of existing file {out_fpath}")
