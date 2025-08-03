"""
Functions that setup experiment inputs from experiment parameters
"""

from typing import Tuple, Mapping
from numpy.typing import NDArray

from os import path

import numpy as np
import dolfin as dfn
import h5py

from femvf import statefile as sf, meshutils
from femvf.models.transient import solid, fluid, coupled
from femvf.load import load_transient_fsi_model

from blockarray import blockvec as bv

from cases import ExpParam

from .solve import solve_static_swollen_config

Model = coupled.BaseTransientFSIModel

POISSONS_RATIO = 0.4

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


def setup_model(param: ExpParam, mesh_dir: str) -> Model:
    """
    Return the model
    """
    mesh_path = f"{mesh_dir}/{setup_mesh_name(param)}.msh"

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
    out_dir='../out',
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
        with h5py.File(f'../out/postprocess.h5', mode='a') as f:
            # damage_key = 'field.tavg_viscous_dissipation'
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

