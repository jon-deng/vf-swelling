"""
Post-process results from simulations
"""

from typing import Optional

from femvf import static
from femvf.models.transient import base as trabase, coupled
from femvf.models.dynamical import base as dynbase

from blockarray import blockvec as bv

Model = coupled.BaseTransientFSIModel

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
        print("Tried direct newton solve", info)
    else:
        solve_success = -1

    if solve_success != 0:
        static_state_n, info = solve_static_swollen_config_stepped(
            model, control, prop, nload=nload
        )
        solve_success = info[f'LoadingStep{nload}']['status']

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

