"""
This module describes all the simulation cases for the swelling problem
"""

from typing import List, Tuple, Mapping, Optional, Callable
import itertools as itls

import numpy as np

from exputils import exputils

## Defaults for 'nominal' parameter values
MESH_BASE_NAME = 'M5_BC'

CLSCALE = 0.5

PSUB = 300 * 10

VCOVERS = np.array([1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3])
MCOVERS = np.array([0.0, -0.4, -0.8, -1.2, -1.6])

ECOV = 2.5e4
EBOD = 5e4
layer_labels = ['cover', 'body']
EMODS = [{'cover': ECOV, 'body': EBOD}]

TF = 2*5e-5
DT = 5e-5

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
    'ModifyEffect': str,
    'SwellingDistribution': str,
    'SwellingModel': str
}

ExpParam = exputils.make_parameters(PARAM_SPEC)

def make_exp_params(study_name: str) -> List[ExpParam]:
    DEFAULT_PARAM_2D = ExpParam({
        'MeshName': MESH_BASE_NAME, 'clscale': CLSCALE,
        'GA': 3, 'DZ': 0.00, 'NZ': 1,
        'Ecov': ECOV, 'Ebod': EBOD,
        'vcov': 1.0, 'mcov': 0.0,
        'psub': PSUB,
        'dt': DT, 'tf': TF,
        'ModifyEffect': '',
        'SwellingDistribution': 'uniform',
        'SwellingModel': 'linear'
    })

    DEFAULT_PARAM_3D = ExpParam({
        'MeshName': MESH_BASE_NAME, 'clscale': 0.25,
        'GA': 3,
        'DZ': 1.5, 'NZ': 15,
        'Ecov': ECOV, 'Ebod': EBOD,
        'vcov': 1, 'mcov': 0.0,
        'psub': 600*10,
        'dt': 5e-5, 'tf': 0.5,
        'ModifyEffect': '',
        'SwellingDistribution': 'uniform',
        'SwellingModel': 'power'
    })

    DEFAULT_PARAM_COARSE_3D = ExpParam({
        'MeshName': MESH_BASE_NAME, 'clscale': 0.75,
        'GA': 3,
        'DZ': 1.5, 'NZ': 15,
        'Ecov': ECOV, 'Ebod': EBOD,
        'vcov': 1, 'mcov': 0.0,
        'psub': 400*10,
        'dt': 5e-5, 'tf': 0.50,
        'ModifyEffect': '',
        'SwellingDistribution': 'uniform',
        'SwellingModel': 'power'
    })
    if study_name == 'none':
        params = []
    elif study_name == 'test':
        vcovs = [1.0, 1.1, 1.2, 1.3]
        vcovs = [1.0]
        params = [
            DEFAULT_PARAM_3D.substitute({
                'MeshName': MESH_BASE_NAME, 'clscale': 1.0,
                'GA': 3,
                'DZ': 1.5, 'NZ': 15,
                'Ecov': ECOV, 'Ebod': EBOD,
                'vcov': vcov, 'mcov': 0.0,
                'psub': 600*10,
                'dt': 5e-5, 'tf': 5e-5*10
            })
            for vcov in vcovs
        ]
    elif study_name == 'independence_2D':
        def make_param(clscale, dt):
            return DEFAULT_PARAM_2D.substitute({
                'clscale': clscale,
                'Ecov': ECOV, 'Ebod': EBOD,
                'vcov': 1.3, 'mcov': -1.6,
                'psub': PSUB,
                'dt': dt, 'tf': 0.5
            })

        clscales = 1 * 2.0**np.arange(1, -2, -1)
        dts = 5e-5 * 2.0**np.arange(0, -5, -1)
        params = [
            make_param(clscale, dt) for clscale in clscales for dt in dts
        ]
    elif study_name == 'main_2D':
        def make_param(elayers, vcov, mcov):
            return DEFAULT_PARAM_2D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov
            })

        params = [
            make_param(*args) for args in itls.product(EMODS, VCOVERS, MCOVERS)
        ]
    elif study_name == 'main_coarse_2D':
        def make_param(elayers, vcov, mcov):
            return DEFAULT_PARAM_2D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov,
                'dt': 1e-4, 'tf': 0.5
            })

        vcovs = np.array([1.0, 1.1, 1.2, 1.3])
        mcovs = np.array([0.0, -0.8])

        params = [
            make_param(*args) for args in itls.product(EMODS, VCOVERS, MCOVERS)
        ]
    elif study_name == 'main_3D_setup':
        # This case is the setup for the unswollen 3D state
        def make_param(elayers, vcov, mcov, damage):
            return DEFAULT_PARAM_3D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov,
                'SwellingDistribution': damage
            })

        vcovs = np.array([1.0])
        mcovs = np.array([0.0, -0.8, -1.6])
        damage_measures = [
            'field.tavg_viscous_rate',
            # 'field.tavg_strain_energy'
        ]

        params = [
            make_param(*args)
            for args in itls.product(EMODS, vcovs, mcovs, damage_measures)
        ]
    elif study_name == 'main_3D':
        # This case is the setup for the unswollen 3D state
        def make_param(elayers, vcov, mcov, damage):
            return DEFAULT_PARAM_3D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov,
                'SwellingDistribution': damage
            })

        vcovs = np.array([1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3])
        mcovs = np.array([0.0, -0.8, -1.6])
        damage_measures = [
            'field.tavg_viscous_rate',
            # 'field.tavg_strain_energy'
        ]

        params = [
            make_param(*args)
            for args in itls.product(EMODS, vcovs, mcovs, damage_measures)
        ]
    elif study_name == 'independence_3D':
        # This case is the setup for the unswollen 3D state
        def make_param(clscale, nz, elayers, vcov, mcov, psub, damage):
            return DEFAULT_PARAM_COARSE_3D.substitute({
                'clscale': clscale, 'NZ': nz,
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'psub': psub,
                'vcov': vcov, 'mcov': mcov, 'SwellingDistribution': damage
            })

        vcovs = np.array([1.0])
        mcovs = np.array([0.0])
        psubs = np.array([600*10])

        # To double the number of elements, reduce the length scale by
        # scale the length scale by `2**(-1/3)`
        # (you want the element volume to be 2 times smaller)
        # Here, go from fewer to more elements in factors of 2
        elem_refinement_factor = 2
        cl_factors = elem_refinement_factor**((-1/3)*np.arange(-1, 4))
        clscales = np.round(0.75 * cl_factors, decimals=2)
        # If `clscale = 0.75` corresponds to 15 z mesh divisions, then the
        # below scales an appropriate number of mesh divisions for other
        # mesh length scales
        nzs = np.array(np.round(15*0.75/clscales, 0), dtype=int)
        damage_measures = [
            'field.tavg_viscous_rate',
            # 'field.tavg_strain_energy'
        ]

        params = [
            make_param(*args[0], *args[1:])
            for args in itls.product(
                zip(clscales, nzs),
                EMODS, vcovs, mcovs, psubs,
                damage_measures
            )
        ]
    elif study_name == 'main_coarse_3D_setup':
        # This case is the setup for the unswollen 3D state
        def make_param(elayers, vcov, mcov, psub, damage):
            return DEFAULT_PARAM_COARSE_3D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'psub': psub,
                'vcov': vcov, 'mcov': mcov, 'SwellingDistribution': damage
            })

        vcovs = np.array([1.0])
        mcovs = np.array([0.0, -0.8, -1.6])
        psubs = np.array([400*10, 410*10])
        damage_measures = [
            'field.tavg_viscous_rate',
            # 'field.tavg_strain_energy'
        ]

        params = [
            make_param(*args)
            for args in itls.product(EMODS, vcovs, mcovs, psubs, damage_measures)
        ]
    elif study_name == 'main_coarse_3D':
        # This case is the setup for the unswollen 3D state
        def make_param(elayers, vcov, mcov, damage):
            return DEFAULT_PARAM_COARSE_3D.substitute({
                'MeshName': MESH_BASE_NAME,
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov, 'SwellingDistribution': damage
            })

        vcovs = np.array([1.0, 1.025, 1.05, 1.075, 1.1])
        mcovs = np.array([0.0, -0.8])
        damage_measures = [
            'field.tavg_viscous_rate',
            # 'field.tavg_strain_energy'
        ]

        params = [
            make_param(*args)
            for args in itls.product(EMODS, vcovs, mcovs, damage_measures)
        ]
    elif study_name == 'main_3D_xdmf':
        # This case is the setup for the unswollen 3D state
        def make_param(elayers, vcov, mcov, damage):
            return DEFAULT_PARAM_3D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov,
                'dt': 1e-4, 'tf': 0.25, 'clscale': 0.5, 'NZ': 10,
                'SwellingDistribution': damage
            })

        vcovs = np.array([1.0, 1.1, 1.2, 1.3])
        mcovs = np.array([0.0, -0.8, -1.6])
        damage_measures = [
            'field.tavg_viscous_rate',
            'field.tavg_strain_energy'
        ]

        params = [
            make_param(*args)
            for args in itls.product(EMODS, vcovs, mcovs, damage_measures)
        ]
    elif study_name == 'const_pregap':
        def make_param(elayers, vcov, mcov):
            return DEFAULT_PARAM_2D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov,
                'ModifyEffect': 'const_pregap'
            })

        params = [
            make_param(*args) for args in itls.product(EMODS, VCOVERS, MCOVERS)
        ]
    elif study_name == 'const_mass':
        def make_param(elayers, vcov, mcov):
            return DEFAULT_PARAM_2D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov,
                'ModifyEffect': 'const_mass'
            })

        params = [
            make_param(*args) for args in itls.product(EMODS, VCOVERS, MCOVERS)
        ]
    elif study_name == 'const_mass_pregap':
        def make_param(elayers, vcov, mcov):
            return DEFAULT_PARAM_2D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov,
                'ModifyEffect': 'const_mass_pregap'
            })

        params = [
            make_param(*args) for args in itls.product(EMODS, VCOVERS, MCOVERS)
        ]
    else:
        raise ValueError(f"Unknown `--study-name` {study_name}")

    return params