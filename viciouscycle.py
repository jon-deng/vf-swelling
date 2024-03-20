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
import numpy as np
import dolfin as dfn

from femvf.postprocess.base import TimeSeriesStats
from femvf.postprocess import solid as slsig
from femvf.models.transient import coupled
from femvf import forward
from blockarray import blockvec as bv

Model = coupled.BaseTransientFSIModel

def compensate(
        model: Model,
        targets: Any,
        prop0: bv.BlockVector, control0: bv.BlockVector
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
    """

def damage_rate(
        model: Model,
        prop: bv.BlockVector, control: bv.BlockVector
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