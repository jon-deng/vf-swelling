"""
Post-process results from simulations
"""

from typing import Mapping, Callable, Iterable
from numpy.typing import NDArray

import numpy as np
from scipy import signal
import dolfin as dfn

from femvf import statefile as sf
from femvf.models.transient import base as trabase, coupled
from femvf.postprocess.base import TimeSeries, TimeSeriesStats
from femvf.postprocess import solid as slsig

from vfsig import clinical, fftutils

dfn.set_log_level(50)

POISSONS_RATIO = 0.4

Model = coupled.BaseTransientFSIModel

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

def proc_time(f: sf.StateFile) -> NDArray:
    """
    Return the simulation time vector
    """
    return f.get_times()

def proc_field_time_statistic(
    model: Model,
    f: sf.StateFile,
    Measure: slsig.BaseFieldMeasure,
    dx: dfn.Measure,
    fspace: dfn.FunctionSpace,
    idxs: Iterable[int],
    statistic: str='mean'
) -> NDArray:
    """
    Return a single field variable from time-varying fields

    Parameters
    ----------
    model: Model
        The model used to simulate voice outputs
    f: sf.StateFile
        The model time history
    """
    state_measure = Measure(model, dx=dx, fspace=fspace)
    if statistic == 'mean':
        return TimeSeriesStats(state_measure).mean(f, idxs)
    elif statistic == 'max':
        return TimeSeriesStats(state_measure).max(f, idxs)
    elif statistic == 'min':
        return TimeSeriesStats(state_measure).max(f, idxs)
    else:
        raise ValueError(f"Unknown `time_statistic` {statistic}")

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

    proc_pos_strain_energy_rate = slsig.PositiveStrainEnergyRate(model, dx=dx, fspace=fspace_dg0)

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
        'time.savg_viscous_dissipation': TimeSeries(proc_visc_rate),
        'field.tavg_viscous_dissipation': lambda f: TimeSeriesStats(
            proc_visc_diss_rate_field
        ).mean(f, range(f.size // 2, f.size)),
        'field.tavg_pos_strain_energy_rate': lambda f: TimeSeriesStats(
            proc_pos_strain_energy_rate
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
