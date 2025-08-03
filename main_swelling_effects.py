"""
Run a sequence of vocal fold simulations with swelling
"""

import sys
from typing import List, Optional
from numpy.typing import NDArray

from os import path
import argparse as ap
import multiprocessing as mp
import functools

import numpy as np
import dolfin as dfn
import h5py

from femvf import forward, statefile as sf
from femvf.models.transient import coupled

from exputils import postprocutils

from cases import ExpParam, make_exp_params

sys.path.append('src')
from experiment.setup import setup_model, setup_state_control_prop
from experiment.post import get_result_name_to_postprocess

dfn.set_log_level(50)

POISSONS_RATIO = 0.4

Model = coupled.BaseTransientFSIModel

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
        model = setup_model(param, 'mesh')
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
        h5py.File(f'./out/{param.to_str()}.h5', mode='r') as fstate,
        h5py.File(f'./out/postprocess.h5', mode='r') as fpost,
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
            'field.tavg_viscous_dissipation',
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
            'field.tavg_viscous_dissipation',
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



def get_model(in_fpath: str) -> Model:
    """Return the model"""
    in_fname = path.splitext(path.split(in_fpath)[-1])[0]
    param = ExpParam(in_fname)
    return setup_model(param, 'mesh')


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

            model = setup_model(param, 'mesh')
            postprocess_xdmf(model, param, xdmf_path)
            # if not path.isfile(out_fpath):
            # else:
            #     print(f"Skipping XDMF export of existing file {out_fpath}")
