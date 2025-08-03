# README

This package accompanies two papers and contains the code to conduct the studies described there.

## Project structure

The project is organized across directories as follows:

- `src/experiment`: A python package containing utilities for setting up the two
- `mesh`: meshes used in the study
- `fig`, `out`: directories for output figures and results

The root folder contains scripts `main_swelling_effects.py`, `main_swelling_growth.py`, `fig.ipynb`
which are used to run/analyze the study.

`main_swelling_effects.py` runs parametric studies of vocal fold vibration for varying magnitudes of swelling.

`main_swelling_growth.py` runs a study on modelling swelling growth over long times due to healing and damage mechanisms.

## Installation

To run this package, you will have to install the packages:

- <https://github.com/jon-deng/vf-fem> or <https://github.com/UWFluidFlowPhysicsGroup/vf-fem> (contains different vocal fold model definitions)
  - Install the version with tag `vf-swelling`
  - To do this, checkout the appropriate commit using `git checkout vf-swelling`
- <https://github.com/jon-deng/block-array> (utilities for working with block matrices/vector)
- <https://github.com/jon-deng/nonlinear-equation> (utilities for solving nonlinear equations)
- <https://github.com/jon-denf/vf-exputils> (miscellaneous utilities for running the experiment)

You will also need some common python packages such as 'matplotlib' and 'jupyter'.

## Running the study

To run the parametric studies of swelling use a command like

```bash
python main_swelling_effects.py --study-name test
```

which will run the `test` study that contains a single simulation case.
Other commandline parameters are given in the script.
To run the main parametric study, use the arguments below.

```bash
python main_swelling_effects.py --study-name main_2D --postprocess
python main_swelling_effects.py --study-name main_3D --postprocess
```

To run the swelling growth simulations, use a command like

```bash
python main_swelling_growth.py --dt 5e-5 --nt 8192 --dv 0.05 --nstart 0 --nstop 5
```

The arguments `--dt 5e-5 --nt 8192` indicate that 'voicing' simulations use a timestep of 5e-5 seconds and a total of 8192 timesteps.
The arguments `--dv 0.05 --nstart 0 --nstop 5` control the timesteps for the growth of swelling.
`--dv 0.05` limits the growth time step to a cause a maximum swelling increase of 5%.
`--nstart 0 --nstop 5` sets the number of swelling growth steps.
