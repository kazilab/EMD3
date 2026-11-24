# EMD3 cell-state conversion and selection

This notebook reproduces Figure 5. It uses synthetic observations to test whether stem-like cell fractions and total cell counts can distinguish conversion from selective proliferation or death. A Boolean network examines how the starting state and regulatory perturbation affect stem-like and immortalization readouts. [Open the notebook](EMD3_simulation_walkthrough.ipynb).

## Run the notebook

From the EMD3 directory, use Python 3.13.9 for the tested environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c requirements-tested.txt
python -m jupyter lab EMD3_simulation_walkthrough.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate`. In Jupyter, select the environment’s Python kernel and choose **Run All Cells**. The notebook runs offline after installation.

The default `reference` run uses 120 synthetic datasets per mechanism and sampling schedule, 60 parameter-sensitivity draws and 32 asynchronous Boolean update sequences per starting state for basin estimates. Set `RUN_MODE = "quick"` for 40 datasets and no sensitivity ensemble.

## Results

The default run saves Figure 5 and its caption under the filename stem `emd3_figure5_state_conversion` in `outputs/reproduced/`. The same directory contains the numerical summary, model tables, settings and log. Quick mode writes to `outputs/quick/`. Choose a new `RUN_NAME` to keep an additional run.

The population rates and observation noise are assigned. The source studies inform the model structure and biological interpretation. [Data and source notes](data/THIRD_PARTY.md).

## Command line and tests

From the EMD3 directory, run the reference analysis and export its tables.

```bash
cd src
python -m emd3_simulation.run --outdir ../outputs/reproduced
python -m emd3_simulation.manuscript_tables --outdir ../outputs/reproduced
```

From the same `src` directory, run `python -m pytest emd3_simulation/tests -q` for the regression tests. Software is distributed under the [MIT license](LICENSE).
