# bright_cosmology

Multi-messenger gravitational-wave + electromagnetic cosmology pipeline:

1. **Mock data** (`code/`): generates a mock GW event catalog for a chosen cosmology, runs a
   GWFish Fisher-matrix analysis to get observed distance uncertainties, and produces
   luminosity-distance posteriors.
2. **Gaussian-process (GP) analysis** (`theory/`, `likelihood/`, `input_cobaya/`): reconstructs
   the dark-energy density $f_{\rm DE}(z)$ from those events, alone or with Pantheon+
   supernovae, with [Cobaya](https://cobaya.readthedocs.io).

Built on [GWFish](https://github.com/janosch314/GWFish), installed automatically as a
dependency. The `bright_cosmology` package ships the project's custom prior module and
a small patch applied to `GWFish.modules.fishermatrix` at import time (skips zero-SNR
signals, writes the event/population name as a string column in the output files).

**Full tutorial:** the notebook [`notebooks/mm_gp_analysis.ipynb`](notebooks/mm_gp_analysis.ipynb)
goes step by step through the whole analysis: mock catalog, GWFish, distance posteriors, the
likelihood and what you have to provide to it, MCMC, reconstruction of $f_{\rm DE}(z)$ and $H(z)$.

## Install

Requires Python >=3.9,<3.12 (GWFish's constraint). In a new environment:

```bash
git clone https://github.com/AndreaCozzumbo/bright_cosmology.git
cd bright_cosmology
pip install ".[class,mpi,notebook]"
```

The extras are optional: `class` is standard CLASS (`classy`, compiled during the
installation, so it needs a C compiler), `mpi` runs MCMC chains in parallel, `notebook` is
Jupyter. Plain $\Lambda\mathrm{CDM}$ and PEDE runs only need standard CLASS. For
the non-minimal modified-gravity configuration leave out `class` and install the `classy` of your
hi_class build: it is the only one that works.

If packages of your `~/.local` folder leak into the environment, `export PYTHONNOUSERSITE=1`.

## Mock GW data

```bash
python code/mm_gw.py                                # code/inputs.yaml
python code/mm_gw.py --config code/pede_ET.yaml     # any other config
```

> **Disclaimer.** The non-minimal modified-gravity configuration only works with the
> [hi_class](https://github.com/miguelzuma/hi_class_public) implementation: `classy` must be the
> Python wrapper of hi_class. It fails with standard CLASS.

Dark-energy models that CLASS does not have (PEDE, TDE, Kink, CPL) come from the scripts in
`code/external/external_w/`, which print $f_{\rm DE}(a)$. With `Mode: 'external'` the pipeline reads
that table and computes $H(z)$ and the distances; to add a model, add a script. No patched CLASS is needed.

It can be run from any folder; paths inside the config files are relative to the repo root.

| config | cosmology | detectors | needs |
|---|---|---|---|
| `code/inputs.yaml` (default), `code/test_lcdm.yaml` | $\Lambda\mathrm{CDM}$ | ET | CLASS |
| `code/pede_ET.yaml` | PEDE | ET | CLASS |
| `code/inputs_nonminimal.yaml` | non-minimally coupled modified gravity (template to fill in); also has commented examples for CPL, DESI, TDE, PEDE, Kink | ET+CE | **hi_class only** |

A run takes a few minutes. The result is
`code/output_mcmc/<cosmology>/<name>/<name>.pkl` with `<name>` =
`<detectors>_<cosmology>_<catalog>`: the $d_L$ probability density of each event, i.e. the
input of the GP likelihood. `code/output_data/` has the Fisher matrices.

`detectors/` has the detectors used here in the format of GWFish, with their noise curves (PSD): ET and
CE40 (Cosmic Explorer with 40 km arms, which is not in the default file of GWFish). To use them set
`DetectorPath : 'detectors/detectors.yaml'`.

## GP analysis

Run from the repo root. `cobaya_data/` ships the catalog made by the notebook (ET, PEDE, 34 events), so
this does not need the previous step:

```bash
# GW only
OMP_NUM_THREADS=1 mpirun -n 4 cobaya-run input_cobaya/ET-delta.yaml

# GW + Pantheon+: download the data, make a mock in the same cosmology, run
cobaya-install sn.pantheonplus -p packages
python code/sn_mock.py
OMP_NUM_THREADS=1 mpirun -n 4 cobaya-run input_cobaya/PantheonPlus-ET-delta.yaml
```

They reach $R-1<0.01$ in about 2 (GW) and 7-15 minutes (GW + SNe) on 4 cores. Medians are
stable between chains; for the 95% tails run longer (see the notebook). Keep `OMP_NUM_THREADS=1`: multi-threading
makes the likelihood ~3x slower. Chains are in `chains/`.

In the yaml files, `interpolation_method` selects the GW likelihood:

- `kde`: the probability densities of $d_L$, with (`peculiar_velocity: integrated`) or without
  (`peculiar_velocity: None`) peculiar velocities;
- `gaussian`: $\chi^2$ with the Fisher errors of GWFish on $d_L$. It needs `peculiar_velocity: None`,
  because these errors do not include the peculiar velocities.

Things to know:

- `provides: ['angular_diameter_distance']` in the theory block is what makes the supernovae
  see the GP expansion history. Without it and with `classy` in the yaml they would be
  compared with $\Lambda\mathrm{CDM}$.
- The helper functions of the notebook are in `code/gp_tools.py`.
- To redraw the curves from a chain use the `seed` column, not `int(1e5*random)`: `random` is
  rounded in the chain file.

## How to cite

If you use this code, please cite

- A. Cozzumbo, U. Dupletsa, R. Calderón, R. Murgia, G. Oganesyan, M. Branchesi, *Model-independent cosmology
  with joint observations of gravitational waves and γ-ray bursts*,
  [JCAP 05 (2025) 021](https://iopscience.iop.org/article/10.1088/1475-7516/2025/05/021),
  [arXiv:2411.02490](https://arxiv.org/abs/2411.02490)

The gravitational-wave results are produced with GWFish, please cite also

- U. Dupletsa et al., *GWFish: A simulation software to evaluate parameter-estimation capabilities of
  gravitational-wave detector networks*, Astronomy and Computing 42, 100671 (2022),
  [arXiv:2205.02499](https://arxiv.org/abs/2205.02499)
- U. Dupletsa et al., *Validating Prior-informed Fisher-matrix Analyses against GWTC Data*, Phys. Rev. D 111,
  024036 (2025), [arXiv:2404.16103](https://arxiv.org/abs/2404.16103), for the priors

and, for the sampler of the truncated Gaussians, [truncated-mvn-sampler](https://github.com/brunzema/truncated-mvn-sampler)
and Z. I. Botev, Journal of the Royal Statistical Society Series B 79, 125-148 (2016).

## Licence

MIT, see `LICENSE`. Code from other projects is listed in `THIRD_PARTY_NOTICES.md`.
