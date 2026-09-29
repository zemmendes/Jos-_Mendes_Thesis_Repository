# Uncertainty Estimation in Multi-Omic Synthetic Augmentation

This repository accompanies a Master's thesis on uncertainty estimation in the Multi-Omic Synthetic Augmentation
(MOSA) model, a variational autoencoder that integrates and reconstructs seven molecular and phenotypic data layers
of cancer cell lines. Two sources of predictive uncertainty are studied: Latent Space Sampling, for the aleatoric
uncertainty encoded in the variational latent space, and Monte Carlo Dropout, an approximate Bayesian method for the
epistemic uncertainty of the model parameters. The resulting predictive distributions are evaluated with metrics for
sharpness, calibration and proper scoring rules, and compared across omics and between in-sample and out-of-sample
reconstructions.

The original MOSA code (named PhenPred) is available at https://github.com/QuantitativeBiology/PhenPred and is
described in Cai, Z. et al. (2024), *Synthetic augmentation of cancer cell line multi-omic datasets using
unsupervised deep learning*, Nature Communications 15, 10390.

## `MOSA_code/`

The MOSA (PhenPred) code extended with the two uncertainty-estimation methods. A latent space sampler
(`PhenPred/vae/LatentSampler.py`) draws repeated samples from the learned latent posterior and decodes them into
per-omic sample cubes, and Monte Carlo Dropout runs repeated stochastic decoder passes for the final predictions,
saving their mean and variance. Both are off by default and are switched on through `hyperparameters.json`. See `MOSA_code/README.md` for installation and use.

## `Notebooks_Analysis/`

The evaluation pipeline. The modules `omics_config.py`, `omics_metrics.py` and `omics_plots.py` hold the
configuration, the metric computations and the plotting helpers. The `metrics_*` notebooks compute the metrics for
each omic, method and setting, and the `exploration_*` notebooks analyse the results. See `Notebooks_Analysis/README.md`
for how to run them.

## `Thesis_LaTex/`

The LaTeX source of the thesis.

## Data

The data (inputs and results) are not part of the repository, they are expected in a `data/` folder at the
repository root, or in the folder set by `DISSERTACAO_DATA_ROOT` (see `Notebooks_Analysis/README.md`).
