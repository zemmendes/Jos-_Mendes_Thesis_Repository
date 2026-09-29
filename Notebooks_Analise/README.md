# Notebooks_Analise

Evaluation of the predictive uncertainty of the MOSA reconstructions. For each omic, generation method and
setting, the notebooks compute sharpness, calibration, a proper scoring rule and the reconstruction error, and then
explore the results across omics, features, cell lines and tissues.

* **Methods:** `latent` (Latent Space Sampling, the saved sample cube) and `mcd` (Monte Carlo Dropout, 100 draws from
  the MC Dropout mean/variance files).
* **Settings:** `IS` (in-sample: every cell line with ground truth) and `OOD` (out-of-sample: the ~10% of cell lines
  withheld from the target omic when training the model).
* **Omics:** metabolomics, drug response, methylation, proteomics, transcriptomics, CRISPR-Cas9 (continuous) and
  copy number (categorical, classes -2..2).

## Contents

```
omics_config.py                          paths, omic lists, per-omic preprocessing, file names, options
omics_metrics.py                         data preparation and every metric (the only code that computes metrics)
omics_plots.py                           loading of the saved metrics and every plot/table helper
metrics_1_continuous.ipynb               OMIC, METHOD, SETTING -> Pearson, RMSE, PIW, ENCE, CRPS files
metrics_2_copynumber.ipynb               METHOD, SETTING -> entropy, ECE, log score, balanced accuracy files
exploration_1_overview.ipynb             all 7 omics: tables, distributions, grids, correlations, tissues
exploration_2_continuous_features.ipynb  one continuous omic in depth, incl. calibration diagrams
exploration_3_copynumber.ipynb           copy-number description and figures
requirements.txt                         Python packages
```

Only the `metrics_*` notebooks write metric files; the `exploration_*` notebooks only read them (`exploration_2` and
`exploration_3` also load the aligned data and samples, read-only, for the calibration diagrams).

## How to run

1. Install the packages in `requirements.txt` (Python 3.12).
2. Place the data folder (see below) at `data/` in the repository root, or point the `DISSERTACAO_DATA_ROOT`
   environment variable to it. `DISSERTACAO_OUTPUT_ROOT` redirects the written metric files.
3. Open a notebook from this folder and edit only its first code cell:

   ```python
   OMIC    = "metabolomics"   # metrics_1 / exploration_2 only
   METHOD  = "latent"         # "latent" (Latent Space Sampling) | "mcd" (MC Dropout)
   SETTING = "IS"             # "IS" (in-sample) | "OOD" (held-out cell lines); metrics notebooks only
   ```

4. Run the `metrics_*` notebooks for the combinations needed (or set `RUN_ALL = True` in their last cell), then the
   `exploration_*` notebooks.

## Data folder

| folder | content |
|---|---|
| `datasets/` | ground truth, `{omic}.csv` |
| `imputed_datasets/`, `cube_datasets/` | in-sample predictions and latent sample cubes |
| `MC_dropout_mean/`, `MC_dropout_var/` | in-sample MC Dropout mean and variance |
| `imputed_removed_rows/`, `Removed_rows/` | out-of-sample predictions, latent sample cubes and held-out cell lines |
| `MC_dropout_OOD/` | out-of-sample MC Dropout mean and variance |
| `SIDM_info/` | cell-line annotation (`model_list_20230505.csv`) and DepMap-to-Sanger ID map (`Model.csv`) |
| `Genes/` | essential and non-essential genes used to scale CRISPR-Cas9 gene effects |

The file names are set in `omics_config.input_paths`. If the two methods use different out-of-sample model runs,
`omics_config.OOD_FILE_SUFFIX` adds a suffix to the file names of one of them.

## Output files

Written under the data folder as `{folder}/{stem}{method}{setting}.csv`, with method = `''` (latent) or `_MCD` and
setting = `''` (IS) or `_OOD`. Each run also appends a summary row to `metrics_summary.csv`.

| metric | folder / stem |
|---|---|
| sharpness (PIW; entropy for copy number) | `sharpness/{omic}_sharpness` |
| calibration (ENCE; ECE for copy number) | `calibration/{omic}_calibration` |
| CRPS | `CRPS/{omic}_CRPS` |
| RMSE | `RMSE/{omic}_feature_rmse`, `RMSE/{omic}_sample_rmse` |
| Pearson | `Pearson_Corr/{omic}/{omic}_feature_pearson`, `..._sample_pearson` |
| copy-number log score | `CRPS/copy_number_logscore` |
| copy-number balanced accuracy | `RMSE/copynumber_feature_balanced_accuracy` |

## Method notes

* **Alignment.** Ground truth, predictions and samples are matched by cell-line ID and feature name. DepMap IDs
  (CRISPR-Cas9, transcriptomics) are mapped to Sanger IDs with `Model.csv`.
* **Standardisation.** Continuous omics are z-scored per feature with the mean and standard deviation of all
  ground-truth rows, in-sample and out-of-sample alike.
* **CRISPR-Cas9.** Gene effects are scaled per cell line so that the median non-essential gene is 0 and the median
  essential gene is -1.
* **MC Dropout samples.** 100 draws from Normal(mean, variance) per datapoint, float32, seeded with
  `cfg.RNG_SEED`. The MC Dropout point prediction is the MC Dropout mean (`cfg.MCD_POINT_ESTIMATE`).
* **Copy number.** Samples are rounded to the nearest class and clipped to -2..2. Sharpness is the Shannon entropy
  (in bits) of the sampled classes divided by log2(5), so it lies in [0, 1]. The log score clips class
  probabilities at 1e-15.
* **Exploration filter.** For the MC Dropout out-of-sample results of methylation and CRISPR-Cas9, features with
  ENCE > 500 (near-zero predicted variance) are left out of the exploration summaries; their number is printed.
* **Speed.** PIW uses `np.percentile` when a chunk has no missing values, CRPS uses the sorted-sample identity
  `sum |x_i - x_j| = 2 sum_k (2k - M - 1) x_(k)` instead of an M x M difference array, the copy-number entropy is
  vectorised, and latent cubes are memory-mapped so that only the needed rows and features are loaded.
