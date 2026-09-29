"""Data preparation and metric computation for the reconstruction/uncertainty evaluation.

Used by the ``metrics_*`` notebooks (which save the results) and, read-only, by the
``exploration_2`` / ``exploration_3`` notebooks (calibration diagrams need the aligned data).

Pipeline for one (omic, method, setting):
    load ground truth -> load point predictions + sample cube -> keep common features
    -> keep rows (IS: rows with ground truth, OOD: held-out rows) -> standardise (continuous only)
    -> metrics.

PIW, CRPS and the copy-number entropy use fast, exact formulations (see the comments in each function).
Paths and options live in ``omics_config``.
"""
import datetime
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import entr
from sklearn.metrics import balanced_accuracy_score

import omics_config as cfg


# ===========================================================================
# Loading
# ===========================================================================
def depmap_to_sanger():
    """DepMap ModelID -> SangerModelID mapping (NaN when a model has no Sanger ID)."""
    model = pd.read_csv(cfg.MODEL_MAP_PATH)[["ModelID", "SangerModelID"]]
    return model.set_index("ModelID")["SangerModelID"].to_dict()


def get_essential_genes(path=None):
    return set(pd.read_csv(path or cfg.ESSENTIAL_GENES_PATH, sep="\t")["gene"])


def get_non_essential_genes(path=None):
    return set(pd.read_csv(path or cfg.NON_ESSENTIAL_GENES_PATH, sep="\t")["gene"])


def crispr_scale(df, essential=None, non_essential=None, metric=np.nanmedian):
    """Scale each sample (column of a genes x samples frame) so the median non-essential gene
    effect is 0 and the median essential gene effect is -1."""
    if essential is None:
        essential = get_essential_genes()
    if non_essential is None:
        non_essential = get_non_essential_genes()
    essential_metric = metric(df.reindex(essential), axis=0)
    non_essential_metric = metric(df.reindex(non_essential), axis=0)
    return df.subtract(non_essential_metric).divide(non_essential_metric - essential_metric)


def load_ground_truth(omic):
    """Raw ground truth as a samples x features DataFrame with Sanger (SIDM) sample IDs."""
    spec = cfg.OMIC_SPECS[omic]
    data = pd.read_csv(cfg.DATA_ROOT / "datasets" / f"{omic}.csv", index_col=0)
    if spec["transpose"]:
        data = data.T
    if spec["depmap_ids"]:
        mapping = depmap_to_sanger()
        data.index = data.index.map(lambda x: mapping.get(x, x))
        data = data.loc[data.index.notna()]  # models without a Sanger ID
    if spec["crispr_scaling"]:
        data.columns = data.columns.str.split(" ").str[0]  # "A1BG (1)" -> "A1BG"
        data = crispr_scale(data.T).T
    return data


def load_held_out_rows(omic, method="latent"):
    """Sanger IDs of the rows removed from `omic` before training the OOD model used for `method`."""
    rows = pd.read_csv(cfg.input_paths(omic, method, "OOD")["held_out_rows"]).iloc[:, 0].tolist()
    if cfg.OMIC_SPECS[omic]["depmap_ids"]:
        mapping = depmap_to_sanger()
        rows = [mapping.get(x, x) for x in rows]
        rows = [x for x in rows if pd.notna(x)]
    return rows


def load_imputed(omic, setting, method="latent"):
    """Point predictions (samples x features). OOD files get the column labels of the IS file."""
    paths = cfg.input_paths(omic, method, setting)
    imputed = pd.read_csv(paths["imputed"], index_col=0)
    file_columns = imputed.columns
    if setting == "OOD":
        ref_cols = pd.read_csv(paths["imputed_reference"], index_col=0, nrows=0).columns
        labels = cfg.OOD_COLUMN_LABELS.get((omic, method))
        if labels is not None:  # numbered columns: names from the training input (omics_config.OOD_COLUMN_LABELS)
            path, name_col = labels
            header = pd.read_csv(path, nrows=0).columns
            names = pd.read_csv(path, usecols=[0, header.get_loc(name_col)], index_col=0)[name_col]
            imputed.columns = file_columns = pd.Index([names.loc[int(c)] for c in imputed.columns])
            imputed = imputed.reindex(columns=ref_cols)
        elif imputed.columns.isin(ref_cols).all():
            imputed = imputed.reindex(columns=ref_cols)  # named columns (possibly fewer): matched by name
        elif imputed.shape[1] == len(ref_cols):
            imputed.columns = file_columns = ref_cols  # OOD files may have numeric column labels
        else:
            imputed = imputed.reindex(columns=ref_cols)  # different feature set: matched by name
    imputed.attrs["file_columns"] = file_columns  # column order of the prediction/cube files
    return imputed


def sample_mcd_cube(mean, var, n_samples=None, seed=None, chunk_size=250):
    """Draw `n_samples` Normal(mean, sqrt(var)) samples per datapoint -> (rows, features, n) float32.

    NaN/inf standard deviations become 0 (the sample equals the mean)."""
    n_samples = n_samples or cfg.N_ENSEMBLE
    means = np.asarray(mean, dtype=np.float32)
    stds = np.sqrt(np.asarray(var, dtype=np.float32))
    stds = np.nan_to_num(stds, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)

    n_rows, n_features = means.shape
    cube = np.empty((n_rows, n_features, n_samples), dtype=np.float32)
    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        z = rng.standard_normal((n_rows, end - start, n_samples), dtype=np.float32)
        z *= stds[:, start:end, None]
        z += means[:, start:end, None]
        cube[:, start:end, :] = z
    return cube


def _take_cube(cube, row_pos, col_pos, chunk_size=1000):
    """cube[row_pos][:, col_pos] without materialising the full cube (works on memmaps)."""
    out = np.empty((len(row_pos), len(col_pos), cube.shape[2]), dtype=cube.dtype)
    for start in range(0, len(col_pos), chunk_size):
        end = min(start + chunk_size, len(col_pos))
        out[:, start:end, :] = cube[np.ix_(row_pos, col_pos[start:end], np.arange(cube.shape[2]))]
    return out


def standardize(data, imputed, cube, stats_from, chunk_size=100):
    """Z-score data, imputed and cube (in place) with the per-feature mean/std of `stats_from`.

    All four must have the same feature order."""
    mean = stats_from.mean(axis=0)
    std = stats_from.std(axis=0)
    data = (data - mean) / std
    imputed = (imputed - mean) / std
    mean_arr = mean.to_numpy().reshape(1, -1, 1)
    std_arr = std.to_numpy().reshape(1, -1, 1)
    for start in range(0, cube.shape[0], chunk_size):
        end = min(start + chunk_size, cube.shape[0])
        cube[start:end] = (cube[start:end] - mean_arr) / std_arr
    return data, imputed, cube


def prepare(omic, method, setting):
    """Aligned (data, imputed, cube) for one omic/method/setting.

    data, imputed: samples x features DataFrames with identical index and columns.
    cube: (samples, features, n_samples) array in the same order.
    Continuous omics are standardised; copy-number samples are rounded to integer classes.
    """
    cfg._check(omic, method, setting)
    paths = cfg.input_paths(omic, method, setting)

    data = load_ground_truth(omic)
    imputed = load_imputed(omic, setting, method)
    file_columns = imputed.attrs["file_columns"]
    if method == "mcd":
        mcd_mean = pd.read_csv(paths["mcd_mean"], index_col=0)
        mcd_var = pd.read_csv(paths["mcd_var"], index_col=0)

    # Features present in both, in the order of the prediction files.
    cols = imputed.columns.intersection(data.columns)
    cols = cols.intersection(file_columns)  # drop features the prediction file does not have
    if cfg.DEBUG_N_FEATURES:
        cols = cols[: cfg.DEBUG_N_FEATURES]

    # Rows: IS -> every sample with ground truth; OOD -> the held-out samples.
    if setting == "OOD":
        rows = imputed.index.intersection(load_held_out_rows(omic, method))
    else:
        rows = imputed.index.intersection(data.index)
    if method == "mcd":
        missing = rows.difference(mcd_mean.index)
        if len(missing):
            print(f"{len(missing)} selected rows have no MC Dropout prediction and are dropped: {list(missing)}")
            rows = rows.intersection(mcd_mean.index)

    # Where the selected rows/features sit in the cube, matched by the labels of the prediction file.
    row_pos = imputed.index.get_indexer(rows)
    file_index = imputed.index
    col_pos = file_columns.get_indexer(cols)

    stats_from = data.loc[:, cols]  # z-scores use every ground-truth row, for IS and OOD alike
    data = data.loc[rows, cols]
    imputed = imputed.loc[rows, cols]
    if len(data) != len(imputed):
        raise ValueError(f"{omic}: duplicated sample IDs in the ground truth ({len(data)} vs {len(imputed)} rows)")

    # Samples cube, restricted to the same rows/features.
    if method == "latent":
        cube_file = np.load(paths["cube"], mmap_mode="r")
        if cube_file.shape[:2] != (len(file_index), len(file_columns)):
            raise ValueError(
                f"{omic} {setting}: the sample cube {paths['cube'].name} has shape {cube_file.shape[:2]} but the "
                f"prediction file {paths['imputed'].name} has {len(file_index)} rows x {len(file_columns)} features; "
                "the cube and the predictions must come from the same model run."
            )
        cube = _take_cube(cube_file, row_pos, col_pos)
    else:
        mcd_rows = mcd_mean.index.get_indexer(rows)
        mean_sel = mcd_mean.to_numpy(dtype=np.float32)[np.ix_(mcd_rows, col_pos)]
        var_sel = mcd_var.to_numpy(dtype=np.float32)[np.ix_(mcd_rows, col_pos)]
        cube = sample_mcd_cube(mean_sel, var_sel, seed=cfg.RNG_SEED)
        if cfg.MCD_POINT_ESTIMATE == "mc_mean":
            point = mean_sel.astype(float)
            if omic == "copynumber":  # a class prediction is needed for balanced accuracy
                point = np.clip(np.round(point), min(cfg.CN_CLASSES), max(cfg.CN_CLASSES))
            imputed = pd.DataFrame(point, index=imputed.index, columns=imputed.columns)

    if omic in cfg.CONTINUOUS:
        data, imputed, cube = standardize(data, imputed, cube, stats_from)
    else:  # copy number: samples rounded to the nearest class in -2..2
        np.round(cube, 0, out=cube)
        np.clip(cube, min(cfg.CN_CLASSES), max(cfg.CN_CLASSES), out=cube)
    return data, imputed, cube


load_prepared = prepare  # read-only use from the exploration notebooks


# ===========================================================================
# Continuous metrics
# ===========================================================================
def pearson(data, imputed):
    """Pearson correlation between truth and prediction per feature and per observation."""
    common_idx = data.index.intersection(imputed.index)
    common_cols = data.columns.intersection(imputed.columns)
    per_feature = data.loc[common_idx, common_cols].corrwith(imputed.loc[common_idx, common_cols], axis=0, method="pearson")
    per_obs = data.loc[common_idx, common_cols].corrwith(imputed.loc[common_idx, common_cols], axis=1, method="pearson")
    return per_feature.rename("pearson_correlation"), per_obs.rename("pearson_correlation_per_obs")


def rmse(data, imputed):
    """RMSE per feature, per sample and overall, ignoring positions without ground truth."""
    mask = data.notna()
    sq_err = ((data - imputed) ** 2).where(mask)

    counts_feat = mask.sum(axis=0)
    sse_feat = sq_err.sum(axis=0)
    feature_rmse = np.sqrt(sse_feat / counts_feat.replace(0, np.nan))

    counts_row = mask.sum(axis=1)
    sample_rmse = np.sqrt(sq_err.sum(axis=1) / counts_row.replace(0, np.nan))

    total_count = counts_feat.sum()
    overall = np.sqrt(sse_feat.sum() / total_count) if total_count > 0 else np.nan
    return feature_rmse.rename("feature_rmse"), sample_rmse.rename("sample_rmse"), overall


def piw(cube, index, columns, n_chunks=10):
    """95% prediction-interval width (97.5th - 2.5th percentile of the samples) per datapoint."""
    widths = []
    for chunk in np.array_split(cube, n_chunks):
        # np.percentile equals np.nanpercentile when there are no NaNs and is much faster.
        pct = np.nanpercentile if np.isnan(chunk).any() else np.percentile
        lo, hi = pct(chunk, [2.5, 97.5], axis=2)
        widths.append(hi - lo)
    return pd.DataFrame(np.vstack(widths), index=index, columns=columns)


def ence(y_true, y_pred_mean, y_pred_std, n_bins=10, strategy="quantile", min_bin_count=1):
    """Expected Normalized Calibration Error for regression.

    ENCE = (1 / B_eff) * sum_b |RMSE_b - RMV_b| / RMV_b over bins of predicted std
    ("quantile": equal-count bins, "equal_width": equal-width bins); bins with fewer than
    `min_bin_count` samples or zero RMV are skipped."""
    y_true = np.asarray(y_true).ravel()
    y_pred_mean = np.asarray(y_pred_mean).ravel()
    y_pred_std = np.asarray(y_pred_std).ravel()
    assert y_true.shape == y_pred_mean.shape == y_pred_std.shape
    if y_true.shape[0] == 0:
        raise ValueError("Empty inputs.")
    if np.any(y_pred_std < 0):
        raise ValueError("Predicted standard deviations must be non-negative.")

    sigma = y_pred_std
    if strategy == "equal_width":
        s_min, s_max = sigma.min(), sigma.max()
        if np.isclose(s_min, s_max):
            bin_edges = np.array([s_min, s_max + 1e-12])
            n_bins_eff = 1
        else:
            bin_edges = np.linspace(s_min, s_max, n_bins + 1)
            n_bins_eff = n_bins
    else:
        bin_edges = np.unique(np.percentile(sigma, np.linspace(0.0, 100.0, n_bins + 1)))
        n_bins_eff = len(bin_edges) - 1
        if n_bins_eff == 0:
            bin_edges = np.array([sigma.min(), sigma.max() + 1e-12])
            n_bins_eff = 1

    bin_indices = np.digitize(sigma, bin_edges[1:-1], right=False)  # bins are [edge_i, edge_{i+1})
    bin_terms = []
    for b in range(n_bins_eff):
        mask = bin_indices == b
        if mask.sum() < min_bin_count:
            continue
        rmse_b = np.sqrt(np.mean((y_true[mask] - y_pred_mean[mask]) ** 2))
        rmv_b = np.sqrt(np.mean(sigma[mask] ** 2))  # root mean variance
        if rmv_b == 0.0:
            continue
        bin_terms.append(np.abs(rmse_b - rmv_b) / rmv_b)
    return float(np.mean(bin_terms)) if bin_terms else np.nan


def crps_ensemble(y_true, y_pred_samples):
    """Mean empirical CRPS over observations (Matheson & Winkler, 1976):
    CRPS_i = mean_m |x_im - y_i| - 1/2 * mean_{m,m'} |x_im - x_im'|."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred_samples = np.asarray(y_pred_samples, dtype=float)
    if y_pred_samples.ndim == 1:
        y_pred_samples = y_pred_samples[:, None]
    if y_pred_samples.ndim != 2:
        raise ValueError("y_pred_samples must be 2D: (n_obs, n_samples)")
    if y_pred_samples.shape[0] != y_true.shape[0]:
        raise ValueError(f"Mismatched n_obs: y_true has {y_true.shape[0]}, but y_pred_samples has {y_pred_samples.shape[0]}")
    if y_pred_samples.shape[0] == 0 or y_pred_samples.shape[1] == 0:
        return np.nan

    abs_to_truth = np.abs(y_pred_samples - y_true[:, None]).mean(axis=1)
    # Mean pairwise |x_m - x_m'| via the sorted-sample identity
    #   sum_{m,m'} |x_m - x_m'| = 2 * sum_k (2k - M - 1) x_(k),
    # exact and O(M log M) instead of the O(M^2) pairwise difference array.
    m = y_pred_samples.shape[1]
    weights = 2 * np.arange(1, m + 1) - m - 1
    pairwise_abs = 2.0 * (np.sort(y_pred_samples, axis=1) @ weights) / m**2
    return float(np.mean(abs_to_truth - 0.5 * pairwise_abs))


def _feature_wise(data, imputed, cube, score, name, n_jobs=-1, backend="threading"):
    """Apply score(y_true, samples) to every feature, using only rows with ground truth.

    The threading backend keeps a single shared copy of the cube in memory."""
    cols = list(data.columns)
    if list(imputed.columns) != cols:
        raise ValueError("data and imputed_data columns must match in name and order")
    if cube.shape[1] != len(cols):
        raise ValueError(f"data_cube axis 1 ({cube.shape[1]}) must equal len(data.columns) ({len(cols)})")
    index_to_pos = {idx: i for i, idx in enumerate(imputed.index)}
    idx_arr = data.index.to_numpy()
    data_mat = data.to_numpy(dtype=float, copy=False)

    def _work(j):
        y = data_mat[:, j]
        valid = ~np.isnan(y)
        if not valid.any():
            return cols[j], np.nan
        row_pos = np.array([index_to_pos[ix] for ix in idx_arr[valid]], dtype=np.intp)
        return cols[j], score(y[valid], cube[row_pos, j, :])

    pairs = Parallel(n_jobs=n_jobs, backend=backend)(delayed(_work)(j) for j in range(len(cols)))
    return pd.DataFrame(pairs, columns=["feature", name]).set_index("feature")


def feature_wise_ence(data, imputed, cube, n_bins=10, strategy="equal_width", min_bin_count=1, **kw):
    def score(y, samples):
        return ence(y, samples.mean(axis=1), samples.std(axis=1, ddof=0), n_bins, strategy, min_bin_count)
    return _feature_wise(data, imputed, cube, score, "ence", **kw)


def feature_wise_crps(data, imputed, cube, **kw):
    return _feature_wise(data, imputed, cube, crps_ensemble, "crps", **kw)


def continuous_metrics(data, imputed, cube):
    """All continuous metrics for one prepared omic/method/setting."""
    feature_pearson, sample_pearson = pearson(data, imputed)
    feature_rmse, sample_rmse, overall_rmse = rmse(data, imputed)
    return {
        "feature_pearson": feature_pearson,
        "sample_pearson": sample_pearson,
        "feature_rmse": feature_rmse,
        "sample_rmse": sample_rmse,
        "overall_rmse": overall_rmse,
        "sharpness": piw(cube, imputed.index, imputed.columns),
        "calibration": feature_wise_ence(data, imputed, cube)["ence"],
        "crps": feature_wise_crps(data, imputed, cube)["crps"],
    }


# ===========================================================================
# Copy-number metrics
# ===========================================================================
def cn_class_probs(samples, classes=cfg.CN_CLASSES):
    """Fraction of samples in each class: (n_obs, n_samples) -> (n_obs, n_classes)."""
    return np.stack([np.mean(samples == c, axis=1) for c in classes], axis=1)


def cn_entropy(cube, index, columns):
    """Normalised Shannon entropy of the sampled classes per datapoint (the copy-number sharpness).

    Entropy in bits divided by log2(5), so 0 = all samples agree and 1 = uniform over the five classes."""
    values = np.unique(cube)
    if len(values) > 50:
        raise ValueError(f"Copy-number cube has {len(values)} distinct values; expected integer classes")
    n = cube.shape[2]
    h = np.zeros(cube.shape[:2])
    for v in values:
        h += entr((cube == v).sum(axis=2) / n)  # -p ln p, 0 for p = 0
    h /= np.log(2)
    h /= np.log2(len(cfg.CN_CLASSES))
    return pd.DataFrame(h, index=index, columns=columns)


def calculate_ece(true_labels, sample_matrix, n_bins=10, classes=cfg.CN_CLASSES):
    """Expected Calibration Error of the majority-class prediction; returns (ece, n_used).

    Confidence = probability of the most sampled class; bins are (edge_i, edge_{i+1}] with
    confidence 1 in the last bin. Rows without ground truth are skipped."""
    mask = ~pd.isna(true_labels)
    n_used = int(mask.sum())
    if n_used == 0:
        return np.nan, 0
    true_labels = np.asarray(true_labels)[mask]
    classes = np.asarray(classes)
    probs = cn_class_probs(sample_matrix[mask, :], classes)
    confidence = probs.max(axis=1)
    accuracy = (classes[np.argmax(probs, axis=1)] == true_labels).astype(float)

    bin_ids = np.clip(np.digitize(confidence, np.linspace(0.0, 1.0, n_bins + 1), right=True) - 1, 0, n_bins - 1)
    ece_value = 0.0
    for b in range(n_bins):
        in_bin = bin_ids == b
        if in_bin.sum():
            ece_value += (in_bin.sum() / len(true_labels)) * abs(accuracy[in_bin].mean() - confidence[in_bin].mean())
    return ece_value, n_used


def feature_wise_ece(data, cube, n_bins=10):
    """Per-feature ECE table (feature, ECE, n_observations_used, pct_missing), worst first."""
    rows = []
    for j, feature in enumerate(data.columns):
        true_labels = data[feature].values
        value, n_used = calculate_ece(true_labels, cube[:, j, :], n_bins=n_bins)
        rows.append({
            "feature": feature,
            "ECE": value,
            "n_observations_used": n_used,
            "pct_missing": (len(true_labels) - n_used) / len(true_labels) * 100,
        })
    return pd.DataFrame(rows).sort_values("ECE", ascending=False).reset_index(drop=True)


def logscore_matrix(data, cube, classes=cfg.CN_CLASSES, epsilon=1e-15):
    """-log P(true class) per datapoint (NaN where there is no ground truth).

    Class probabilities are clipped to `epsilon` and renormalised before the log."""
    classes = np.asarray(classes)
    out = np.full(data.shape, np.nan, dtype=float)
    for j in range(data.shape[1]):
        true_labels = data.iloc[:, j].values
        mask = ~pd.isna(true_labels)
        if not mask.any():
            continue
        probs = np.clip(cn_class_probs(cube[mask, j, :], classes), epsilon, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = true_labels[mask]
        valid = np.isin(labels, classes)
        if not valid.any():
            continue
        label_idx = np.array([np.where(classes == label)[0][0] for label in labels[valid]])
        out[np.flatnonzero(mask)[valid], j] = -np.log(probs[valid][np.arange(len(label_idx)), label_idx])
    return pd.DataFrame(out, index=data.index, columns=data.columns)


def balanced_accuracy(data, imputed):
    """Balanced accuracy of the point prediction per feature (NaN when no ground truth)."""
    gt = data.reindex(index=imputed.index, columns=imputed.columns)
    pred = imputed.reindex(index=gt.index, columns=gt.columns)
    scores = {}
    with warnings.catch_warnings():
        # predictions may contain classes absent from a feature's ground truth; that is expected
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        for feature in gt.columns:
            mask = gt[feature].notna() & pred[feature].notna()
            scores[feature] = balanced_accuracy_score(gt[feature][mask], pred[feature][mask]) if mask.sum() > 0 else np.nan
    return pd.Series(scores, name="balanced_accuracy").sort_index()


def copynumber_metrics(data, imputed, cube):
    ece_table = feature_wise_ece(data, cube)
    return {
        "sharpness": cn_entropy(cube, imputed.index, imputed.columns),
        "calibration": ece_table.set_index("feature")[["ECE"]],
        "ece_table": ece_table,
        "logscore": logscore_matrix(data, cube),
        "balanced_accuracy": balanced_accuracy(data, imputed),
    }


# ===========================================================================
# Saving and running
# ===========================================================================
SAVED_METRICS = {
    "continuous": ["feature_pearson", "sample_pearson", "feature_rmse", "sample_rmse", "sharpness", "calibration", "crps"],
    "copynumber": ["sharpness", "calibration", "logscore", "balanced_accuracy"],
}


def save_results(results, omic, method, setting):
    """Write each metric to its usual CSV (see omics_config.OUTPUT_FILES). Returns the paths."""
    if not cfg.SAVE_OUTPUTS:
        print("SAVE_OUTPUTS is False: nothing written.")
        return []
    if cfg.DEBUG_N_FEATURES:
        print("DEBUG_N_FEATURES is set: results are partial, nothing written.")
        return []
    kind = "copynumber" if omic == "copynumber" else "continuous"
    written = []
    for metric in SAVED_METRICS[kind]:
        path = cfg.output_path(metric, omic, method, setting)
        path.parent.mkdir(parents=True, exist_ok=True)
        results[metric].to_csv(path, index=True)
        written.append(path)
    print(f"Wrote {len(written)} files, e.g. {written[0]}")
    return written


def summarize(results, omic, method, setting):
    """One row: mean and variance over features of the feature-wise metrics."""
    sharp = results["sharpness"].mean(axis=0)
    if omic == "copynumber":
        cal, score = results["calibration"]["ECE"], results["logscore"].mean(axis=0)
        score_name = "logscore"
    else:
        cal, score = results["calibration"], results["crps"]
        score_name = "crps"
    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "omic": omic, "method": method, "setting": setting,
        "sharpness_mean": sharp.mean(), "sharpness_var": sharp.var(),
        "calibration_mean": cal.mean(), "calibration_var": cal.var(),
        f"{score_name}_mean": score.mean(), f"{score_name}_var": score.var(),
    }


def append_summary(row, path=None):
    """Append a summary row to OUTPUT_ROOT/metrics_summary.csv."""
    if not cfg.SAVE_OUTPUTS or cfg.DEBUG_N_FEATURES:
        return
    path = path or cfg.OUTPUT_ROOT / "metrics_summary.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def run(omic, method, setting):
    """Prepare, compute, save and summarise one omic/method/setting. Returns (results, summary)."""
    data, imputed, cube = prepare(omic, method, setting)
    print(f"{omic} | {method} | {setting}: data {data.shape}, cube {cube.shape}")
    if omic == "copynumber":
        results = copynumber_metrics(data, imputed, cube)
    else:
        results = continuous_metrics(data, imputed, cube)
    del data, imputed, cube
    save_results(results, omic, method, setting)
    row = summarize(results, omic, method, setting)
    append_summary(row)
    return results, row
