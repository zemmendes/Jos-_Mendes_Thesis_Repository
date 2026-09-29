"""Loading of the saved metric CSVs and every plotting/summary helper used by the
``exploration_*`` notebooks. Nothing here writes metric files.

Metrics are held in a nested dict ``m[omic][setting][key]`` where key is one of
    sharpness   datapoint-level DataFrame (PIW, or entropy for copy number)
    calibration feature Series (ENCE, or ECE for copy number)
    scoring     feature Series (CRPS, or mean log score for copy number)
    error       feature Series (RMSE, or balanced accuracy for copy number)
    pearson     feature Series (continuous omics only)
Feature-level series for plotting come from ``feature_metric(m, omic, setting, key)``, which also
serves the setting-independent keys "variance", "missing" and "diversity".
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.plotting import parallel_coordinates
from scipy.cluster.hierarchy import linkage
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from scipy.stats import t as t_dist

import omics_config as cfg
import omics_metrics as om

SETTING_LABELS = {"IS": "In Sample Data", "OOD": "Out of Sample Data"}
PALETTE = {"IS": "tab:blue", "OOD": "tab:orange"}
BLUES_NO_WHITE = mpl.colors.LinearSegmentedColormap.from_list("blues_no_white", plt.cm.GnBu(np.linspace(0.3, 1, 256)))

# Axis labels per metric key: (continuous, copy number)
LABELS = {
    "sharpness": ("Feature PIW", "Mean Feature Reconstruction Entropy"),
    "calibration": ("Feature ENCE", "Feature ECE"),
    "scoring": ("Feature CRPS", "Feature Mean Log Score"),
    "error": ("Feature RMSE", "Feature Balanced Accuracy"),
    "pearson": ("Feature Pearson Correlation", None),
    "variance": ("Original Feature's Variance", "Original Feature's Variance"),
    "missing": ("Percentage of missing values", None),
    "diversity": (None, "Feature Diversity"),
}


def label(key, omic):
    return LABELS[key][omic == "copynumber"]


# ===========================================================================
# Loading
# ===========================================================================
_MISSING_FILES = []


def _read(path):
    if not path.exists():
        _MISSING_FILES.append(path)
        return None
    return pd.read_csv(path, index_col=0)


def _col(df, col):
    if df is None:
        return None
    return df.iloc[:, 0].rename(col) if col is None or col not in df.columns else df[col]


def load_metrics(method="latent", omics=None, ood_ence_outliers=None):
    """Read all saved metric CSVs for `method` (both settings) into m[omic][setting][key].

    ood_ence_outliers: {omic: threshold} -> drop OOD features whose ENCE exceeds threshold
    (default: methylation and crisprcas9, ENCE > 500; features with a near-zero PIW have a very
    large ENCE). Counts of removed features are printed."""
    omics = omics or cfg.OMICS
    ood_ence_outliers = {"methylation": 500, "crisprcas9": 500} if ood_ence_outliers is None else ood_ence_outliers
    m = {}
    for omic in omics:
        m[omic] = {}
        for setting in cfg.SETTINGS:
            p = lambda metric: cfg.output_path(metric, omic, method, setting)
            d = {"sharpness": _read(p("sharpness"))}
            if omic == "copynumber":
                d["calibration"] = _col(_read(p("calibration")), "ECE")
                logscore = _read(p("logscore"))
                d["scoring"] = None if logscore is None else logscore.mean().rename("LogScore")
                bal = _read(p("balanced_accuracy"))  # first column: balanced accuracy
                d["error"] = None if bal is None else bal.iloc[:, 0].rename("balanced_accuracy")
            else:
                d["calibration"] = _col(_read(p("calibration")), "ence")
                d["scoring"] = _col(_read(p("crps")), "crps")
                d["error"] = _col(_read(p("feature_rmse")), "feature_rmse")
                d["pearson"] = _col(_read(p("feature_pearson")), "pearson_correlation")
            if setting == "OOD" and omic in ood_ence_outliers and d["calibration"] is not None:
                cal = d["calibration"]
                drop = cal[cal > ood_ence_outliers[omic]].index
                d["calibration"] = cal.drop(drop)
                print(f"Removed {len(drop)} outliers (ENCE > {ood_ence_outliers[omic]}) from {omic} OOD calibration")
            m[omic][setting] = d
    if _MISSING_FILES:
        print(f"{len(_MISSING_FILES)} metric files not found (run the metrics notebooks for {method!r}):")
        for path in _MISSING_FILES:
            print("  ", path)
        _MISSING_FILES.clear()
    return m


def load_raw_feature_info(omics=None):
    """Setting-independent feature info from the raw ground truth:
    variance[omic] (per feature), missing[omic] (% missing, OMICS_WITH_MISSING only),
    diversity (copy number normalised Shannon diversity)."""
    omics = omics or cfg.OMICS
    variance, missing, diversity = {}, {}, None
    for omic in omics:
        spec = cfg.OMIC_SPECS[omic]
        raw = pd.read_csv(cfg.DATA_ROOT / "datasets" / f"{omic}.csv", index_col=0)
        raw = raw.T if spec["transpose"] else raw  # samples x features
        if omic == "crisprcas9":
            raw.columns = raw.columns.astype(str).str.replace(r"\s*\(\d+\)$", "", regex=True)
        variance[omic] = raw.var(axis=0).rename("feature_variance")
        if omic in cfg.OMICS_WITH_MISSING:
            missing[omic] = (raw.isna().sum() / raw.shape[0] * 100).rename("percent_missing")
        if omic == "copynumber":
            diversity = raw.apply(normalized_shannon_diversity, k=5).rename("normalized_shannon_diversity")
    return variance, missing, diversity


def load_tissue():
    """Sanger model ID -> tissue."""
    return pd.read_csv(cfg.TISSUE_INFO_PATH, index_col=0)["tissue"]


def normalized_shannon_diversity(series, k=5):
    """Shannon entropy of a feature's class distribution / log(k), in [0, 1] (k = number of classes)."""
    s = series.dropna()
    if s.empty:
        return np.nan
    if s.nunique() == 1:
        return 0.0
    p = s.value_counts(normalize=True)
    h = -(p * np.log(p + 1e-10)).sum()
    return h / np.log(k)


def feature_metric(m, omic, setting, key, info=None):
    """One feature-level Series. info = (variance, missing, diversity) from load_raw_feature_info."""
    if key in ("variance", "missing", "diversity"):
        variance, missing, diversity = info
        return {"variance": variance.get(omic), "missing": missing.get(omic),
                "diversity": diversity if omic == "copynumber" else None}[key]
    if setting not in m.get(omic, {}):
        return None
    value = m[omic][setting].get(key)
    if value is None:
        return None
    return value.mean() if key == "sharpness" else value


def feature_table(m, omic, setting, info=None, **columns):
    """DataFrame of feature-level metrics, e.g. feature_table(m, o, "IS", x="sharpness", y="calibration").
    Rows with any missing value are dropped."""
    parts = []
    for name, key in columns.items():
        s = feature_metric(m, omic, setting, key, info)
        if s is None:  # setting not available (or metric file missing)
            return pd.DataFrame(columns=list(columns))
        parts.append(s.rename(name))
    return pd.concat(parts, axis=1).dropna()


# ===========================================================================
# Tables
# ===========================================================================
def summary_table(m, setting="OOD", stat="mean", omics=None):
    """Per omic: feature-wise mean (or std) of sharpness, calibration and scoring."""
    rows = []
    for omic in omics or m:
        if setting not in m[omic]:
            continue
        d = m[omic][setting]
        agg = getattr(pd.Series, stat)
        rows.append({
            "Omic": cfg.FULL_NAMES[omic],
            f"Sharpness ({stat})": round(agg(d["sharpness"].mean(axis=0)), 4),
            f"Calibration ({stat})": round(agg(d["calibration"]), 4),
            f"Scoring ({stat})": round(agg(d["scoring"]), 4),
        })
    return pd.DataFrame(rows).set_index("Omic")


def overconfident_features(m, setting="IS"):
    """Features whose mean PIW is smaller than their RMSE."""
    rows = []
    for omic in cfg.CONTINUOUS:
        if omic not in m or setting not in m[omic]:
            continue
        sharp = m[omic][setting]["sharpness"].mean()
        rmse = m[omic][setting]["error"]
        n = int((sharp < rmse).sum())
        rows.append({"omic": omic, "overconfident": n, "total": len(sharp), "percent": round(100 * n / len(sharp), 2)})
    return pd.DataFrame(rows).set_index("omic")


def small_piw_features(sharpness, threshold=1e-4, min_percentage=50):
    """Features with at least `min_percentage`% of datapoints whose PIW is below `threshold`."""
    pct = (sharpness < threshold).sum() / len(sharpness) * 100
    return pct[pct >= min_percentage].sort_values(ascending=False)


def outliers(series, method="zscore", k=None):
    """Outlying values of a Series: |z| > k (default 3) or outside the k*IQR fences (default 1.5)."""
    s = series.dropna()
    if method == "zscore":
        k = 3 if k is None else k
        return s[(s - s.mean()).abs() > k * s.std()]
    k = 1.5 if k is None else k
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s[(s < q1 - k * iqr) | (s > q3 + k * iqr)].sort_values()


def sharpness_outlier_report(m, omic, setting="IS", n_show=10):
    """Z-score outliers of a sharpness matrix at feature, observation and datapoint level."""
    df = m[omic][setting]["sharpness"]
    print(f"--- Outliers for {omic} ({setting}) ---")
    print("Feature mean outliers:", outliers(df.mean(axis=0)).index.tolist())
    print("Observation mean outliers:", outliers(df.mean(axis=1)).index.tolist())
    values = df.values
    rows, cols = np.where(np.abs(values - np.nanmean(values)) > 3 * np.nanstd(values))
    print(f"Individual datapoint outliers (count {len(rows)}):")
    for r, c in list(zip(rows, cols))[:n_show]:
        print(f"  Obs: {df.index[r]}, Feature: {df.columns[c]}, Value: {df.iloc[r, c]}")


def metric_ranges(m, omic):
    """Min/max of each feature-level metric, in-sample and out-of-sample."""
    rows = {}
    for setting in m[omic]:
        for key in ("sharpness", "calibration", "scoring", "error"):
            s = feature_metric(m, omic, setting, key)
            rows[f"{key} ({setting})"] = {"Min": s.min(), "Max": s.max()}
    return pd.DataFrame(rows).T


# ===========================================================================
# Trend line + scatter
# ===========================================================================
def fit_trend(x, y, x0=None):
    """Least-squares line with 95% confidence band. Returns dict(x_line, y_line, conf, r2) or None."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2:
        return None
    m, b = np.polyfit(x, y, 1)
    lo = x.min() if x0 is None else max(x0, x.min())
    x_line = np.linspace(lo, x.max(), 200)
    y_line = m * x_line + b
    y_fit = m * x + b
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - y_fit) ** 2) / ss_tot if ss_tot != 0 else np.nan
    conf = None
    if x.size >= 3:
        n = len(x)
        s_err = np.sqrt(np.sum((y - y_fit) ** 2) / (n - 2))
        conf = t_dist.ppf(0.975, df=n - 2) * s_err * np.sqrt(1 / n + (x_line - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    return dict(x_line=x_line, y_line=y_line, conf=conf, r2=r2)


def _hide_marginal(ax):
    ax.grid(False)
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_marginals(ax, x, y, size="18%", pad=0.12, alpha=0.7, log_y=False):
    """Histograms of x (top) and y (right) attached to `ax`."""
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size=size, pad=pad, sharex=ax)
    ax_right = divider.append_axes("right", size=size, pad=pad, sharey=ax)
    bins = min(30, max(5, len(x) // 3))
    ax_top.hist(x, bins=bins, color="steelblue", alpha=alpha, density=True)
    ax_right.hist(y, bins=bins, color="steelblue", alpha=alpha, orientation="horizontal", density=True)
    ax_top.margins(y=0)
    ax_right.margins(x=0)
    if log_y:
        ax_right.set_yscale("log")
    _hide_marginal(ax_top)
    _hide_marginal(ax_right)
    return ax_top, ax_right


def _density_contour(ax, x, y, smooth=0.08):
    """Contours at the 50/75/90% quantiles of a 2D histogram of the points."""
    if x.size < 5:
        return
    bins = min(40, max(10, x.size // 5))
    z, xe, ye = np.histogram2d(x, y, bins=bins)
    z = gaussian_filter(z.T, sigma=smooth)
    positive = z[z > 0]
    if positive.size < 3:
        return
    levels = np.unique(np.quantile(positive, [0.5, 0.75, 0.9]))
    if levels.size >= 2:
        ax.contour((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, z, levels=levels, colors="black", linewidths=1.0, alpha=0.8)


def _highlight(ax, df, x, y, feature, offset=(6, 6)):
    """Circle and label a feature (case-insensitive exact match, else substring)."""
    idx = df.index.astype(str)
    mask = idx.str.lower() == feature.lower()
    if not mask.any():
        mask = idx.str.lower().str.contains(feature.lower(), regex=False)
    if not mask.any():
        print(f"Feature {feature!r} not found.")
        return
    for f in df.index[mask]:
        ax.scatter(df.loc[f, x], df.loc[f, y], s=90, facecolors="none", edgecolors="red", linewidths=1.5, zorder=6)
        ax.annotate(feature, (df.loc[f, x], df.loc[f, y]), textcoords="offset points", xytext=offset, fontsize=9, color="red")


def _inset_zoom(ax, df, x, y, c, trend, cmap, norm, quantiles=(0.005, 0.995), rect=(0.45, 0.30, 0.4125, 0.4125)):
    """Inset showing the central mass of the points (between the given quantiles)."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    cx, cy = df[x].median(), df[y].median()
    sx = df[x].quantile(quantiles[1]) - df[x].quantile(quantiles[0])
    sy = df[y].quantile(quantiles[1]) - df[y].quantile(quantiles[0])
    zx = (max(xlim[0], cx - sx / 2), min(xlim[1], cx + sx / 2))
    zy = (max(ylim[0], cy - sy / 2), min(ylim[1], cy + sy / 2))
    ins = ax.inset_axes(list(rect))
    ins.scatter(df[x], df[y], c=df[c] if c else None, cmap=cmap, norm=norm, alpha=0.35, s=12)
    if trend:
        ins.plot(trend["x_line"], trend["y_line"], color=cmap(0.5), linewidth=1.5)
        if trend["conf"] is not None:
            ins.fill_between(trend["x_line"], trend["y_line"] - trend["conf"], trend["y_line"] + trend["conf"], color=cmap(0.6), alpha=0.15)
    ins.set_xlim(zx)
    ins.set_ylim(zy)
    ins.tick_params(labelsize=8)
    ax.add_patch(mpl.patches.Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, edgecolor="black", linewidth=0.5))
    for xy_a, xy_b in (((zx[0], zy[1]), (0.0, 1.0)), ((zx[1], zy[0]), (1.0, 0.0))):
        ax.figure.add_artist(mpl.patches.ConnectionPatch(xyA=xy_a, xyB=xy_b, coordsA=ax.transData, coordsB=ins.transAxes, color="black", linewidth=0.5))


def scatter_trend(ax, df, x, y, c=None, *, cmap=BLUES_NO_WHITE, norm=None, alpha=1.0, s=20,
                  trend=True, ci=True, trend_x0=None, marginals=False, contour=False, log_y=False,
                  highlight=None, sample=None, random_state=42, inset_zoom=False,
                  xlim=None, ylim=None, nonneg=True, colorbar=None, xlabel=None, ylabel=None,
                  legend_loc="best", uniform_color=None):
    """Feature-wise scatter of df[x] vs df[y] coloured by df[c], with least-squares trend line,
    95% CI and R^2. The trend is always fitted on all rows; `sample` (fraction) only thins the
    plotted points. Returns the scatter artist (for a shared colour bar)."""
    fit = fit_trend(df[x], df[y], x0=trend_x0) if trend else None
    pts = df.sample(frac=sample, random_state=random_state) if sample else df
    if c is None:
        sc = ax.scatter(pts[x], pts[y], color=uniform_color or cmap(0.65), alpha=alpha, s=s)
    else:
        sc = ax.scatter(pts[x], pts[y], c=pts[c], cmap=cmap, norm=norm, alpha=alpha, s=s)
    if fit:
        ax.plot(fit["x_line"], fit["y_line"], color=cmap(0.5), linewidth=2, label=f"Trend line ($R^2$ = {fit['r2']:.3f})")
        if ci and fit["conf"] is not None:
            ax.fill_between(fit["x_line"], fit["y_line"] - fit["conf"], fit["y_line"] + fit["conf"], color=cmap(0.6), alpha=0.20, label="95% CI")
    if contour:
        _density_contour(ax, df[x].to_numpy(), df[y].to_numpy())
    if highlight:
        _highlight(ax, df, x, y, highlight)  # search all features, not only the plotted sample
    if log_y:
        ax.set_yscale("log")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if nonneg and not log_y:
        ax.set_xlim(left=max(0.0, ax.get_xlim()[0]))
        ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0]))
    if inset_zoom:
        _inset_zoom(ax, df, x, y, c, fit, cmap, norm)
    if marginals:
        add_marginals(ax, df[x].to_numpy(), df[y].to_numpy(), log_y=log_y)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if colorbar and c is not None:
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(colorbar)
    if fit:
        leg = ax.legend(frameon=True, fontsize=9, loc=legend_loc)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(1)
    return sc


def feature_scatter(m, omic, setting, x="sharpness", y="calibration", c="scoring", info=None, figsize=(10, 6), **kw):
    """One omic, one setting: scatter_trend of three feature-level metrics with default labels."""
    df = feature_table(m, omic, setting, info, x=x, y=y, **({"c": c} if c else {}))
    fig, ax = plt.subplots(figsize=figsize)
    scatter_trend(ax, df, "x", "y", "c" if c else None, xlabel=label(x, omic), ylabel=label(y, omic),
                  colorbar=label(c, omic) if c else None, **kw)
    fig.tight_layout()
    plt.show()


def compare_is_ood(m, omic, x="sharpness", y="calibration", c="scoring", info=None, figsize=(10, 6), **kw):
    """Two figures (in-sample, out-of-sample) with identical axis limits and colour scale."""
    dfs = {s: feature_table(m, omic, s, info, x=x, y=y, **({"c": c} if c else {})) for s in m[omic]}
    both = pd.concat(dfs.values())
    x_pad = (both["x"].max() - both["x"].min()) * 0.05
    y_pad = (both["y"].max() - both["y"].min()) * 0.05
    kw.setdefault("xlim", (max(0.0, both["x"].min() - x_pad), both["x"].max() + x_pad))
    kw.setdefault("ylim", (0, both["y"].max() + y_pad))
    norm = mpl.colors.Normalize(vmin=both["c"].min(), vmax=both["c"].max()) if c else None
    for setting, df in dfs.items():
        fig, ax = plt.subplots(figsize=figsize)
        sc = scatter_trend(ax, df, "x", "y", "c" if c else None, norm=norm,
                           xlabel=label(x, omic), ylabel=label(y, omic), **kw)
        if c:
            fig.colorbar(sc, ax=ax, label=label(c, omic))
        fig.suptitle(f"{cfg.FULL_NAMES[omic]} - {SETTING_LABELS[setting]}")
        fig.tight_layout()
        plt.show()


def grid_scatter(m, setting, x="sharpness", y="calibration", c="scoring", omics=None, info=None,
                 marginals=True, contour=(), no_trend=(), log_y=(), sample_max=None, title_suffix="", **kw):
    """One panel per omic (2x3 for the continuous omics, 2x4 with copy number).

    contour / no_trend / log_y: omics for which to draw density contours, skip the trend line,
    or use a log y-axis. sample_max: thin panels to about this many plotted points.
    A missing colour metric for an omic (e.g. % missing) falls back to a single colour."""
    omics = omics or cfg.CONTINUOUS
    ncols = 3 if len(omics) <= 6 else 4
    fig = plt.figure(figsize=(6 * ncols, 12))
    gs = fig.add_gridspec(2, ncols, hspace=0.4, wspace=0.4)
    for i, omic in enumerate(omics):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        has_c = c is not None and feature_metric(m, omic, setting, c, info) is not None
        df = feature_table(m, omic, setting, info, x=x, y=y, **({"c": c} if has_c else {}))
        name = f"{cfg.FULL_NAMES[omic]}{title_suffix}"
        if df.empty:
            ax.text(0.5, -0.20, f"{name}: no data", transform=ax.transAxes, ha="center", va="top", fontsize=10)
            continue
        sample = None
        if sample_max and len(df) > sample_max:
            sample = max(0.05, sample_max / len(df))
        norm = mpl.colors.Normalize(vmin=0, vmax=100) if c == "missing" and has_c else None
        sc = scatter_trend(ax, df, "x", "y", "c" if has_c else None, norm=norm, sample=sample,
                           trend=omic not in no_trend, contour=omic in contour, log_y=omic in log_y,
                           marginals=marginals, xlabel=label(x, omic), ylabel=label(y, omic), **kw)
        ax.text(0.5, -0.20, name, transform=ax.transAxes, ha="center", va="top", fontsize=16)
        if has_c:
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(label(c, omic))
            cb.solids.set_alpha(0.8)
    plt.show()


# ===========================================================================
# Distributions across omics
# ===========================================================================
def plot_distribution(m, key, axis=0, kind="boxen", settings=cfg.SETTINGS, omics=None, ylabel=None,
                      positive_only=False, figsize=None, legend_loc="upper right"):
    """Distribution per omic of a metric, in-sample vs out-of-sample.

    For sharpness, axis=0 gives the feature-wise mean and axis=1 the observation-wise mean."""
    omics = omics or cfg.CONTINUOUS
    frames = []
    for omic in omics:
        for setting in settings:
            if setting not in m[omic]:
                continue
            v = m[omic][setting][key]
            v = v.mean(axis=axis) if isinstance(v, pd.DataFrame) else v
            v = v.dropna()
            if positive_only:
                v = v[v > 0]
            frames.append(pd.DataFrame({"Mean Value": v.to_numpy(), "Omic": cfg.FULL_NAMES[omic], "Dataset": SETTING_LABELS[setting]}))
    plot_data = pd.concat(frames, ignore_index=True)
    palette = {SETTING_LABELS[s]: PALETTE[s] for s in settings}

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize or (14 if len(omics) > 1 else 6, 6))
    if kind == "violin":
        sns.violinplot(data=plot_data, x="Omic", y="Mean Value", hue="Dataset", palette=palette,
                       dodge=True, inner="quartile", cut=0, density_norm="width")
    else:
        sns.boxenplot(data=plot_data, x="Omic", y="Mean Value", hue="Dataset", palette=palette)
    plt.ylabel(ylabel or key, rotation=0, labelpad=40)
    plt.ylim(bottom=0)
    plt.legend(title="Dataset", loc=legend_loc)
    plt.show()


def plot_omic_means(m, x="sharpness", y="calibration"):
    """One point per omic and setting: mean over features of two metrics."""
    plt.figure(figsize=(8, 6))
    for setting, dy in (("IS", 4), ("OOD", -10)):
        for omic in cfg.CONTINUOUS:
            if setting not in m.get(omic, {}):
                continue
            px_ = feature_metric(m, omic, setting, x).mean()
            py_ = feature_metric(m, omic, setting, y).mean()
            plt.scatter(px_, py_, c=PALETTE[setting], s=80, alpha=0.8,
                        label=SETTING_LABELS[setting] if omic == cfg.CONTINUOUS[0] else None)
            plt.annotate(cfg.FULL_NAMES[omic], (px_, py_), xytext=(4, dy), textcoords="offset points", fontsize=9, color=PALETTE[setting])
    plt.xlabel(f"Mean {label(x, 'x')}")
    plt.ylabel(f"Mean {label(y, 'x')}")
    plt.title(f"Per omic: mean {label(x, 'x')} vs mean {label(y, 'x')}")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Correlations
# ===========================================================================
def corr_matrix(df, method="pearson"):
    """Pairwise correlation and p-value matrices, each pair on its complete rows."""
    fn = pearsonr if method == "pearson" else spearmanr
    labels = df.columns.tolist()
    r = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
    p = pd.DataFrame(np.zeros((len(labels), len(labels))), index=labels, columns=labels)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            pair = df[[a, b]].dropna()
            ri, pi = fn(pair[a], pair[b]) if len(pair) >= 2 else (np.nan, np.nan)
            r.loc[a, b] = r.loc[b, a] = ri
            p.loc[a, b] = p.loc[b, a] = pi
    return r, p


def plot_corr_heatmap(r, ax=None, cbar_label="Pearson correlation", cbar=True, title=None, fontsize=10):
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
    sns.heatmap(r, annot=True, fmt=".2f", cmap="GnBu", vmin=-1, vmax=1, center=0, square=True,
                linewidths=0.5, linecolor="white", cbar=cbar,
                cbar_kws={"label": cbar_label} if cbar else None, ax=ax)
    ax.tick_params(axis="x", rotation=0, labelsize=fontsize)
    ax.tick_params(axis="y", rotation=0, labelsize=fontsize)
    if title:
        ax.set_title(title, fontsize=12)
    return ax


def plot_clustermap(r, figsize=(8, 6)):
    """Correlation heatmap with rows/columns ordered by average-linkage clustering on 1 - r."""
    dist = 1 - r.clip(-1, 1)
    np.fill_diagonal(dist.values, 0)
    link = linkage(squareform(dist.values, checks=False), method="average")
    g = sns.clustermap(r, annot=True, fmt=".2f", cmap="GnBu", vmin=-1, vmax=1, center=0,
                       figsize=figsize, row_linkage=link, col_linkage=link)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=0, fontsize=8.75)
    plt.show()


def omic_metric_table(m, omic, setting, info, with_pearson=True):
    """Feature-level metrics of one omic used for the correlation matrices."""
    if omic == "copynumber":
        cols = {"Mean Feature\nEntropy": "sharpness", "ECE": "calibration", "Feature\nLog Score": "scoring",
                "Balanced\nAccuracy": "error", "Feature\nDiversity": "diversity"}
    else:
        cols = {"Feature \nPIW": "sharpness", "ENCE": "calibration", "CRPS": "scoring", "Feature\nRMSE": "error"}
        if with_pearson:
            cols["Feature\nCorrelation"] = "pearson"
        cols["Original \n Feature's \n Variance"] = "variance"
        if omic in cfg.OMICS_WITH_MISSING:
            cols["Missing \n Values"] = "missing"
    return feature_table(m, omic, setting, info, **cols)


def corr_grid(m, setting, info, omics=None, with_pearson=False):
    """Pearson correlation heatmap of the feature-level metrics, one panel per omic."""
    omics = omics or cfg.OMICS
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.4)
    for i, omic in enumerate(omics):
        if setting not in m[omic]:
            ax = fig.add_subplot(gs[i // 4, i % 4])
            ax.set_axis_off()
            ax.set_title(f"{omic}: not available", fontsize=12)
            continue
        r, _ = corr_matrix(omic_metric_table(m, omic, setting, info, with_pearson))
        print(f"Pearson correlation matrix for {SETTING_LABELS[setting]} {omic}:")
        display(r.round(2))
        plot_corr_heatmap(r, ax=fig.add_subplot(gs[i // 4, i % 4]), cbar=False, title=omic, fontsize=11)
    plt.show()


# ===========================================================================
# Observation-wise / tissue analysis
# ===========================================================================
def true_obs_index(idx):
    """Five-digit sample key shared by 'SIDM01974', '01974', 'ACH-001974', ..."""
    idx = pd.Index(idx.astype(str))
    key = idx.to_series().str.extract(r"(\d{5})$")[0]
    key = key.fillna(idx.to_series().str.extract(r"(\d+)$")[0].str[-5:])
    return pd.Index(key.str.zfill(5).to_numpy())


def mean_sharpness_by_obs(m, tissue, setting="IS", omics=None):
    """Mean sharpness per observation for each omic, plus number of omics and tissue."""
    omics = omics or cfg.OMICS
    cols = []
    for omic in omics:
        df = m[omic][setting]["sharpness"]
        cols.append(df.set_axis(true_obs_index(df.index), axis=0).groupby(level=0).mean().mean(axis=1).rename(omic))
    out = pd.concat(cols, axis=1)
    out["n_omics"] = out[omics].notna().sum(axis=1)
    tissue_map = tissue.copy()
    tissue_map.index = true_obs_index(tissue_map.index)
    tissue_map = tissue_map.groupby(level=0).apply(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    out["tissue"] = out.index.map(tissue_map).fillna("Unknown")
    return out


def plot_sharpness_by_n_omics(by_obs, omics=None, min_obs=30):
    """Mean sharpness per omic, grouped by how many omics an observation has (n >= 2)."""
    omics = omics or cfg.OMICS
    rows = []
    for omic in omics:
        for n in sorted(by_obs["n_omics"].unique()):
            if n < 2:
                continue
            subset = by_obs.loc[by_obs["n_omics"] == n, omic]
            if subset.notna().sum() < min_obs:
                print(f"{omic}, n_omics = {n}: only {subset.notna().sum()} observations, skipped")
                continue
            rows.append({"Omic": omic, "n_omics": int(n), "Mean Sharpness": subset.mean()})
    plot_df = pd.DataFrame(rows).dropna()
    fig, ax = plt.subplots(figsize=(14, 6))
    omics_present = plot_df["Omic"].unique()
    n_values = sorted(plot_df["n_omics"].unique())
    x = np.arange(len(omics_present))
    width = 0.12
    for i, n in enumerate(n_values):
        sub = plot_df[plot_df["n_omics"] == n].set_index("Omic")["Mean Sharpness"]
        ax.bar(x + i * width, [sub.get(o, 0) for o in omics_present], width, label=f"n_omics={n}")
    ax.set_xlabel("Omic")
    ax.set_ylabel("Mean Sharpness")
    ax.set_xticks(x + width * (len(n_values) - 1) / 2)
    ax.set_xticklabels(omics_present, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.show()


def tissue_mean_sharpness(by_obs, omics=None):
    omics = omics or cfg.OMICS
    out = by_obs.groupby("tissue")[omics].mean()
    out["n_obs"] = by_obs.groupby("tissue").size()
    out["continuous_mean_sharpness"] = out[[o for o in cfg.CONTINUOUS if o in omics]].mean(axis=1)
    return out.sort_values("continuous_mean_sharpness", ascending=False)


def plot_tissue_parallel(tissue_means, omics=None, min_obs=20):
    """Parallel coordinates of min-max normalised mean sharpness per tissue (tissues with > min_obs)."""
    omics = omics or cfg.OMICS
    data = tissue_means[tissue_means["n_obs"] > min_obs]
    scaled = data[omics].copy()
    for col in omics:
        lo, hi = scaled[col].min(), scaled[col].max()
        if hi - lo > 0:
            scaled[col] = (scaled[col] - lo) / (hi - lo)
    scaled["Tissue"] = data.index
    plt.figure(figsize=(16, 8))
    ax = plt.gca()
    parallel_coordinates(scaled, "Tissue", alpha=0.8, linewidth=2.5, color=[plt.cm.tab20(i) for i in range(len(scaled))])
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
    plt.ylabel("Normalized Mean Sharpness", fontsize=12, fontweight="bold")
    plt.ylim(-0.01, 1.01)
    plt.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    plt.legend(title="Tissue", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, framealpha=0.95)
    plt.tight_layout()
    plt.show()


def plot_obs_sharpness_by_tissue(m, omic, tissue, setting="IS"):
    """Mean sharpness of every observation, sorted by tissue."""
    df = m[omic][setting]["sharpness"].mean(axis=1).rename("mean_sharpness").to_frame()
    df["tissue"] = df.index.map(tissue)
    if df["tissue"].isna().all():  # numeric IDs -> SIDMxxxxx
        df["tissue"] = df.index.to_series().astype(int).map(lambda x: f"SIDM{x:05d}").map(tissue)
    df["tissue"] = df["tissue"].fillna("Unknown")
    df = df.sort_values(["tissue", "mean_sharpness"]).reset_index(drop=True)
    df["obs_order"] = np.arange(len(df))
    plt.figure(figsize=(16, 6))
    sns.scatterplot(data=df, x="obs_order", y="mean_sharpness", hue="tissue", s=24, alpha=0.85, linewidth=0)
    plt.title(f"{omic} mean sharpness per observation ({SETTING_LABELS[setting]})")
    plt.xlabel("Observation (sorted by tissue)")
    plt.ylabel("Mean sharpness")
    plt.legend(title="Tissue", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.show()


def plot_mean_histograms(sharpness, omic, what="PIW", xlim=None):
    """Histograms of the feature-wise and observation-wise mean of a sharpness matrix."""
    for axis, who in ((0, "Feature"), (1, "Cell line")):
        means = sharpness.mean(axis=axis)
        plt.figure(figsize=(12, 5))
        sns.histplot(means, bins=50, color="skyblue", edgecolor="black")
        plt.axvline(means.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean = {means.mean():.3f}")
        plt.title(f"{cfg.FULL_NAMES[omic]}: histogram of {who.lower()}-wise average {what}", fontsize=16, fontweight="bold")
        plt.xlabel(f"Average {what} per {'feature' if axis == 0 else 'observation'}", fontsize=13)
        plt.ylabel("Frequency", fontsize=13, rotation=0, labelpad=40)
        if xlim:
            plt.xlim(*xlim)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ===========================================================================
# Calibration diagrams for continuous omics (need the prepared data and cube)
# ===========================================================================
def plot_ence_reliability(data, cube, features, ence_values=None, style="overlay_bars", n_bins=10, title=None):
    """RMSE vs RMV per bin of predicted std (equal-width bins, as in the saved ENCE), 2x3 panels.

    style="overlay_bars": RMSE and RMV bars per bin; style="line": RMSE against RMV."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for ax, feature in zip(axes.ravel(), features):
        j = data.columns.get_loc(feature)
        y = data[feature].to_numpy(dtype=float)
        samples = cube[:, j, :]
        mu, sd = samples.mean(axis=1), samples.std(axis=1, ddof=0)
        valid = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sd)
        y, mu, sd = y[valid], mu[valid], sd[valid]
        if y.size == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        frame = pd.DataFrame({"std": sd, "sq_error": (y - mu) ** 2})
        n_eff = min(n_bins, frame["std"].nunique())
        if n_eff <= 1:
            frame["bin"] = 0
            edges = np.array([frame["std"].min(), frame["std"].max() + 1e-12])
        else:
            _, edges = pd.cut(frame["std"], bins=n_eff, retbins=True, duplicates="drop")
            frame["bin"] = pd.cut(frame["std"], bins=edges, include_lowest=True, labels=False)
        g = frame.groupby("bin", observed=True, sort=True)
        left = np.array([edges[int(b)] for b in g.groups])
        width = np.array([edges[int(b) + 1] - edges[int(b)] for b in g.groups])
        rmv = g["std"].apply(lambda s: np.sqrt(np.mean(s.to_numpy() ** 2))).to_numpy()
        rmse_ = g["sq_error"].apply(lambda s: np.sqrt(np.mean(s.to_numpy()))).to_numpy()
        max_val = max(rmv.max(), rmse_.max(), 1e-12)
        pad = 0.08 * max_val
        if style == "line":
            ax.plot(rmv, rmse_, "o-", linewidth=2, markersize=8, color="#1f77b4", label="ENCE curve")
            ax.set_xlabel("Root Mean Predicted Std (RMV)", fontsize=11)
            ax.set_ylabel("Root Mean Squared Error (RMSE)", fontsize=11)
            ax.set_xlim(0, max_val + pad)
        else:
            bar_w = np.nanmin(width) if np.isfinite(np.nanmin(width)) and np.nanmin(width) > 0 else 0.05
            ax.bar(left, rmse_, width=bar_w, align="edge", color="#fdae6b", edgecolor="#d94801", alpha=0.55, label="RMSE per bin", zorder=2)
            ax.bar(left, rmv, width=bar_w, align="edge", color="#9ecae1", edgecolor="#1f77b4", alpha=0.8, label="RMV per bin", zorder=3)
            ax.set_xlabel("Binned predicted std", fontsize=11)
            ax.set_ylabel("Value", fontsize=11)
            ax.set_xlim(0, max(float(np.max(left + width)) + pad, max_val + pad))
            ax.grid(True, alpha=0.3, axis="y")
        ax.plot([0, max_val + pad], [0, max_val + pad], "k--", linewidth=1.5, label="Perfect calibration", zorder=4)
        ax.set_ylim(0, max_val + pad)
        ax.set_title(feature, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        if ence_values is not None and feature in ence_values.index:
            ax.text(0.05, 0.95, f"ENCE={ence_values[feature]:.3f}", transform=ax.transAxes, fontsize=10,
                    verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.show()


def plot_coverage_calibration(data, cube, features, crps=None, title=None):
    """Empirical vs nominal coverage of central prediction intervals, 2x3 panels."""
    nominal = np.linspace(0.10, 0.99, 18)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    for ax, feature in zip(axes.ravel(), features):
        j = data.columns.get_loc(feature)
        y = data[feature].to_numpy(dtype=float)
        valid = ~np.isnan(y)
        samples = cube[valid, j, :]
        empirical = []
        for cov in nominal:
            a = (1.0 - cov) / 2.0
            lo, hi = np.nanquantile(samples, a, axis=1), np.nanquantile(samples, 1.0 - a, axis=1)
            empirical.append(np.mean((y[valid] >= lo) & (y[valid] <= hi)))
        empirical = np.asarray(empirical)
        gap = np.mean(np.abs(empirical - nominal))
        ax.plot(nominal, empirical, marker="o", lw=2, label="Empirical")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Ideal")
        ax.fill_between(nominal, nominal, empirical, alpha=0.15)
        crps_txt = f"CRPS={crps[feature]:.4f} | " if crps is not None and feature in crps.index else ""
        ax.set_title(f"{feature}\n{crps_txt}Gap={gap:.4f}")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes.ravel()[0].legend(loc="lower right")
    fig.suptitle(title or "Probability calibration curves", fontsize=16, fontweight="bold")
    fig.supxlabel("Nominal Coverage")
    fig.supylabel("Empirical Coverage")
    plt.tight_layout()
    plt.show()


def plot_feature_truth_vs_prediction(data, imputed, feature):
    """Ground truth vs point prediction for one feature, with RMSE and R^2."""
    x, y = data[feature], imputed[feature]
    mask = x.notna() & y.notna()
    xv, yv = x[mask], y[mask]
    rmse_ = np.sqrt(np.mean((xv - yv) ** 2))
    ss_tot = np.sum((xv - xv.mean()) ** 2)
    r2 = 1 - np.sum((xv - yv) ** 2) / ss_tot if ss_tot != 0 else np.nan
    lo, hi = min(xv.min(), yv.min()), max(xv.max(), yv.max())
    plt.figure(figsize=(6, 6))
    plt.scatter(xv, yv, alpha=0.6, label=f"RMSE = {rmse_:.3f}\n$R^2$ = {r2:.3f}")
    plt.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1, label="y = x")
    plt.xlabel(f"ground truth: {feature}")
    plt.ylabel(f"prediction: {feature}")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.show()


def plot_truth_vs_prediction_means(truth, imputed, xlabel="Mean of ground truth per feature", ylabel="Mean of prediction per feature"):
    """Per-feature mean of the ground truth against the per-feature mean of the prediction."""
    a, b = truth.mean(axis=0), imputed.mean(axis=0)
    a, b = a.align(b, join="inner")
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    plt.figure(figsize=(6, 6))
    plt.scatter(a, b, alpha=0.6)
    plt.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis("equal")
    plt.show()


# ===========================================================================
# Copy number
# ===========================================================================
def cn_dataset_description(data, imputed=None, classes=cfg.CN_CLASSES):
    """Missingness, class diversity and value ranges of the copy-number ground truth."""
    n_rows = data.shape[0]
    pct_missing = data.isna().sum() / n_rows * 100
    with_na = pct_missing[pct_missing > 0]
    print(f"The dataset has {round(data.isna().values.mean() * 100, 2)}% missing values")
    print(f"Features with missing values: {len(with_na)}")
    if len(with_na):
        print(f"  mean {with_na.mean():.2f}%, max {with_na.max():.2f}% {with_na[with_na == with_na.max()].index.tolist()}, "
              f"min {with_na.min():.2f}% {with_na[with_na == with_na.min()].index.tolist()}")

    n_unique = data.nunique(dropna=True)
    print(f"\nTotal features: {data.shape[1]}")
    print(f"Unique values per feature: mean {n_unique.mean():.2f}, median {n_unique.median()}, min {n_unique.min()}, max {n_unique.max()}")
    print("Number of features by number of unique values:")
    print(n_unique.value_counts().sort_index().to_string())
    print("\nMost frequent values across all features (%):")
    print((pd.Series(data.values.ravel()).value_counts(dropna=True, normalize=True).head(10) * 100).round(2).to_string())

    if imputed is not None:
        truth_sets = data.apply(lambda col: set(col.dropna().unique()))
        pred_sets = imputed.apply(lambda col: set(col.unique()))
        match = [f for f in data.columns if truth_sets[f] == pred_sets[f]]
        print(f"\nFeatures whose predicted values cover exactly the observed classes: {len(match)} of {data.shape[1]}")
        for f in [f for f in data.columns if f not in match][:3]:
            print(f"  e.g. {f}: data {truth_sets[f]} vs imputed {pred_sets[f]}")

    # Class diversity: entropy of the class distribution and number of classes with p > 0.05
    classes = np.asarray(classes)
    stats = {}
    for col in data.columns:
        counts = data[col].value_counts(dropna=False)
        p = np.array([counts.get(c, 0) for c in classes], dtype=float)
        p = p / p.sum() if p.sum() > 0 else np.zeros_like(p)
        h = -(p[p > 0] * np.log2(p[p > 0])).sum()
        stats[col] = {"entropy": h, "n_classes_above_005": int((p > 0.05).sum())}
    stats = pd.DataFrame(stats).T
    high = stats[(stats["entropy"] > 0.1) & (stats["n_classes_above_005"] >= 3)].index
    print(f"\nFeatures with only 2 classes: {(n_unique == 2).sum()}")
    print(f"High-diversity features (entropy > 0.1 bits and >= 3 classes with p > 0.05): {len(high)} of {data.shape[1]}")
    example = high[0] if len(high) else data.columns[0]
    counts = data[example].value_counts(dropna=False)
    p = np.array([counts.get(c, 0) for c in classes], dtype=float)
    print(f"\nExample class distribution for {example!r}:")
    print(pd.DataFrame({"class": classes, "p_i": p / p.sum()}).to_string(index=False))
    return stats


def class_sharpness_table(truth, sharpness, classes=cfg.CN_CLASSES):
    """Mean/std of datapoint sharpness grouped by the ground-truth class (and missing truth)."""
    rows = truth.index.intersection(sharpness.index)
    cols = truth.columns.intersection(sharpness.columns)
    cls = truth.loc[rows, cols].stack(future_stack=True)
    sharp = sharpness.loc[rows, cols].stack(future_stack=True)
    finite = np.isfinite(sharp.to_numpy())
    out = []
    for c in list(classes) + ["N/a"]:
        mask = (cls.isna().to_numpy() if c == "N/a" else (cls.to_numpy() == c)) & finite
        v = sharp[mask]
        out.append({"Class": c, "Mean Sharpness": v.mean() if len(v) else np.nan,
                    "Std Sharpness": v.std() if len(v) else np.nan, "n_datapoints": int(len(v))})
    out = pd.DataFrame(out)
    out["percent_datapoints"] = out["n_datapoints"] / out["n_datapoints"].sum() * 100
    return out


def plot_class_sharpness_bars(tables, ylabel="Mean Sample Entropy", error_bars=False):
    """Grouped bars of mean sharpness per ground-truth class, one bar per setting."""
    order = next(iter(tables.values()))["Class"].astype(str).tolist()
    x = np.arange(len(order))
    width = 0.38
    plt.figure(figsize=(8, 4.5))
    for setting, off in (("IS", -width / 2), ("OOD", width / 2)):
        if setting not in tables:
            continue
        t = tables[setting].assign(Class=tables[setting]["Class"].astype(str)).set_index("Class").reindex(order)
        extra = dict(yerr=t["Std Sharpness"].to_numpy(), error_kw=dict(elinewidth=1.2, capsize=4, capthick=1.2, ecolor="black")) if error_bars else {}
        plt.bar(x + off, t["Mean Sharpness"].to_numpy(), width=width, color=PALETTE[setting], label=SETTING_LABELS[setting], alpha=0.9, **extra)
    plt.xticks(x, order)
    plt.xlabel("Ground Truth Class")
    plt.ylabel(ylabel)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def plot_class_sharpness_boxen(truth, sharpness_by_setting, classes=cfg.CN_CLASSES):
    """Boxen plot of datapoint sharpness per ground-truth class (incl. missing), IS vs OOD."""
    frames = []
    for setting, sharp in sharpness_by_setting.items():
        rows = truth.index.intersection(sharp.index)
        cols = truth.columns.intersection(sharp.columns)
        df = pd.DataFrame({"Class_raw": truth.loc[rows, cols].stack(future_stack=True),
                           "Sharpness": sharp.loc[rows, cols].stack(future_stack=True)})
        df = df[np.isfinite(df["Sharpness"])].copy()
        df["Class"] = "Other"
        for c in classes:
            df.loc[df["Class_raw"] == c, "Class"] = str(c)
        df.loc[df["Class_raw"].isna(), "Class"] = "Missing"
        df["Dataset"] = SETTING_LABELS[setting]
        frames.append(df[df["Class"] != "Other"])
    plot_data = pd.concat(frames, ignore_index=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 6))
    sns.boxenplot(data=plot_data, x="Class", y="Sharpness", hue="Dataset",
                  order=[str(c) for c in classes] + ["Missing"],
                  palette={SETTING_LABELS[s]: PALETTE[s] for s in sharpness_by_setting}, showfliers=False)
    plt.xlabel("Ground Truth Class")
    plt.ylabel("Sharpness")
    plt.ylim(bottom=0)
    plt.legend(title="", frameon=False)
    plt.tight_layout()
    plt.show()


def plot_entropy_overview(entropy_df):
    """Heatmap of datapoint entropy and box plots of its feature/observation means."""
    plt.figure(figsize=(16, 6))
    sns.heatmap(entropy_df.values, cmap="viridis", cbar_kws={"label": "Entropy"}, xticklabels=False, yticklabels=False)
    plt.title("Datapoint-wise entropy of the sampled classes")
    plt.xlabel("Features")
    plt.ylabel("Observations")
    plt.tight_layout()
    plt.show()
    for axis, who, box, dot in ((0, "Feature", "lightblue", "red"), (1, "Observation", "lightgreen", "orange")):
        means = entropy_df.mean(axis=axis)
        plt.figure(figsize=(8, 4))
        plt.boxplot(means.values, vert=False, patch_artist=True, boxprops=dict(facecolor=box))
        plt.scatter(means.values, [1] * len(means), color=dot, alpha=0.6, label=f"{who} means")
        plt.yticks([1], [f"{who}s"])
        plt.xlabel("Average Entropy")
        plt.title(f"Box plot of average entropy per {who.lower()}")
        plt.legend()
        plt.show()


def entropy_outlier_report(entropy_df, n_show=20):
    """IQR outliers of entropy at feature, observation and datapoint level."""
    for axis, who in ((0, "features"), (1, "observations")):
        means = entropy_df.mean(axis=axis)
        out = outliers(means, method="iqr")
        print(f"Outlier {who} (by average entropy): {len(out)} / {len(means)} ({100 * len(out) / len(means):.2f}%)")
        print(out.index.tolist())
    flat = entropy_df.stack(future_stack=True)
    out = outliers(flat, method="iqr")
    print(f"\nOutlier datapoints: {len(out)} / {flat.size} ({100 * len(out) / flat.size:.2f}%)")
    print(out.head(n_show).to_string())


def plot_entropy_hist_is_ood(entropy_by_setting, axis=0, bins=50):
    """Overlaid histograms of the feature-wise (axis=0) or observation-wise mean entropy."""
    plt.figure(figsize=(12, 5))
    for setting, df in entropy_by_setting.items():
        means = df.mean(axis=axis)
        sns.histplot(means, bins=bins, color=PALETTE[setting], edgecolor="black", alpha=0.6, label=SETTING_LABELS[setting])
        plt.axvline(means.mean(), color=PALETTE[setting], linestyle="--", linewidth=2, label=f"Mean ({setting}) = {means.mean():.3f}")
    who = "Feature" if axis == 0 else "Observation"
    plt.title(f"Histogram of {who.lower()}-wise average entropy", fontsize=16, fontweight="bold")
    plt.xlabel(f"Average Entropy per {who}", fontsize=13)
    plt.ylabel("Frequency", fontsize=13, rotation=0, labelpad=40)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _cn_confidence_accuracy(true_labels, samples, classes=cfg.CN_CLASSES):
    """Confidence and correctness of the most-sampled class, rows with ground truth only
    (the same definition as omics_metrics.calculate_ece)."""
    classes = np.asarray(classes)
    mask = ~pd.isna(true_labels)
    probs = om.cn_class_probs(samples[mask], classes)
    conf = probs.max(axis=1)
    acc = (classes[np.argmax(probs, axis=1)] == np.asarray(true_labels)[mask]).astype(float)
    return conf, acc


def _cn_bins(conf, n_bins=10):
    return np.clip(np.digitize(conf, np.linspace(0.0, 1.0, n_bins + 1), right=True) - 1, 0, n_bins - 1)


def cn_reliability(data, cube, features, style="curve", n_bins=10, min_obs=0):
    """Reliability diagram per feature (2x3 panels, or one figure per feature for style="gap_bars").

    Binning and confidence follow calculate_ece, so the annotated ECE equals the saved ECE.
    style="curve": mean accuracy vs mean confidence; "gap_bars": accuracy bars plus gap to the
    bin's upper edge (bins with < min_obs observations are not drawn)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    if style == "curve":
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()
    for i, feature in enumerate(features):
        conf, acc = _cn_confidence_accuracy(data[feature].values, cube[:, data.columns.get_loc(feature), :])
        bins = _cn_bins(conf, n_bins)
        stats = [(b, (bins == b).sum(), conf[bins == b].mean(), acc[bins == b].mean()) for b in range(n_bins) if (bins == b).any()]
        ece_value = sum(n / len(conf) * abs(a - c) for _, n, c, a in stats)
        if style == "curve":
            ax = axes[i]
            ax.plot([s[2] for s in stats], [s[3] for s in stats], "o-", linewidth=2, markersize=8, color="#1f77b4", label="Calibration Curve")
            ax.set_xlabel("Confidence", fontsize=11)
            ax.set_ylabel("Accuracy", fontsize=11)
            box = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            ax.text(0.05, 0.95, f"ECE={ece_value:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=box)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            keep = [s for s in stats if s[1] >= min_obs]
            centers = [(edges[b] + edges[b + 1]) / 2 for b, *_ in keep]
            heights = [a * 100 for *_, a in keep]
            gaps = [max(0, edges[b + 1] * 100 - a * 100) for b, _, _, a in keep]
            ax.bar(centers, heights, 0.08, label="Outputs", color="#1f77b4", edgecolor="black", linewidth=1)
            ax.bar(centers, gaps, 0.08, bottom=heights, label="Gap", color="#ffd700", edgecolor="black", linewidth=1)
            ax.plot([0, 1], [0, 100], "k--", linewidth=1.5, label="Perfect Calibration")
            ax.set_xlabel("Confidence", fontsize=12)
            ax.set_ylabel("Accuracy(%)", fontsize=12)
            ax.set_ylim(0, 100)
            ax.set_yticks(np.arange(0, 101, 10))
            ax.text(0.5, 0.05, f"ECE={ece_value:.2f}", transform=ax.transAxes, fontsize=14, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8), horizontalalignment="center")
        ax.set_xlim(0, 1)
        if style == "curve":
            ax.set_ylim(0, 1)
        ax.set_title(f"Calibration Curve: {feature}" if style == "curve" else f"Reliability Diagram for {feature}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left" if style != "curve" else "best")
        if style != "curve":
            plt.tight_layout()
            plt.show()
            print(pd.DataFrame([{"Bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "Count": n, "Mean Confidence": round(c, 3),
                                 "Accuracy": round(a, 3)} for b, n, c, a in stats]).to_string(index=False))
    if style == "curve":
        plt.tight_layout()
        plt.show()


def cn_overall_calibration(data, cube, n_bins=10):
    """Confidence histogram and calibration curve pooled over all features."""
    conf_all, acc_all = [], []
    for j, feature in enumerate(data.columns):
        if data[feature].notna().any():
            conf, acc = _cn_confidence_accuracy(data[feature].values, cube[:, j, :])
            conf_all.append(conf)
            acc_all.append(acc)
    conf_all, acc_all = np.concatenate(conf_all), np.concatenate(acc_all)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    counts, _ = np.histogram(conf_all, bins=edges)
    plt.figure(figsize=(9, 5))
    plt.bar((edges[:-1] + edges[1:]) / 2, counts, width=0.09, color="steelblue", edgecolor="black")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence distribution (all observations, all features)")
    plt.xlim(0, 1)
    plt.xticks(edges)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    bins = _cn_bins(conf_all, n_bins)
    summary = pd.DataFrame([{
        "bin": f"{edges[b]:.2f}-{edges[b + 1]:.2f}", "count": int((bins == b).sum()),
        "mean_confidence": conf_all[bins == b].mean() if (bins == b).any() else np.nan,
        "mean_accuracy": acc_all[bins == b].mean() if (bins == b).any() else np.nan,
        "std_accuracy": acc_all[bins == b].std() if (bins == b).any() else np.nan,
    } for b in range(n_bins)])
    plt.figure(figsize=(9, 6))
    plt.errorbar(summary["mean_confidence"], summary["mean_accuracy"], yerr=summary["std_accuracy"], fmt="o-", capsize=5, markersize=8, color="#1f77b4")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5)
    for _, r in summary.dropna().iterrows():
        plt.text(r["mean_confidence"], r["mean_accuracy"] - 0.05, f"n={r['count']}", ha="center", fontsize=8)
    plt.xlabel("Mean Confidence")
    plt.ylabel("Accuracy")
    plt.title("Overall calibration curve (error bars = std of accuracy in bin)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return summary.round(4)


def cn_logscore_by_confidence(data, cube, logscores, feature, n_bins=10, min_obs=10):
    """Log score of one feature by confidence bin, and its distribution."""
    conf, _ = _cn_confidence_accuracy(data[feature].values, cube[:, data.columns.get_loc(feature), :])
    ls = logscores[feature].dropna().to_numpy()
    if len(ls) != len(conf):  # rows with an unknown class have no log score
        mask = data[feature].notna().to_numpy()
        keep = logscores[feature].notna().to_numpy()[mask]
        conf = conf[keep]
    bins = _cn_bins(conf, n_bins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    stats = pd.DataFrame([{
        "Bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "center": (edges[b] + edges[b + 1]) / 2,
        "Count": int((bins == b).sum()), "Mean Conf": conf[bins == b].mean(),
        "Mean LogScore": ls[bins == b].mean(), "Std LogScore": ls[bins == b].std(),
    } for b in range(n_bins) if (bins == b).sum() >= min_obs])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.bar(stats["center"], stats["Mean LogScore"], 0.08, color="#1f77b4", edgecolor="black", linewidth=1,
            yerr=stats["Std LogScore"], capsize=3, error_kw={"linewidth": 1.5})
    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("LogScore", fontsize=12)
    ax1.set_title(f"LogScore by Confidence Bin for {feature}", fontsize=14, fontweight="bold")
    ax1.set_xlim(0, 1)
    ax1.text(0.5, 0.95, f"Overall LogScore: {ls.mean():.4f}", transform=ax1.transAxes, fontsize=12, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="grey", alpha=0.8), horizontalalignment="center", verticalalignment="top")
    ax2.hist(ls, bins=50, color="#1f77b4", edgecolor="black", alpha=0.7)
    ax2.axvline(ls.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {ls.mean():.4f}")
    ax2.set_xlabel("LogScore", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title(f"Distribution of LogScores for {feature}", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return stats.drop(columns="center")
