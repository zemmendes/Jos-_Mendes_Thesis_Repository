"""Paths, omic definitions, file naming and options shared by every notebook.

Nothing here computes anything. Change a path or an option here (or override it from a
notebook with ``cfg.OPTION = value`` before running) instead of editing notebook code.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
# Folder holding datasets/, imputed_datasets/, cube_datasets/, MC_dropout_*/, sharpness/, ...
# Default: data/ at the repository root; set DISSERTACAO_DATA_ROOT to use another location.
DATA_ROOT = Path(os.environ.get(
    "DISSERTACAO_DATA_ROOT",
    Path(__file__).resolve().parents[1] / "data",
))
# Where metric CSVs are written/read. Same as DATA_ROOT unless redirected with DISSERTACAO_OUTPUT_ROOT.
OUTPUT_ROOT = Path(os.environ.get("DISSERTACAO_OUTPUT_ROOT", DATA_ROOT))

# DepMap ModelID -> SangerModelID table (used for crisprcas9 and transcriptomics).
MODEL_MAP_PATH = Path(os.environ.get("DISSERTACAO_MODEL_MAP", DATA_ROOT / "SIDM_info" / "Model.csv"))
TISSUE_INFO_PATH = DATA_ROOT / "SIDM_info" / "model_list_20230505.csv"
ESSENTIAL_GENES_PATH = DATA_ROOT / "Genes" / "EssentialGenes.csv"
NON_ESSENTIAL_GENES_PATH = DATA_ROOT / "Genes" / "NonessentialGenes.csv"

# ---------------------------------------------------------------------------
# Omics
# ---------------------------------------------------------------------------
OMICS = ["metabolomics", "drugresponse", "methylation", "proteomics", "transcriptomics", "crisprcas9", "copynumber"]
CONTINUOUS = OMICS[:6]
FULL_NAMES = {
    "metabolomics": "Metabolomics",
    "drugresponse": "Drug Response",
    "methylation": "Methylation",
    "proteomics": "Proteomics",
    "transcriptomics": "Transcriptomics",
    "crisprcas9": "CRISPR-Cas9",
    "copynumber": "Copy Number",
}
# Omics whose ground truth has missing values worth analysing.
OMICS_WITH_MISSING = ["proteomics", "metabolomics", "drugresponse"]

METHODS = ("latent", "mcd")   # latent sampler | MC Dropout
SETTINGS = ("IS", "OOD")      # in-sample | held-out (out-of-sample) rows

CN_CLASSES = (-2, -1, 0, 1, 2)
N_ENSEMBLE = 100              # samples drawn per datapoint for MC Dropout

# How each raw file in datasets/ is laid out, and omic-specific preprocessing.
#   transpose:       raw file is features x samples (True) or samples x features (False)
#   depmap_ids:      samples are DepMap ACH- IDs and must be mapped to Sanger SIDM IDs
#   crispr_scaling:  apply the essential/non-essential gene scaling
OMIC_SPECS = {
    "metabolomics":    dict(transpose=True,  depmap_ids=False, crispr_scaling=False),
    "drugresponse":    dict(transpose=True,  depmap_ids=False, crispr_scaling=False),
    "methylation":     dict(transpose=True,  depmap_ids=False, crispr_scaling=False),
    "proteomics":      dict(transpose=True,  depmap_ids=False, crispr_scaling=False),
    "transcriptomics": dict(transpose=True,  depmap_ids=True,  crispr_scaling=False),
    "crisprcas9":      dict(transpose=False, depmap_ids=True,  crispr_scaling=True),
    "copynumber":      dict(transpose=False, depmap_ids=False, crispr_scaling=False),
}

# ---------------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------------
# Optional {(omic, method): suffix} added to the names of the OOD prediction file and held-out row list, for when
# the two methods use different out-of-sample model runs, e.g. {("proteomics", "mcd"): "_run2"}.
OOD_FILE_SUFFIX = {}
# Optional {(omic, method): (file, column)} for OOD prediction files with numbered instead of named columns: the
# numbers are row labels of `file` (e.g. the model's training input) and `column` holds the feature names.
OOD_COLUMN_LABELS = {}


def input_paths(omic, method, setting):
    """Files needed to evaluate `omic` for one method/setting combination."""
    _check(omic, method, setting)
    d = DATA_ROOT
    paths = {"ground_truth": d / "datasets" / f"{omic}.csv"}

    if setting == "IS":
        paths["imputed"] = d / "imputed_datasets" / f"imputed_{omic}.csv.gz"
        if method == "latent":
            paths["cube"] = d / "cube_datasets" / f"generated_{omic}_cube.npy"
        else:
            paths["mcd_mean"] = d / "MC_dropout_mean" / f"mc_dropout_data_mean_{omic}.csv.gz"
            paths["mcd_var"] = d / "MC_dropout_var" / f"mc_dropout_data_var_{omic}.csv.gz"
    else:
        suffix = OOD_FILE_SUFFIX.get((omic, method), "")
        paths["imputed"] = d / "imputed_removed_rows" / f"imputed_{omic}_missing{suffix}.csv.gz"
        paths["imputed_reference"] = d / "imputed_datasets" / f"imputed_{omic}.csv.gz"  # column labels
        paths["held_out_rows"] = d / "Removed_rows" / f"{omic}_missing_rows_indices{suffix}.csv"
        if method == "latent":
            paths["cube"] = d / "imputed_removed_rows" / f"{omic}_cube_missing.npy"
        else:
            paths["mcd_mean"] = d / "MC_dropout_OOD" / f"mc_dropout_data_mean_{omic}.csv.gz"
            paths["mcd_var"] = d / "MC_dropout_OOD" / f"mc_dropout_data_var_{omic}.csv.gz"
    return paths


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------
# metric -> (folder, file stem); {omic} is filled in.
OUTPUT_FILES = {
    "sharpness":         ("sharpness", "{omic}_sharpness"),
    "calibration":       ("calibration", "{omic}_calibration"),
    "crps":              ("Scoring", "{omic}_CRPS"),
    "feature_rmse":      ("Error", "{omic}_feature_rmse"),
    "sample_rmse":       ("Error", "{omic}_sample_rmse"),
    "feature_pearson":   ("Pearson_Corr/{omic}", "{omic}_feature_pearson"),
    "sample_pearson":    ("Pearson_Corr/{omic}", "{omic}_sample_pearson"),
    "logscore":          ("Scoring", "copy_number_logscore"),
    "balanced_accuracy": ("Error", "{omic}_feature_balanced_accuracy"),
}


def suffix(method, setting):
    """'' | '_MCD'  followed by  '' | '_OOD'  (e.g. '_MCD_OOD')."""
    return ("_MCD" if method == "mcd" else "") + ("_OOD" if setting == "OOD" else "")


def output_path(metric, omic, method, setting, root=None):
    _check(omic, method, setting)
    folder, stem = OUTPUT_FILES[metric]
    root = Path(root) if root is not None else OUTPUT_ROOT
    return root / folder.format(omic=omic) / f"{stem.format(omic=omic)}{suffix(method, setting)}.csv"


def _check(omic, method, setting):
    if omic not in OMICS:
        raise ValueError(f"Unknown omic {omic!r}; choose from {OMICS}")
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    if setting not in SETTINGS:
        raise ValueError(f"Unknown setting {setting!r}; choose from {SETTINGS}")


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
MCD_POINT_ESTIMATE = "mc_mean"         # MC Dropout point prediction: "mc_mean" = MC Dropout mean (rounded to a class
                                       # for copy number); "imputed" = the same prediction file as latent sampling
RNG_SEED = 100                          # MC Dropout sampling seed (None = unseeded)
SAVE_OUTPUTS = True                    # write metric CSVs
DEBUG_N_FEATURES = None                # e.g. 100 to run on the first 100 features only (never saves)
