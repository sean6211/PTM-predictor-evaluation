# main_script1.py
# STEP 1, STEP 2, STEP 3, STEP 4 for PTM immunopeptidomics 
# (Ligand.MHC Atlas-style exports)

from __future__ import annotations
import os
import re
import sys
import warnings
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ----------------------------- CONFIG -------------------------------- #

DATA_DIR = os.path.join("Data") # Store "Data" (string) in variable DATA_DIR

# Dictionary with keys and values mapping PTM types (string) to corresponding data file names (string)
PTM_FILES = { # Assign to variable PTM_FILES
    "phospho": "Phospho.txt",
    "acetyl": "Acetyl.txt",
    "methyl": "Methyl.txt",
    "ubiquitin": "Ubiquitin.txt",
}

# Dictionary with keys and values mapping PTM type name (normalized: lowercase) 
# to annotation token (short symbol) 
PTM_ANNOTATION_TOKEN = { 
    "phospho": "p",
    "acetyl": "ac",
    "methyl": "me",
    "ubiquitin": "ub",
    # add more if needed
}

# SET of strings listing artifact PTM types caused by mass-spec
ARTIFACT_PTMS = {
    "oxidation",
    "metox",
    "carbamidomethyl",
    "carboxyamidomethyl",
    "deamidated",
    "dioxidation",
    "ammonia loss",
    "water loss",
    "gln->pyro-glu",
    "glu->pyro-glu",
    "pyro-glu",
    "pyroglutamate",
}

LEN_MIN, LEN_MAX = 8, 11 # Each variable (LEN_MIN, LEN_MAX) is integer (8 and 11)

RANK_STRONG = float(os.getenv("RANK_STRONG", "0.5"))  # Store either RANK_STRONG variable or 0.5 integer
RANK_WEAK   = float(os.getenv("RANK_WEAK", "2.0"))    # Store either RANK_WEAK variable or 2.0 integer

MAKE_PLOTS = os.getenv("MAKE_PLOTS", "0").strip() in {"1", "true", "True"} 
    # 'in' checks if the left side is a member of right side (set containing strings)
    # Returns TRUE if left side matches any of those values
    # Stores boolean result in variable MAKE_PLOTS

# Optional external tools (set to binary paths to enable)
NETMHCPAN_BIN = os.getenv("NETMHCPAN_BIN", "").strip()     # e.g., /usr/local/bin/netMHCpan as STRING
MIXMHCPRED_BIN = os.getenv("MIXMHCPRED_BIN", "").strip()   # e.g., /usr/local/bin/MixMHCpred as STRING
MHCFLURRY_PREDICT_BIN = os.getenv("MHCFLURRY_PREDICT_BIN", "mhcflurry-predict").strip()



# ---------------------- SAVE HELPERS (Parquet fallback) --------------- #

## This is to import either PyArrow or Fastparquet library
def parquet_available() -> bool: # This function returns boolean value
    try: # Try/except block
        import pyarrow  # Attempts to import PyArrow library
        return True # If PyArrow is available, function returns True
    except Exception: # Catches exceptions that occur in above try block
        try: # 2nd Try/except block
            import fastparquet  
            return True
        except Exception:
            return False # Function returns False if neither PyArrow nor FastParquet is available

## This is for better parquet compabilitbility
def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame: # Type hint indicating that this function returns pandas DataFrame
    """Avoid Arrow extension dtype issues by ensuring plain Python/Numpy types and None for missing.""" # Docstring
    out = df.copy() # Create a copy of input DataFrame and assign in to variable "out"
    out = out.where(~out.isna(), None)  # replace pd.NA with None
    for c in out.columns:
        if pd.api.types.is_string_dtype(out[c]) or pd.api.types.is_object_dtype(out[c]):
            out[c] = out[c].astype(object) # Convert string or object columns to object type (for better parquet compabilitbility)
    return out

##
def save_table_with_fallback(df: pd.DataFrame, out_dir: str, basename: str) -> str: # Returns string
    os.makedirs(out_dir, exist_ok=True) # Create directory if it doesn't exist
    forced = os.getenv("SAVE_FORMAT", "").strip().lower()  # 'parquet' or 'csv' assigned to variable "forced"

    if forced == "csv": 
        out_path = os.path.join(out_dir, f"{basename}.csv") # Complete file path (directory, file) assigned to "out_path" variable
        df.to_csv(out_path, index=False) # Save out_path DataFrame to CSV file, not including row numbers
        return out_path 

    if forced in {"parquet", ""} and parquet_available():
        try:
            try:
                import pyarrow  # Try importing pyarrow first
                df_sanitized = _sanitize_for_parquet(df) # Convert df DataFrame to parquet-compatible format
                out_path = os.path.join(out_dir, f"{basename}.parquet") # Complete file path assigned to "out_path" variable
                df_sanitized.to_parquet(out_path, index=False) 
                return out_path
            except ImportError: # When PyArrow import fails....
                out_path = os.path.join(out_dir, f"{basename}.parquet") 
                df.to_parquet(out_path, index=False, engine="fastparquet")
                return out_path
        except Exception as e:
            print(f"[WARN] Could not write Parquet ({type(e).__name__}: {e}). Falling back to CSV.")

    out_path = os.path.join(out_dir, f"{basename}.csv")
    df.to_csv(out_path, index=False)
    return out_path

# ----------------------------- HELPERS -------------------------------- #

_ptm_site_regex = re.compile(
    r"(?P<pos>\d+)\s*,\s*(?P<type>[A-Za-z]+)\[(?P<aa>[A-Z])\]\s*#(?P<idx>\d+)",
    re.IGNORECASE,
)

def _normalize_ptm_type(s: str) -> str:
    return s.strip().lower()

def _looks_like_artifact(ptm_type_norm: str) -> bool:
    t = ptm_type_norm.lower()
    return any(a in t for a in ARTIFACT_PTMS)

def parse_ptm_sites(ptm_field: str) -> List[Dict]:
    """
    Parse a PTM field like:
      "4,Phospho[S]#0"
      "2,Phospho[T]#0; 7,Phospho[S]#1"
    Returns a list of dicts: {pos:int, type:str, aa:str, idx:int}
    """
    if pd.isna(ptm_field) or not str(ptm_field).strip():
        return []
    sites = []
    for m in _ptm_site_regex.finditer(str(ptm_field)):
        sites.append(
            {
                "pos": int(m.group("pos")),
                "type": m.group("type"),
                "aa": m.group("aa"),
                "idx": int(m.group("idx")),
            }
        )
    return sites

def annotate_sequence_with_ptm(seq: str, sites: List[Dict]) -> str:
    """
    Build an annotated sequence like A[p]STMGK for phospho on the residue at position 2.
    We insert the token *before* the modified residue: A + [p] + S ...
    If multiple sites: insert from right to left to preserve indices.
    """
    if not sites:
        return seq
    s = list(seq)
    for site in sorted(sites, key=lambda d: d["pos"], reverse=True):
        pos1 = site["pos"]
        aa_expected = site["aa"]
        ptm_type_norm = _normalize_ptm_type(site["type"])
        token = PTM_ANNOTATION_TOKEN.get(ptm_type_norm, ptm_type_norm) 
            # Dictionary (PTM_ANNOTATION_TOKEN) lookup method
            # Stored as "token" variable (string) 

        try:
            aa_actual = s[pos1 - 1]
        except IndexError:
            warnings.warn(f"PTM position {pos1} out of range for sequence '{seq}'")
            continue
        if aa_actual != aa_expected:
            warnings.warn(
                f"PTM AA mismatch at pos {pos1}: expected {aa_expected}, found {aa_actual} in '{seq}'"
            )
        s.insert(pos1 - 1, f"[{token}]")
    return "".join(s)

def load_ptm_file(path: str, ptm_type_hint: Optional[str] = None) -> pd.DataFrame:
    """
    Load a single PTM file (Ligand.MHC Atlas-style) and return the raw DataFrame.
    Expected columns:
      HLA, Immunopeptide, PTMs, Immunopeptide ID, Protein, HLA alleles
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", dtype=str)
    expected = {"HLA", "Immunopeptide", "PTMs", "Immunopeptide ID", "Protein", "HLA alleles"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")
    if ptm_type_hint:
        df["_ptm_type_hint"] = ptm_type_hint
    df["_source_path"] = path
    return df

def explode_ptm_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode multi-site PTM rows into one row per site with annotations.
    """
    rows = []
    for _, r in df.iterrows():
        base_seq = r["Immunopeptide"]
        sites = parse_ptm_sites(r["PTMs"])
        if not sites:
            continue
        for site in sites:
            ptm_type_norm = _normalize_ptm_type(site["type"])
            if _looks_like_artifact(ptm_type_norm):
                continue
            raw_seq = annotate_sequence_with_ptm(base_seq, [site])
            rows.append(
                {
                    "raw_seq": raw_seq,
                    "seq_base": base_seq,
                    "ptm_type": ptm_type_norm,
                    "ptm_residue": site["aa"],
                    "ptm_pos": int(site["pos"]),  # 1-based
                    "site_index": int(site["idx"]),
                    "length": len(base_seq),
                    "HLA_allele": r["HLA"],
                    "all_HLA_alleles": r["HLA alleles"],
                    "source_protein": r["Protein"],
                    "immunopeptide_id": r["Immunopeptide ID"],
                    # Placeholders (not present in file)
                    "cancer_type": pd.NA,
                    "sample_id": pd.NA,
                    "binder_score": pd.NA,
                    "PSM_count": pd.NA,
                    "_source_path": r["_source_path"],
                }
            )
    out = pd.DataFrame.from_records(rows)
    return out

def keep_class_I_lengths(df: pd.DataFrame, min_len: int = LEN_MIN, max_len: int = LEN_MAX) -> pd.DataFrame:
    return df.loc[(df["length"] >= min_len) & (df["length"] <= max_len)].copy()

def summarize_step1(df_all: pd.DataFrame) -> None:
    print("\n=== STEP 1 — HEAD (annotated peptides) ===")
    print(df_all[["raw_seq", "seq_base", "ptm_type", "ptm_pos", "HLA_allele", "source_protein"]].head(10))

    print("\n=== PTM type distribution (overall) ===")
    vc = df_all["ptm_type"].str.lower().value_counts(dropna=False)
    total = int(vc.sum())
    pct = (vc / total * 100).round(2)
    dist = pd.DataFrame({"count": vc, "percent": pct})
    print(dist)

    print("\n=== Length distribution (after parsing, before class I filter) ===")
    print(df_all["length"].value_counts().sort_index())

    print("\n=== Number of peptides (rows) ===")
    print(len(df_all))

def summarize_step2(df_clean: pd.DataFrame) -> None:
    print("\n=== STEP 2 — Cleaned (8–11mers, artifacts dropped) HEAD ===")
    print(df_clean[["raw_seq", "seq_base", "ptm_type", "ptm_pos", "length", "HLA_allele"]].head(10))

    print("\n=== Length distribution (8–11 only) ===")
    print(df_clean["length"].value_counts().sort_index())

    print("\nSample check (random 5): df[['raw_seq','seq_base','ptm_type','ptm_pos']].sample(5)")
    with pd.option_context('display.max_colwidth', None):
        print(df_clean[["raw_seq", "seq_base", "ptm_type", "ptm_pos"]].sample(min(5, len(df_clean)), random_state=7))

# -------------------------- STEP 3: Anchors --------------------------- #

def _normalize_allele(a: str) -> str:
    """Ensure allele has 'HLA-' prefix and no spaces, e.g., 'HLA-A*02:01'."""
    if pd.isna(a):
        return a
    x = re.sub(r"\s+", "", str(a))
    if not x.startswith("HLA-"):
        x = "HLA-" + x if not x.startswith("HLA") else x.replace("HLA", "HLA-")
    return x

def load_anchor_map_from_csv(path: str) -> Dict[Tuple[str, int], List[int]]:
    """
    Read config/anchors.csv with columns: allele,length,anchors
    - allele like HLA-A*02:01
    - length in {8,9,10,11}
    - anchors like '2,9' (1-based positions; use terminal index for PΩ)
    """
    df = pd.read_csv(path)
    required = {"allele", "length", "anchors"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"anchors.csv missing columns: {missing}")

    amap: Dict[Tuple[str, int], List[int]] = {}
    for _, r in df.iterrows():
        allele = _normalize_allele(r["allele"])
        length = int(r["length"])
        anchors_str = str(r["anchors"])
        positions = [int(p.strip()) for p in anchors_str.split(",") if str(p).strip().isdigit()]
        amap[(allele, length)] = sorted(set(positions))
    return amap

def default_anchor_positions(allele: str, length: int) -> List[int]:
    """Default anchors at P2 and PΩ (terminal). Override via CSV for allele-specific rules."""
    return [2, length]

def anchors_for(allele: str, length: int, anchor_map: Dict[Tuple[str, int], List[int]]) -> Tuple[List[int], str]:
    key = (_normalize_allele(allele), int(length))
    if key in anchor_map:
        return anchor_map[key], "config"
    return default_anchor_positions(allele, length), "default"

def omega_str(pos: int, length: int) -> str:
    """Format position list replacing terminal index with Ω."""
    return "Ω" if pos == length else str(pos)

def classify_ptm_position(ptm_pos: int, length: int, anchor_positions: List[int]) -> str:
    if ptm_pos in anchor_positions:
        return "anchor"
    if ptm_pos == 1:
        return "N-flank"
    if ptm_pos == length:
        return "C-flank"
    return "TCR-facing"

def apply_anchor_mapping(df: pd.DataFrame, anchor_map: Dict[Tuple[str, int], List[int]]) -> pd.DataFrame:
    """
    Add anchor-aware columns:
      - anchor_positions (string, e.g., '2, Ω')
      - ptm_is_anchor (bool)
      - ptm_position_class ∈ {anchor, TCR-facing, N-flank, C-flank}
      - anchor_source ∈ {config, default}
    """
    ap_src, ap_str, cls, is_anchor = [], [], [], []
    for _, r in df.iterrows():
        length = int(r["length"])
        allele = r["HLA_allele"]
        ptm_pos = int(r["ptm_pos"])
        anchors, src = anchors_for(allele, length, anchor_map)
        ap_src.append(src)
        ap_str.append(", ".join(omega_str(p, length) for p in anchors))
        cls_val = classify_ptm_position(ptm_pos, length, anchors)
        cls.append(cls_val)
        is_anchor.append(cls_val == "anchor")
    out = df.copy()
    out["anchor_positions"] = ap_str
    out["anchor_source"] = ap_src
    out["ptm_position_class"] = cls
    out["ptm_is_anchor"] = is_anchor
    return out

def summarize_step3(df_anchors: pd.DataFrame) -> None:
    print("\n=== STEP 3 — Anchor-aware summary ===")
    print("Position class distribution:")
    print(df_anchors["ptm_position_class"].value_counts())
    print("\nAnchor source (config vs default):")
    print(df_anchors["anchor_source"].value_counts())
    print("\nSample rows with anchors:")
    cols = ["raw_seq", "seq_base", "length", "HLA_allele", "ptm_pos", "anchor_positions", "ptm_position_class", "ptm_is_anchor"]
    print(df_anchors[cols].head(10))

def plot_anchor_position_histogram(df: pd.DataFrame, allele: str, length: int, out_dir: str) -> Optional[str]:
    """Save a simple histogram of PTM positions for a specific (allele, length) with anchor lines."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] Matplotlib not available for plotting: {e}")
        return None

    os.makedirs(out_dir, exist_ok=True)
    allele_norm = _normalize_allele(allele)
    sub = df[(df["HLA_allele"].apply(_normalize_allele) == allele_norm) & (df["length"] == length)]
    if sub.empty:
        print(f"[INFO] No data for plotting {allele} length {length}.")
        return None

    anchors_str = sub["anchor_positions"].iloc[0]
    anchors = []
    for tok in anchors_str.split(","):
        tok = tok.strip()
        anchors.append(length if tok in {"Ω", "omega", "Omega"} else int(tok))

    counts = sub["ptm_pos"].value_counts().sort_index()
    positions = list(range(1, length + 1))
    heights = [counts.get(p, 0) for p in positions]

    fig = plt.figure(figsize=(8, 4.2))
    plt.bar(positions, heights)
    for a in anchors:
        plt.axvline(a, linestyle="--")
    plt.xticks(positions, [omega_str(p, length) for p in positions])
    plt.xlabel("Peptide position (P1 … PΩ)")
    plt.ylabel("# PTM sites")
    plt.title(f"PTM positions for {allele_norm} {length}-mers\n(anchors: {anchors_str})")
    plt.tight_layout()

    fname = f"ptm_positions_{allele_norm.replace('*','').replace(':','').replace('HLA-','HLA')}_{length}mer.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

# -------------------------- STEP 4: Prediction ------------------------- #

def allele_to_netmhcpan(allele: str) -> str:
    """Convert 'HLA-A*02:01' to a common netMHCpan format 'HLA-A02:01'."""
    a = _normalize_allele(allele)  # e.g., HLA-A*02:01
    return a.replace("*", "")      # → HLA-A02:01

def classify_from_rank(rank_percent: float) -> str:
    """Strong/weak/non-binder classification based on percentile rank."""
    if pd.isna(rank_percent):
        return "unknown"
    try:
        r = float(rank_percent)
    except Exception:
        return "unknown"
    if r <= RANK_STRONG:
        return "strong-binder"
    if r <= RANK_WEAK:
        return "weak-binder"
    return "non-binder"

# ---- Fallback: parse Atlas HLA-*.txt predictions if present ----------- #

_binding_info_re = re.compile(
    r"\s*(NetMHCpan|MixMHCpred|MHCflurry)\s*\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)\s*",
    re.IGNORECASE,
)

def parse_binding_info_field(s: str) -> Dict[str, float]:
    """
    Parse 'Binding information' like:
      'NetMHCpan (0.668, 0.25)::MixMHCpred (0.126144, 1)::MHCflurry (0.044, 1)'
    Return dict with ranks by tool: {'netmhcpan_rank': 0.668, 'mixmhcpred_rank': 0.126144, 'mhcflurry_rank': 0.044}
    """
    out: Dict[str, float] = {}
    if pd.isna(s) or not str(s).strip():
        return out
    for seg in str(s).split("::"):
        m = _binding_info_re.match(seg.strip())
        if not m:
            continue
        tool, rank_str, _flag = m.groups()
        key = tool.strip().lower()
        try:
            out[f"{key}_rank"] = float(rank_str)
        except Exception:
            pass
    return out

def load_atlas_predictions_from_hla_files(data_dir: str) -> pd.DataFrame:
    """Harvest predictions from any Data/HLA-*.txt files if you have them."""
    if not os.path.isdir(data_dir):
        return pd.DataFrame()
    frames = []
    for fname in os.listdir(data_dir):
        if not (fname.startswith("HLA-") and fname.endswith(".txt")):
            continue
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path, sep="\t", dtype=str)
        except Exception:
            continue
        required = {"HLA", "Immunopeptide", "Binding information"}
        if not required.issubset(df.columns):
            continue
        parsed = df["Binding information"].apply(parse_binding_info_field)
        dfp = pd.DataFrame(parsed.tolist())
        keep_cols = [c for c in ["netmhcpan_rank", "mixmhcpred_rank", "mhcflurry_rank"] if c in dfp.columns]
        if not keep_cols:
            continue
        tmp = pd.concat(
            [df[["HLA", "Immunopeptide"]].rename(columns={"HLA": "HLA_allele", "Immunopeptide": "seq_base"}),
             dfp[keep_cols]],
            axis=1
        )
        frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    allp = pd.concat(frames, ignore_index=True)
    agg_dict = {c: "min" for c in allp.columns if c.endswith("_rank")}
    allp_best = allp.groupby(["seq_base", "HLA_allele"], as_index=False).agg(agg_dict)
    for tool in ["netmhcpan", "mixmhcpred", "mhcflurry"]:
        col = f"{tool}_rank"
        if col in allp_best.columns:
            allp_best[f"{tool}_binder_class"] = allp_best[col].apply(classify_from_rank)
    return allp_best

# --------------------- Tool runners + diagnostics ---------------------- #

def run_mhcflurry_predict(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Try MHCflurry (Python). If unavailable, we raise to let caller try CLI fallback.
    """
    from mhcflurry import Class1BindingPredictor  # may raise ImportError / AttributeError if env mismatch
    pep = pairs_df[["seq_base", "HLA_allele"]].drop_duplicates().reset_index(drop=True)
    predictor = Class1BindingPredictor.load()  # uses installed models
    preds = predictor.predict(peptides=pep["seq_base"].tolist(),
                              alleles=pep["HLA_allele"].tolist(),
                              verbose=False)
    out = pd.DataFrame({
        "seq_base": pep["seq_base"].values,
        "HLA_allele": pep["HLA_allele"].values,
        "mhcflurry_affinity_nM": preds["affinity"].values,
        "mhcflurry_rank": preds["percentile_rank"].values
    })
    out["mhcflurry_binder_class"] = out["mhcflurry_rank"].apply(classify_from_rank)
    return out

def _print_cli_failure(prefix: str, e: subprocess.CalledProcessError) -> None:
    print(f"[{prefix}] returncode={e.returncode}")
    if e.stdout:
        print(f"[{prefix}] STDOUT:\n{e.stdout}")
    if e.stderr:
        print(f"[{prefix}] STDERR:\n{e.stderr}")

def mhcflurry_cli_selftest(bin_path: Optional[str]) -> bool:
    """Run a tiny self-test to surface clear CLI errors before the big job."""
    import shutil
    exe = shutil.which(bin_path or MHCFLURRY_PREDICT_BIN) or (bin_path if bin_path and Path(bin_path).exists() else None)
    if not exe:
        print("[INFO] MHCflurry CLI not found in PATH; self-test skipped.")
        return False
    test = pd.DataFrame({
        "peptide": ["SIINFEKL", "GLCTLVAML", "GILGFVFTL"],
        "allele": ["HLA-A*02:01", "HLA-A*02:01", "HLA-A*02:01"],
    })
    try:
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "t.csv"
            outp = Path(td) / "o.csv"
            test.to_csv(inp, index=False)
            cmd = [exe, "--input", str(inp), "--out", str(outp)]
            res = subprocess.run(cmd, check=True, text=True, capture_output=True)
            # Verify output file exists and has expected columns
            df = pd.read_csv(outp)
            ok_cols = set(c.lower() for c in df.columns)
            if not (("peptide" in ok_cols) and ("allele" in ok_cols) and (("percentile_rank" in ok_cols) or ("rank" in ok_cols) or ("percentile" in ok_cols))):
                print("[INFO] MHCflurry CLI self-test ran but output columns look unexpected; proceeding anyway.")
            print("[INFO] MHCflurry CLI self-test: OK")
            return True
    except subprocess.CalledProcessError as e:
        _print_cli_failure("MHCflurry-CLI-SELFTEST", e)
    except Exception as e:
        print(f"[MHCflurry-CLI-SELFTEST] Unexpected error: {type(e).__name__}: {e}")
    return False

def run_mhcflurry_cli(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    MHCflurry CLI fallback using `mhcflurry-predict` (or MHCFLURRY_PREDICT_BIN).
    We feed a CSV with columns 'peptide','allele'. Output is parsed for 'affinity' and 'percentile_rank'.
    """
    import shutil
    bin_path = shutil.which(MHCFLURRY_PREDICT_BIN) or (MHCFLURRY_PREDICT_BIN if Path(MHCFLURRY_PREDICT_BIN).exists() else None)
    if not bin_path:
        print("[INFO] MHCflurry CLI not found. Skipping MHCflurry CLI.")
        return pd.DataFrame(columns=["seq_base", "HLA_allele", "mhcflurry_affinity_nM", "mhcflurry_rank", "mhcflurry_binder_class"])

    # Self-test before full run
    mhcflurry_cli_selftest(bin_path)

    df_in = pairs_df[["seq_base", "HLA_allele"]].drop_duplicates().rename(columns={"seq_base": "peptide", "HLA_allele": "allele"})
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "input.csv"
        outp = Path(td) / "output.csv"
        df_in.to_csv(inp, index=False)
        cmd = [bin_path, "--input", str(inp), "--out", str(outp)]
        try:
            subprocess.run(cmd, check=True, text=True, capture_output=True)
            df_out = pd.read_csv(outp)
        except subprocess.CalledProcessError as e:
            _print_cli_failure("MHCflurry-CLI", e)
            return pd.DataFrame(columns=["seq_base", "HLA_allele", "mhcflurry_affinity_nM", "mhcflurry_rank", "mhcflurry_binder_class"])
        except FileNotFoundError:
            print("[WARN] mhcflurry-predict output file not found.")
            return pd.DataFrame(columns=["seq_base", "HLA_allele", "mhcflurry_affinity_nM", "mhcflurry_rank", "mhcflurry_binder_class"])

    # Robust column grabbing
    def pick(colnames: List[str], opts: List[str]) -> Optional[str]:
        for o in opts:
            if o in colnames:
                return o
        lower = {c.lower(): c for c in colnames}
        for o in opts:
            if o.lower() in lower:
                return lower[o.lower()]
        return None

    cols = list(df_out.columns)
    pep_col = pick(cols, ["peptide", "Peptide"])
    allele_col = pick(cols, ["allele", "Allele"])
    aff_col = pick(cols, ["affinity", "Affinity", "predicted_affinity", "prediction_affinity"])
    rank_col = pick(cols, ["percentile_rank", "Percentile_Rank", "percentile", "rank"])

    if pep_col is None or allele_col is None:
        print("[WARN] mhcflurry-predict output missing peptide/allele columns; skipping.")
        return pd.DataFrame(columns=["seq_base", "HLA_allele", "mhcflurry_affinity_nM", "mhcflurry_rank", "mhcflurry_binder_class"])

    out = pd.DataFrame({
        "seq_base": df_out[pep_col],
        "HLA_allele": df_out[allele_col],
        "mhcflurry_affinity_nM": df_out[aff_col] if aff_col else None,
        "mhcflurry_rank": df_out[rank_col] if rank_col else None,
    })
    out["mhcflurry_binder_class"] = out["mhcflurry_rank"].apply(classify_from_rank)
    return out

def run_netmhcpan_predict(pairs_df: pd.DataFrame) -> pd.DataFrame:
    if not NETMHCPAN_BIN:
        print("[INFO] NETMHCPAN_BIN not set. Skipping NetMHCpan.")
        return pd.DataFrame(columns=["seq_base", "HLA_allele", "netmhcpan_affinity_nM", "netmhcpan_rank", "netmhcpan_binder_class"])

    rows = []
    for allele, sub in pairs_df[["seq_base", "HLA_allele"]].drop_duplicates().groupby("HLA_allele"):
        pep_list = sub["seq_base"].tolist()
        if not pep_list:
            continue
        with tempfile.TemporaryDirectory() as td:
            in_path = Path(td) / "peptides.txt"
            with open(in_path, "w") as f:
                for p in pep_list:
                    f.write(p + "\n")
            allele_fmt = allele_to_netmhcpan(allele)
            cmd = [NETMHCPAN_BIN, "-p", str(in_path), "-a", allele_fmt, "-BA"]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                out = res.stdout.splitlines()
            except subprocess.CalledProcessError as e:
                print(f"[WARN] netMHCpan failed for {allele}: {e}")
                if e.stderr:
                    print(f"[netMHCpan STDERR]\n{e.stderr}")
                continue

            for line in out:
                if not line or line.startswith("#"):
                    continue
                if "Pos" in line and "Peptide" in line and "Rank" in line:
                    continue
                toks = line.split()
                if len(toks) < 5:
                    continue
                peptide_tok = None
                for t in toks:
                    if re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{8,15}", t):
                        peptide_tok = t
                        break
                if peptide_tok is None:
                    continue
                rank_val = None
                aff_val = None
                floats = []
                for t in toks:
                    try:
                        floats.append(float(t.strip("%")))
                    except Exception:
                        pass
                if floats:
                    rank_val = floats[-1]
                    if len(floats) >= 2:
                        aff_val = floats[-2]
                rows.append({
                    "seq_base": peptide_tok,
                    "HLA_allele": allele,
                    "netmhcpan_affinity_nM": aff_val,
                    "netmhcpan_rank": rank_val,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["netmhcpan_binder_class"] = df["netmhcpan_rank"].apply(classify_from_rank)
    else:
        df = pd.DataFrame(columns=["seq_base", "HLA_allele", "netmhcpan_affinity_nM", "netmhcpan_rank", "netmhcpan_binder_class"])
    return df

def run_mixmhcpred(pairs_df: pd.DataFrame) -> pd.DataFrame:
    if not MIXMHCPRED_BIN:
        print("[INFO] MIXMHCPRED_BIN not set. Skipping MixMHCpred.")
        return pd.DataFrame(columns=["seq_base", "HLA_allele", "mixmhcpred_affinity_nM", "mixmhcpred_rank", "mixmhcpred_binder_class"])

    rows = []
    for allele, sub in pairs_df[["seq_base", "HLA_allele"]].drop_duplicates().groupby("HLA_allele"):
        pep_list = sub["seq_base"].tolist()
        if not pep_list:
            continue
        with tempfile.TemporaryDirectory() as td:
            in_path = Path(td) / "peptides.txt"
            with open(in_path, "w") as f:
                for p in pep_list:
                    f.write(p + "\n")
            out_path = Path(td) / "out.txt"
            cmd = [MIXMHCPRED_BIN, "-a", allele, "-i", str(in_path), "-o", str(out_path)]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                lines = Path(out_path).read_text().splitlines()
            except subprocess.CalledProcessError as e:
                print(f"[WARN] MixMHCpred failed for {allele}: {e}")
                if e.stderr:
                    print(f"[MixMHCpred STDERR]\n{e.stderr}")
                continue
            except FileNotFoundError:
                print(f"[WARN] MixMHCpred output not found for {allele}.")
                continue

            header_idx = None
            for i, line in enumerate(lines):
                if "Peptide" in line and ("Rank" in line or "Percentile" in line):
                    header_idx = i
                    break
            data_lines = lines[header_idx + 1:] if header_idx is not None else []
            for line in data_lines:
                toks = re.split(r"\s+", line.strip())
                if len(toks) < 2:
                    continue
                peptide_tok = toks[0]
                rank_val = None
                for t in toks[1:]:
                    try:
                        rank_val = float(t.strip("%"))
                        break
                    except Exception:
                        continue
                rows.append({
                    "seq_base": peptide_tok,
                    "HLA_allele": allele,
                    "mixmhcpred_affinity_nM": None,
                    "mixmhcpred_rank": rank_val,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["mixmhcpred_binder_class"] = df["mixmhcpred_rank"].apply(classify_from_rank)
    else:
        df = pd.DataFrame(columns=["seq_base", "HLA_allele", "mixmhcpred_affinity_nM", "mixmhcpred_rank", "mixmhcpred_binder_class"])
    return df

def build_consensus(row: pd.Series) -> str:
    """
    Consensus rule:
      - if any tool says strong → strong-binder
      - else if any tool says weak → weak-binder
      - else if at least one tool gave a prediction and all non/unknown → non-binder
      - else unknown
    """
    calls = []
    for col in ["netmhcpan_binder_class", "mixmhcpred_binder_class", "mhcflurry_binder_class"]:
        v = row.get(col)
        if pd.isna(v) or v is None:
            continue
        calls.append(str(v))
    if not calls:
        return "unknown"
    if any(c == "strong-binder" for c in calls):
        return "strong-binder"
    if any(c == "weak-binder" for c in calls):
        return "weak-binder"
    if any(c == "non-binder" for c in calls):
        return "non-binder"
    return "unknown"

def run_predictions_step4(ptm_anchor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the STEP 3 table, run predictions on unique (seq_base, HLA_allele) pairs.
    Merge predictions back to the PTM rows.
    """
    pairs = ptm_anchor_df[["seq_base", "HLA_allele"]].drop_duplicates().reset_index(drop=True)

    dfs = []
    # NetMHCpan (optional)
    try:
        df_net = run_netmhcpan_predict(pairs)
        if not df_net.empty:
            dfs.append(df_net)
    except Exception as e:
        print(f"[WARN] NetMHCpan wrapper error: {type(e).__name__}: {e}")

    # MixMHCpred (optional)
    try:
        df_mix = run_mixmhcpred(pairs)
        if not df_mix.empty:
            dfs.append(df_mix)
    except Exception as e:
        print(f"[WARN] MixMHCpred wrapper error: {type(e).__name__}: {e}")

    # MHCflurry (Python → CLI fallback)
    try:
        df_flurry = run_mhcflurry_predict(pairs)
        if not df_flurry.empty:
            dfs.append(df_flurry)
    except Exception as e:
        print(f"[INFO] MHCflurry Python not available ({type(e).__name__}: {e}). Trying CLI...")
        try:
            df_flurry_cli = run_mhcflurry_cli(pairs)
            if not df_flurry_cli.empty:
                dfs.append(df_flurry_cli)
        except Exception as e2:
            print(f"[WARN] MHCflurry CLI wrapper error: {type(e2).__name__}: {e2}")

    # Atlas fallback (only if HLA-*.txt files exist)
    try:
        df_atlas = load_atlas_predictions_from_hla_files(DATA_DIR)
        if not df_atlas.empty:
            print(f"[INFO] Using atlas HLA-*.txt predictions for {len(df_atlas)} (seq,allele) pairs.")
            dfs.append(df_atlas)
        else:
            print("[INFO] No Data/HLA-*.txt files with 'Binding information' found; atlas fallback skipped.")
    except Exception as e:
        print(f"[WARN] Atlas predictions parsing failed: {type(e).__name__}: {e}")

    if dfs:
        pred_pairs = pairs.copy()
        for d in dfs:
            pred_pairs = pred_pairs.merge(d, on=["seq_base", "HLA_allele"], how="left")
        pred_pairs["consensus_binder_class"] = pred_pairs.apply(build_consensus, axis=1)
    else:
        pred_pairs = pairs.copy()
        pred_pairs["consensus_binder_class"] = "unknown"

    merged = ptm_anchor_df.merge(pred_pairs, on=["seq_base", "HLA_allele"], how="left")
    return merged

def summarize_step4(df_pred: pd.DataFrame) -> None:
    print("\n=== STEP 4 — Prediction summary ===")
    for tool in ["netmhcpan", "mixmhcpred", "mhcflurry"]:
        col = f"{tool}_binder_class"
        if col in df_pred.columns:
            c = df_pred[col].value_counts(dropna=True)
            if not c.empty:
                print(f"{tool} classes:\n{c}\n")
    print("Consensus classes:")
    print(df_pred["consensus_binder_class"].value_counts())

    # Checkpoint: show some non-binders (consensus)
    print("\nCheckpoint: empirically observed ligands predicted as non-binders (consensus):")
    cols_show = [
        "raw_seq", "seq_base", "length", "HLA_allele", "ptm_pos", "ptm_position_class",
        "netmhcpan_rank", "netmhcpan_binder_class",
        "mixmhcpred_rank", "mixmhcpred_binder_class",
        "mhcflurry_rank", "mhcflurry_binder_class",
        "consensus_binder_class",
    ]
    cols_show = [c for c in cols_show if c in df_pred.columns]
    print(df_pred[df_pred["consensus_binder_class"] == "non-binder"][cols_show].head(10))

# ----------------------------- MAIN ----------------------------------- #

def main():
    # Collect available PTM files in the Data/ folder
    loaded_frames: List[pd.DataFrame] = []
    for ptm_key, fname in PTM_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            print(f"Loading: {path}")
            loaded_frames.append(load_ptm_file(path, ptm_type_hint=ptm_key))

    if not loaded_frames:
        print(f"No PTM files found in '{DATA_DIR}'. Expected one of: {list(PTM_FILES.values())}")
        sys.exit(1)

    raw_df = pd.concat(loaded_frames, ignore_index=True)

    # Explode to one row per PTM site; build annotated sequences
    ptm_df = explode_ptm_rows(raw_df)
    if ptm_df.empty:
        print("After parsing, no PTM rows remained (check PTM parsing or artifact filters).")
        sys.exit(1)

    ptm_df["length"] = ptm_df["length"].astype(int)
    ptm_df["HLA_allele"] = ptm_df["HLA_allele"].str.replace(r"\s+", "", regex=True)

    # ----------------- STEP 1: Exploration ----------------- #
    summarize_step1(ptm_df)

    # ----------------- STEP 2: Cleaning & Structuring ----------------- #
    ptm_clean = keep_class_I_lengths(ptm_df, LEN_MIN, LEN_MAX)
    ptm_clean = ptm_clean.drop_duplicates(
        subset=["seq_base", "ptm_pos", "ptm_type", "HLA_allele", "immunopeptide_id"]
    ).reset_index(drop=True)

    cols_out = [
        "raw_seq", "seq_base", "ptm_type", "ptm_pos", "source_protein", "HLA_allele",
        "cancer_type", "sample_id", "binder_score", "PSM_count", "length",
        "ptm_residue", "immunopeptide_id", "all_HLA_alleles", "_source_path",
    ]
    for c in cols_out:
        if c not in ptm_clean.columns:
            ptm_clean[c] = pd.NA
    ptm_clean = ptm_clean[cols_out]

    summarize_step2(ptm_clean)

    # ----------------- STEP 3: Anchor-aware mapping ------------------- #
    anchor_map: Dict[Tuple[str, int], List[int]] = {}
    cfg_path = os.path.join("config", "anchors.csv")
    if os.path.exists(cfg_path):
        print(f"Loading anchor map from {cfg_path}")
        try:
            anchor_map = load_anchor_map_from_csv(cfg_path)
        except Exception as e:
            print(f"[WARN] Failed to load anchors.csv ({type(e).__name__}: {e}). Using defaults.")
            anchor_map = {}
    else:
        print("No config/anchors.csv found. Using default anchors (P2 & PΩ).")

    ptm_anchor = apply_anchor_mapping(ptm_clean, anchor_map)
    summarize_step3(ptm_anchor)

    if MAKE_PLOTS:
        figures_dir = os.path.join("results", "figures")
        for allele in ["HLA-A*02:01", "HLA-B*07:02", "HLA-C*07:01"]:
            for L in [8, 9, 10, 11]:
                p = plot_anchor_position_histogram(ptm_anchor, allele, L, figures_dir)
                if p:
                    print(f"[PLOT] Saved: {p}")

    # ----------------- STEP 4: Prediction (PTM-stripped) --------------- #
    print("\nRunning STEP 4: predictions on stripped peptides...")
    df_pred = run_predictions_step4(ptm_anchor)
    summarize_step4(df_pred)

    # --------------- Save with fallback/sanitizing ----------------
    out_dir = os.path.join("data", "processed")
    out_path = save_table_with_fallback(df_pred, out_dir, "ptm_peptides_step1_2_3_4")
    print(f"\nSaved table with predictions → {out_path}")
    print("Tip: set SAVE_FORMAT=parquet or SAVE_FORMAT=csv to force output format.")
    if MAKE_PLOTS:
        print("Tip: set MAKE_PLOTS=1 to write PTM-position histograms.")

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 100)
    main()
