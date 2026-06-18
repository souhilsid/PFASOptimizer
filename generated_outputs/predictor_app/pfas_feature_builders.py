from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "generated_outputs"


def clean_text(value: Any, lower: bool = False) -> Any:
    if pd.isna(value):
        return np.nan
    text = " ".join(str(value).replace("\xa0", " ").split()).strip()
    return text.lower() if lower else text


def clean_smiles_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\xa0", " ").split()).strip()


def normalize_category(value: Any) -> str:
    text = clean_text(value, lower=True)
    if pd.isna(text) or text in {"", "nan", "lack", "none", "missing"}:
        return "unknown"
    return text


def infer_dataset1_pfas_family(smiles: str) -> str:
    if "S(=O)(=O)O" in smiles:
        return "sulfonic_acid_like"
    if "C(=O)" in smiles:
        return "carboxylic_acid_like"
    return "other_or_ether_like"


def lightweight_smiles_features(smiles_values: pd.Series) -> pd.DataFrame:
    s = smiles_values.fillna("").map(clean_smiles_text)
    out = pd.DataFrame(index=s.index)
    out["smiles_length"] = s.str.len()
    out["smiles_F_count"] = s.str.count("F")
    out["smiles_C_count"] = s.str.count("C")
    out["smiles_O_count"] = s.str.count("O")
    out["smiles_S_count"] = s.str.count("S")
    out["smiles_branch_count"] = s.str.count(r"\(")
    out["smiles_has_sulfonic"] = s.str.contains(r"S\(=O\)\(=O\)O", regex=True).astype(int)
    out["smiles_has_carboxyl"] = s.str.contains(r"C\(=O\).*O|O.*C\(=O\)", regex=True).astype(int)
    out["smiles_has_ether"] = s.str.contains(r"OC|CO", regex=True).astype(int)
    out["smiles_f_per_c"] = out["smiles_F_count"] / out["smiles_C_count"].replace(0, np.nan)
    out["smiles_f_per_c"] = out["smiles_f_per_c"].fillna(0)
    return out


def rdkit_features(
    smiles_values: pd.Series,
    *,
    prefix: str,
    n_bits: int,
    radius: int,
    fingerprint_type: str,
    include_hbd_hba: bool,
    include_availability: bool,
) -> pd.DataFrame:
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import Descriptors
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

        RDLogger.DisableLog("rdApp.warning")
    except Exception:
        descriptor_cols = ["rdkit_mw", "rdkit_logp", "rdkit_tpsa"]
        if include_hbd_hba:
            descriptor_cols += ["rdkit_hbd", "rdkit_hba"]
        if include_availability:
            descriptor_cols += ["smiles_available"]
        fp_cols = [f"{prefix}_{i:04d}" if prefix == "cmorgan" else f"{prefix}_{i:03d}" for i in range(n_bits)]
        return pd.DataFrame(np.nan, index=smiles_values.index, columns=descriptor_cols + fp_cols)

    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows = []
    for value in smiles_values.fillna("").map(clean_smiles_text):
        mol = Chem.MolFromSmiles(value) if value else None
        row: dict[str, float] = {}
        if mol is None:
            row.update({"rdkit_mw": np.nan, "rdkit_logp": np.nan, "rdkit_tpsa": np.nan})
            if include_hbd_hba:
                row.update({"rdkit_hbd": np.nan, "rdkit_hba": np.nan})
            if include_availability:
                row["smiles_available"] = 0
            fp = np.zeros(n_bits, dtype=float)
        else:
            row.update(
                {
                    "rdkit_mw": Descriptors.MolWt(mol),
                    "rdkit_logp": Descriptors.MolLogP(mol),
                    "rdkit_tpsa": Descriptors.TPSA(mol),
                }
            )
            if include_hbd_hba:
                row.update({"rdkit_hbd": Descriptors.NumHDonors(mol), "rdkit_hba": Descriptors.NumHAcceptors(mol)})
            if include_availability:
                row["smiles_available"] = 1
            bitvect = generator.GetCountFingerprint(mol) if fingerprint_type == "count" else generator.GetFingerprint(mol)
            fp = np.zeros(n_bits, dtype=float)
            DataStructs.ConvertToNumpyArray(bitvect, fp)

        fp_cols = [f"{prefix}_{i:04d}" if prefix == "cmorgan" else f"{prefix}_{i:03d}" for i in range(n_bits)]
        row.update(dict(zip(fp_cols, fp)))
        rows.append(row)
    return pd.DataFrame(rows, index=smiles_values.index)


def pfas_name_features(pfas_series: pd.Series) -> pd.DataFrame:
    s = pfas_series.fillna("").map(lambda x: clean_text(x, lower=False) or "")
    upper = s.str.upper()
    out = pd.DataFrame(index=s.index)
    out["pfas_name_length"] = s.str.len()
    out["pfas_contains_pf"] = upper.str.contains("PF").astype(int)
    out["pfas_contains_fts"] = upper.str.contains("FTS|FTSA").astype(int)
    out["pfas_contains_ftab"] = upper.str.contains("FTAB").astype(int)
    out["pfas_contains_fosa"] = upper.str.contains("FOSA|FOSAA|FBSA").astype(int)
    out["pfas_contains_genx"] = upper.str.contains("GENX").astype(int)
    out["pfas_contains_cl"] = upper.str.contains("CL").astype(int)
    out["pfas_contains_branched"] = upper.str.contains("BRANCHED").astype(int)
    out["pfas_is_sulfonate_like"] = upper.str.contains("S$|SO|FOS|FTS|PFBS|PFHXS|PFOS|PFHPS|PFPES|PFPRS").astype(int)
    out["pfas_is_carboxylate_like"] = upper.str.contains("A$|COOH|PFOA|PFBA|PFPEA|PFHXA|PFHPA|PFNA|PFDA|PFUNA|PFDOA|GENX").astype(int)
    carbon_map = {
        "PFBA": 4,
        "PFPEA": 5,
        "PFHXA": 6,
        "PFHPA": 7,
        "PFOA": 8,
        "PFNA": 9,
        "PFDA": 10,
        "PFBS": 4,
        "PFPES": 5,
        "PFHXS": 6,
        "PFHPS": 7,
        "PFOS": 8,
        "PFPRS": 3,
        "GENX": 6,
    }
    out["pfas_approx_carbon"] = np.nan
    for key, val in carbon_map.items():
        out.loc[upper.str.contains(key, regex=False), "pfas_approx_carbon"] = val
    ratio = upper.str.extract(r"(\d+)\s*:\s*(\d+)")[0]
    out.loc[ratio.notna(), "pfas_approx_carbon"] = pd.to_numeric(ratio[ratio.notna()], errors="coerce")
    out["pfas_approx_fluorinated_chain"] = out["pfas_approx_carbon"]
    out["pfas_class_inferred"] = np.select(
        [
            out["pfas_contains_genx"].eq(1),
            out["pfas_contains_ftab"].eq(1),
            out["pfas_contains_fts"].eq(1),
            out["pfas_is_sulfonate_like"].eq(1),
            out["pfas_is_carboxylate_like"].eq(1),
        ],
        ["ether_carboxylate", "zwitterionic_ftab", "fluorotelomer_sulfonate", "sulfonate_like", "carboxylate_like"],
        default="other_pfas",
    )
    return out


def _ensure_input_columns(df: pd.DataFrame, input_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in input_columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[input_columns]


def build_dataset1_table(bundle: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    target = bundle["target"]
    input_columns = bundle["input_columns"]
    data_path = ROOT / "PFAS DATASET 1" / "1-s2.0-S221334372504686X-mmc2.xlsx"
    df = pd.read_excel(data_path, sheet_name="data_original")
    df["source_row_id"] = np.arange(len(df))
    df["SMILES_clean"] = df["SMILES"].map(clean_smiles_text)
    df["pfas_family_inferred"] = df["SMILES_clean"].map(infer_dataset1_pfas_family)
    df = pd.concat([df, lightweight_smiles_features(df["SMILES_clean"])], axis=1)
    n_bits = max([int(c.rsplit("_", 1)[1]) for c in input_columns if c.startswith("morgan_")], default=-1) + 1
    if n_bits > 0:
        df = pd.concat(
            [
                df,
                rdkit_features(
                    df["SMILES_clean"],
                    prefix="morgan",
                    n_bits=n_bits,
                    radius=2,
                    fingerprint_type="binary",
                    include_hbd_hba=False,
                    include_availability=False,
                ),
            ],
            axis=1,
        )
    X = _ensure_input_columns(df, input_columns)
    y = df[target].astype(float)
    return X, y, df


def add_dataset2_smiles_lookup(df: pd.DataFrame) -> pd.DataFrame:
    lookup_path = OUTPUT_DIR / "pfas_smiles_lookup.csv"
    if not lookup_path.exists():
        out = df.copy()
        out["SMILES"] = np.nan
        return out
    lookup = pd.read_csv(lookup_path)
    lookup["PFAS"] = lookup["PFAS"].map(lambda x: clean_text(x, lower=False))
    lookup["SMILES"] = lookup["SMILES"].map(lambda x: clean_text(x, lower=False))
    return df.merge(lookup, on="PFAS", how="left")


def build_dataset2_table(bundle: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    target = bundle["target"]
    input_columns = bundle["input_columns"]
    data_path = ROOT / "PFAS DATASET 2" / "es4c14223_si_001.xlsx"
    source = pd.read_excel(data_path, sheet_name="source data")
    reference = pd.read_excel(data_path, sheet_name="reference")
    df = source.copy()
    df["source_row_id"] = np.arange(len(df))
    df["Title_filled"] = df["Title"].ffill().map(lambda x: clean_text(x, lower=False))
    ref_map = dict(zip(reference["Title"].map(lambda x: clean_text(x, lower=False)), reference["ID"]))
    df["ID_by_title"] = df["Title_filled"].map(ref_map)

    for col in ["PFAS", "Resin"]:
        df[col] = df[col].map(lambda x: clean_text(x, lower=False))
    for col in ["Solution", "Polymer_matrix ", "Porosity", "Functional group", "Resin_type"]:
        df[col] = df[col].map(normalize_category)

    df["Stirring_rate_numeric"] = pd.to_numeric(df["Stirring_rate\n(rpm)"], errors="coerce")
    df["initial_concentration_mg_L"] = pd.to_numeric(df["initial PFAS concentration\n(μg/L)"], errors="coerce") / 1000.0
    df["resin_dosage_g_L"] = pd.to_numeric(df["Mresin\n(mg/L)"], errors="coerce") / 1000.0
    df = pd.concat([df, pfas_name_features(df["PFAS"])], axis=1)
    df = add_dataset2_smiles_lookup(df)

    n_bits = max([int(c.rsplit("_", 1)[1]) for c in input_columns if c.startswith("cmorgan_")], default=-1) + 1
    if n_bits > 0:
        df = pd.concat(
            [
                df,
                rdkit_features(
                    df["SMILES"],
                    prefix="cmorgan",
                    n_bits=n_bits,
                    radius=1,
                    fingerprint_type="count",
                    include_hbd_hba=True,
                    include_availability=True,
                ),
            ],
            axis=1,
        )
    X = _ensure_input_columns(df, input_columns)
    y = df[target].astype(float)
    return X, y, df


def _coerce_default(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def build_defaults(X: pd.DataFrame) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            defaults[col] = _coerce_default(series.median())
        else:
            mode = series.dropna().mode()
            defaults[col] = _coerce_default(mode.iloc[0] if not mode.empty else "")
    return defaults


def build_options(df: pd.DataFrame, columns: list[str], limit: int = 300) -> dict[str, list[Any]]:
    options: dict[str, list[Any]] = {}
    for col in columns:
        vals = sorted([v for v in df[col].dropna().unique().tolist() if str(v).strip() != ""], key=lambda x: str(x))
        options[col] = [_coerce_default(v) for v in vals[:limit]]
    return options


def build_dataset1_row(values: dict[str, Any], defaults: dict[str, Any], input_columns: list[str]) -> pd.DataFrame:
    row = defaults.copy()
    for col, value in values.items():
        if col in row and value not in ("", None):
            row[col] = value

    smiles = clean_smiles_text(values.get("SMILES") or values.get("SMILES_clean") or row.get("SMILES_clean") or "")
    row["SMILES_clean"] = smiles
    row["pfas_family_inferred"] = infer_dataset1_pfas_family(smiles)

    tmp = pd.DataFrame({"SMILES_clean": [smiles]})
    for col, value in lightweight_smiles_features(tmp["SMILES_clean"]).iloc[0].items():
        row[col] = _coerce_default(value)

    n_bits = max([int(c.rsplit("_", 1)[1]) for c in input_columns if c.startswith("morgan_")], default=-1) + 1
    if n_bits > 0:
        fp = rdkit_features(
            tmp["SMILES_clean"],
            prefix="morgan",
            n_bits=n_bits,
            radius=2,
            fingerprint_type="binary",
            include_hbd_hba=False,
            include_availability=False,
        ).iloc[0]
        for col, value in fp.items():
            row[col] = _coerce_default(value)

    return pd.DataFrame([{col: row.get(col, np.nan) for col in input_columns}])


def build_dataset2_row(values: dict[str, Any], defaults: dict[str, Any], input_columns: list[str]) -> pd.DataFrame:
    row = defaults.copy()
    for col, value in values.items():
        if col in row and value not in ("", None):
            row[col] = value

    raw_map = {
        "initial_pfas_concentration_ug_L": ("initial_concentration_mg_L", 1 / 1000.0),
        "resin_dosage_mg_L": ("resin_dosage_g_L", 1 / 1000.0),
        "stirring_rate_rpm": ("Stirring_rate_numeric", 1.0),
        "temperature_C": ("temperature\n(℃)", 1.0),
        "contact_time_h": ("contact_time\n(h)", 1.0),
        "CDOC_mg_L": ("CDOC\n(mg/L)", 1.0),
    }
    for friendly, (col, scale) in raw_map.items():
        if values.get(friendly) not in ("", None):
            row[col] = float(values[friendly]) * scale

    for col in ["Solution", "Polymer_matrix ", "Porosity", "Functional group", "Resin_type"]:
        if col in row and row[col] is not None:
            row[col] = normalize_category(row[col])

    pfas = clean_text(values.get("PFAS") or row.get("PFAS") or "", lower=False)
    row["PFAS"] = pfas
    pfas_features = pfas_name_features(pd.Series([pfas])).iloc[0]
    for col, value in pfas_features.items():
        row[col] = _coerce_default(value)

    lookup_path = OUTPUT_DIR / "pfas_smiles_lookup.csv"
    smiles = ""
    if lookup_path.exists():
        lookup = pd.read_csv(lookup_path)
        lookup["PFAS"] = lookup["PFAS"].map(lambda x: clean_text(x, lower=False))
        match = lookup.loc[lookup["PFAS"].eq(pfas), "SMILES"]
        if not match.empty:
            smiles = clean_smiles_text(match.iloc[0])
    n_bits = max([int(c.rsplit("_", 1)[1]) for c in input_columns if c.startswith("cmorgan_")], default=-1) + 1
    if n_bits > 0:
        fp = rdkit_features(
            pd.Series([smiles]),
            prefix="cmorgan",
            n_bits=n_bits,
            radius=1,
            fingerprint_type="count",
            include_hbd_hba=True,
            include_availability=True,
        ).iloc[0]
        for col, value in fp.items():
            row[col] = _coerce_default(value)

    return pd.DataFrame([{col: row.get(col, np.nan) for col in input_columns}])


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
