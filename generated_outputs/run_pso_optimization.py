from __future__ import annotations

import json
import gzip
import pickle
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from predictor_app.pfas_feature_builders import build_dataset1_row, build_dataset2_row

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", category=FutureWarning)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "generated_outputs"
PKL_DIR = OUTPUT_DIR / "pkl_models"
APP_ASSETS_PATH = OUTPUT_DIR / "predictor_app" / "app_assets.json"
OPT_DIR = OUTPUT_DIR / "optimization"

PARTICLE_SIZE = 48
ITERATIONS = 80
INERTIA = 0.72
COGNITIVE = 1.45
SOCIAL = 1.45
RANGE_MODE = "p05_p95"
INVALID_CATEGORY_VALUES = {"", "nan", "none", "unknown", "lack", "missing"}


@dataclass(frozen=True)
class CategoricalDimension:
    name: str
    choices: tuple[Any, ...]


@dataclass(frozen=True)
class NumericDimension:
    name: str
    lower: float
    upper: float
    source_lower: float
    source_upper: float


@dataclass
class DatasetSearchSpace:
    key: str
    title: str
    bundle: dict[str, Any]
    numeric: list[NumericDimension]
    categorical: list[CategoricalDimension]
    assets: dict[str, Any]


def load_pickle(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def as_float(value: Any, fallback: float | None = None) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    if fallback is None:
        raise ValueError(f"Cannot convert to float: {value!r}")
    return fallback


def range_from_metadata(name: str, meta: dict[str, Any]) -> NumericDimension:
    lower = as_float(meta.get("p05"), as_float(meta.get("min")))
    upper = as_float(meta.get("p95"), as_float(meta.get("max")))
    source_lower = as_float(meta.get("min"), lower)
    source_upper = as_float(meta.get("max"), upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower = source_lower
        upper = source_upper
    if lower > upper:
        lower, upper = upper, lower
    return NumericDimension(name=name, lower=lower, upper=upper, source_lower=source_lower, source_upper=source_upper)


def load_search_spaces() -> list[DatasetSearchSpace]:
    assets = json.loads(APP_ASSETS_PATH.read_text(encoding="utf-8"))
    dataset1_bundle = load_pickle(PKL_DIR / "dataset1_biochar_removal_model.pkl.gz")
    dataset2_bundle = load_pickle(PKL_DIR / "dataset2_resin_removal_known_catalog_model.pkl.gz")

    ds1_assets = assets["datasets"]["dataset1"]
    ds2_assets = assets["datasets"]["dataset2"]

    ds1_numeric = [range_from_metadata(name, ds1_assets["ranges"][name]) for name in ds1_assets["numeric_fields"]]
    ds1_categorical = [
        CategoricalDimension("PFAS", tuple(ds1_assets["pfas_options"])),
    ]

    resin_profiles = tuple(
        {
            (
                ex["values"].get("Resin"),
                ex["values"].get("Polymer_matrix "),
                ex["values"].get("Porosity"),
                ex["values"].get("Functional group"),
                ex["values"].get("Resin_type"),
            )
            for ex in ds2_assets["examples"]
        }
    )
    # Use all options from metadata for PFAS/Solution and all observed resin profiles
    # when the source workbook is available. The deployed app does not ship raw
    # supplementary workbooks, so it falls back to the packaged example profiles.
    source_path = ROOT / "PFAS DATASET 2" / "es4c14223_si_001.xlsx"
    profile_cols = ["Resin", "Polymer_matrix ", "Porosity", "Functional group", "Resin_type"]
    if source_path.exists():
        source = pd.read_excel(source_path, sheet_name="source data")
        resin_profile_df = source[profile_cols].dropna().drop_duplicates()
        for col in profile_cols:
            resin_profile_df = resin_profile_df[
                ~resin_profile_df[col].astype(str).str.strip().str.lower().isin(INVALID_CATEGORY_VALUES)
            ]
        resin_profiles = tuple(
            tuple(str(row[col]).strip() for col in profile_cols)
            for _, row in resin_profile_df.iterrows()
        )

    ds2_numeric = [range_from_metadata(name, ds2_assets["ranges"][name]) for name in ds2_assets["numeric_fields"]]
    ds2_categorical = [
        CategoricalDimension("PFAS", tuple(ds2_assets["options"]["PFAS"])),
        CategoricalDimension(
            "Solution",
            tuple(
                value
                for value in ds2_assets["options"]["Solution"]
                if str(value).strip().lower() not in INVALID_CATEGORY_VALUES
            ),
        ),
        CategoricalDimension("Resin profile", resin_profiles),
    ]

    return [
        DatasetSearchSpace("dataset1", ds1_assets["title"], dataset1_bundle, ds1_numeric, ds1_categorical, ds1_assets),
        DatasetSearchSpace("dataset2", ds2_assets["title"], dataset2_bundle, ds2_numeric, ds2_categorical, ds2_assets),
    ]


def bounds_for(space: DatasetSearchSpace) -> tuple[np.ndarray, np.ndarray]:
    lows: list[float] = []
    highs: list[float] = []
    for cat in space.categorical:
        lows.append(0.0)
        highs.append(float(len(cat.choices) - 1))
    for num in space.numeric:
        lows.append(num.lower)
        highs.append(num.upper)
    return np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)


def decode_particle(space: DatasetSearchSpace, particle: np.ndarray) -> dict[str, Any]:
    values: dict[str, Any] = {}
    offset = 0
    for cat in space.categorical:
        idx = int(np.clip(np.rint(particle[offset]), 0, len(cat.choices) - 1))
        choice = cat.choices[idx]
        if space.key == "dataset2" and cat.name == "Resin profile":
            resin, polymer, porosity, functional_group, resin_type = choice
            values.update(
                {
                    "Resin": resin,
                    "Polymer_matrix ": polymer,
                    "Porosity": porosity,
                    "Functional group": functional_group,
                    "Resin_type": resin_type,
                }
            )
        else:
            values[cat.name] = choice
        offset += 1
    for num in space.numeric:
        values[num.name] = float(np.clip(particle[offset], num.lower, num.upper))
        offset += 1
    return values


def candidate_key(space: DatasetSearchSpace, values: dict[str, Any]) -> tuple[Any, ...]:
    parts: list[Any] = []
    if space.key == "dataset1":
        parts.append(values.get("PFAS"))
    else:
        parts.extend(
            [
                values.get("PFAS"),
                values.get("Solution"),
                values.get("Resin"),
                values.get("Polymer_matrix "),
                values.get("Porosity"),
                values.get("Functional group"),
                values.get("Resin_type"),
            ]
        )
    for num in space.numeric:
        parts.append(round(float(values[num.name]), 4))
    return tuple(parts)


class Predictor:
    def __init__(self, space: DatasetSearchSpace):
        self.space = space

    @lru_cache(maxsize=4096)
    def _base_row_dataset1(self, pfas: str):
        smiles_map = self.space.assets.get("pfas_smiles_map", {})
        values = {"PFAS": pfas, "SMILES": smiles_map.get(pfas, "")}
        return build_dataset1_row(values, self.space.assets["defaults"], self.space.bundle["input_columns"])

    @lru_cache(maxsize=8192)
    def _base_row_dataset2(
        self,
        pfas: str,
        solution: str,
        resin: str,
        polymer: str,
        porosity: str,
        functional_group: str,
        resin_type: str,
    ):
        values = {
            "PFAS": pfas,
            "Solution": solution,
            "Resin": resin,
            "Polymer_matrix ": polymer,
            "Porosity": porosity,
            "Functional group": functional_group,
            "Resin_type": resin_type,
        }
        return build_dataset2_row(values, self.space.assets["defaults"], self.space.bundle["input_columns"])

    def row_for(self, values: dict[str, Any]) -> pd.DataFrame:
        if self.space.key == "dataset1":
            row = self._base_row_dataset1(str(values["PFAS"])).copy()
            for num in self.space.numeric:
                if num.name in row.columns:
                    row.loc[:, num.name] = values[num.name]
            return row

        row = self._base_row_dataset2(
            str(values["PFAS"]),
            str(values["Solution"]),
            str(values["Resin"]),
            str(values["Polymer_matrix "]),
            str(values["Porosity"]),
            str(values["Functional group"]),
            str(values["Resin_type"]),
        ).copy()
        numeric_mapping = {
            "initial_pfas_concentration_ug_L": ("initial_concentration_mg_L", 1 / 1000.0),
            "resin_dosage_mg_L": ("resin_dosage_g_L", 1 / 1000.0),
            "pH": ("pH", 1.0),
            "temperature_C": ("temperature\n(℃)", 1.0),
            "contact_time_h": ("contact_time\n(h)", 1.0),
            "stirring_rate_rpm": ("Stirring_rate_numeric", 1.0),
            "CDOC_mg_L": ("CDOC\n(mg/L)", 1.0),
        }
        for field, (model_col, scale) in numeric_mapping.items():
            if field in values and model_col in row.columns:
                row.loc[:, model_col] = float(values[field]) * scale
        return row

    def predict_many(self, candidates: list[dict[str, Any]]) -> np.ndarray:
        rows = [self.row_for(values) for values in candidates]
        X = pd.concat(rows, ignore_index=True)
        pred = self.space.bundle["model"].predict(X)
        return np.clip(np.asarray(pred, dtype=float), 0, 100)


def run_pso(
    space: DatasetSearchSpace,
    direction: str,
    *,
    particle_size: int = PARTICLE_SIZE,
    iterations: int = ITERATIONS,
    inertia: float = INERTIA,
    cognitive: float = COGNITIVE,
    social: float = SOCIAL,
    top_k: int = 5,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if direction not in {"max", "min"}:
        raise ValueError("direction must be max or min")
    particle_size = int(np.clip(particle_size, 4, 160))
    iterations = int(np.clip(iterations, 1, 300))
    top_k = int(np.clip(top_k, 1, 25))

    predictor = Predictor(space)
    low, high = bounds_for(space)
    dim = len(low)
    rng = np.random.default_rng()

    positions = rng.uniform(low, high, size=(particle_size, dim))
    velocities = rng.normal(0, 0.15, size=(particle_size, dim)) * (high - low)
    personal_best_positions = positions.copy()
    personal_best_scores = np.full(particle_size, np.inf)
    global_best_position = positions[0].copy()
    global_best_score = np.inf
    archive: dict[tuple[Any, ...], dict[str, Any]] = {}
    history: list[dict[str, float]] = []

    def emit_progress(iteration: int, message: str) -> None:
        if progress_callback is None:
            return
        try:
            archive_df_now = pd.DataFrame(list(archive.values()))
            if not archive_df_now.empty:
                archive_df_now = archive_df_now.sort_values("prediction", ascending=(direction == "min")).head(500)
            progress_callback(
                {
                    "status": "running",
                    "phase": "pso",
                    "message": message,
                    "progress": float(np.clip(iteration / max(iterations, 1), 0.0, 1.0)),
                    "current_iteration": int(iteration),
                    "total_iterations": int(iterations),
                    "history": list(history),
                    "all_candidates": archive_df_now.replace({np.nan: None}).to_dict(orient="records") if not archive_df_now.empty else [],
                }
            )
        except Exception:
            pass

    for iteration in range(iterations):
        candidates = [decode_particle(space, positions[i]) for i in range(particle_size)]
        predictions = predictor.predict_many(candidates)
        fitness = -predictions if direction == "max" else predictions

        for i, values in enumerate(candidates):
            score = float(fitness[i])
            pred = float(predictions[i])
            key = candidate_key(space, values)
            existing = archive.get(key)
            if existing is None or (direction == "max" and pred > existing["prediction"]) or (direction == "min" and pred < existing["prediction"]):
                archive[key] = {"prediction": pred, **values}
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i].copy()
            if score < global_best_score:
                global_best_score = score
                global_best_position = positions[i].copy()

        best_prediction = float(-global_best_score if direction == "max" else global_best_score)
        archive_predictions = np.asarray([row["prediction"] for row in archive.values()], dtype=float)
        history.append(
            {
                "iteration": iteration + 1,
                "best_prediction": best_prediction,
                "mean_archive_prediction": float(np.mean(archive_predictions)) if archive_predictions.size else best_prediction,
                "evaluated_unique_candidates": int(len(archive)),
            }
        )
        emit_progress(iteration + 1, f"PSO iteration {iteration + 1}/{iterations}: best predicted removal {best_prediction:.2f}%.")

        r1 = rng.random(size=(particle_size, dim))
        r2 = rng.random(size=(particle_size, dim))
        velocities = (
            inertia * velocities
            + cognitive * r1 * (personal_best_positions - positions)
            + social * r2 * (global_best_position - positions)
        )
        positions = np.clip(positions + velocities, low, high)

    archive_df = pd.DataFrame(list(archive.values()))
    archive_df = archive_df.sort_values("prediction", ascending=(direction == "min"))
    top5 = archive_df.head(top_k).copy()
    details = {
        "direction": direction,
        "particle_size": particle_size,
        "iterations": iterations,
        "inertia": inertia,
        "cognitive": cognitive,
        "social": social,
        "top_k": top_k,
        "range_mode": RANGE_MODE,
        "evaluated_unique_candidates": int(len(archive_df)),
        "history": history,
    }
    return top5, details


def parameter_ranges(space: DatasetSearchSpace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cat in space.categorical:
        rows.append(
            {
                "parameter": cat.name,
                "type": "categorical_discrete",
                "optimization_lower": 0,
                "optimization_upper": len(cat.choices) - 1,
                "training_min": "",
                "training_max": "",
                "choices_count": len(cat.choices),
                "choices": "; ".join(map(str, cat.choices[:80])),
            }
        )
    for num in space.numeric:
        rows.append(
            {
                "parameter": num.name,
                "type": "continuous",
                "optimization_lower": num.lower,
                "optimization_upper": num.upper,
                "training_min": num.source_lower,
                "training_max": num.source_upper,
                "choices_count": "",
                "choices": "",
            }
        )
    return pd.DataFrame(rows)


def save_bar_plot(space: DatasetSearchSpace, max_df: pd.DataFrame, min_df: pd.DataFrame) -> Path:
    plot_path = OPT_DIR / f"{space.key}_pso_top5_bar.png"
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=160)
    labels = [f"Max {i + 1}" for i in range(len(max_df))] + [f"Min {i + 1}" for i in range(len(min_df))]
    values = max_df["prediction"].tolist() + min_df["prediction"].tolist()
    colors = ["#157f3b"] * len(max_df) + ["#b42318"] * len(min_df)
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Predicted removal efficiency (%)")
    ax.set_title(f"{space.title}: PSO top candidates")
    ax.grid(axis="y", color="#e5e7eb")
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def render_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, columns: list[str] | None = None) -> None:
    shown = df.copy()
    if columns:
        shown = shown[[c for c in columns if c in shown.columns]]
    shown = shown.head(10).copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: f"{x:.4g}")
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=18)
    table = ax.table(cellText=shown.astype(str).values, colLabels=shown.columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pdf_report(results: list[dict[str, Any]]) -> Path:
    pdf_path = OUTPUT_DIR / "PFAS_PSO_optimization_top5_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        lines = [
            "PFAS PSO Optimization Report",
            "",
            f"Particle size: {PARTICLE_SIZE}",
            f"Iterations: {ITERATIONS}",
            f"Objective: single-objective model-predicted removal efficiency",
            f"Numeric range mode: {RANGE_MODE} from training data",
            "Categorical variables are optimized as discrete choices.",
            "Dataset 2 resin properties are constrained to observed resin profiles.",
            "",
            "Note: results are model-suggested candidates, not experimentally verified optima.",
        ]
        ax.text(0.07, 0.90, "\n".join(lines), va="top", ha="left", fontsize=14)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for item in results:
            plot = plt.imread(item["bar_plot"])
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.axis("off")
            ax.imshow(plot)
            ax.set_title(item["title"], fontsize=16, fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            common_cols = [
                "prediction",
                "PFAS",
                "Solution",
                "Resin",
                "Polymer_matrix ",
                "Functional group",
                "resin_dosage_mg_L",
                "initial_pfas_concentration_ug_L",
                "contact_time_h",
                "stirring_rate_rpm",
                "pH",
                "temperature_C",
                "CDOC_mg_L",
                "Initial concentration",
                "Adsorption time",
                "S/L",
            ]
            render_table_page(pdf, f"{item['title']} - Top 5 maximum", item["max_df"], common_cols)
            render_table_page(pdf, f"{item['title']} - Top 5 minimum", item["min_df"], common_cols)
    return pdf_path


def main() -> None:
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    spaces = load_search_spaces()
    all_results: list[dict[str, Any]] = []

    for space in spaces:
        print(f"\n=== Optimizing {space.title} ===")
        ranges_df = parameter_ranges(space)
        ranges_path = OPT_DIR / f"{space.key}_pso_parameter_ranges.csv"
        ranges_df.to_csv(ranges_path, index=False)

        max_df, max_details = run_pso(space, "max")
        min_df, min_details = run_pso(space, "min")

        max_path = OPT_DIR / f"{space.key}_pso_top5_maximum.csv"
        min_path = OPT_DIR / f"{space.key}_pso_top5_minimum.csv"
        max_df.to_csv(max_path, index=False)
        min_df.to_csv(min_path, index=False)
        bar_plot = save_bar_plot(space, max_df, min_df)

        details = {
            "dataset": space.key,
            "title": space.title,
            "model_name": space.bundle.get("best_model_name"),
            "parameter_ranges_csv": str(ranges_path),
            "top5_maximum_csv": str(max_path),
            "top5_minimum_csv": str(min_path),
            "bar_plot": str(bar_plot),
            "max_details": max_details,
            "min_details": min_details,
        }
        details_path = OPT_DIR / f"{space.key}_pso_optimization_details.json"
        details_path.write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")

        print("Parameter ranges:", ranges_path)
        print("Top 5 maximum:", max_path)
        print(max_df.head(5).to_string(index=False))
        print("Top 5 minimum:", min_path)
        print(min_df.head(5).to_string(index=False))

        all_results.append({"title": space.title, "max_df": max_df, "min_df": min_df, "bar_plot": str(bar_plot), **details})

    summary_path = OPT_DIR / "pso_optimization_summary.json"
    summary_path.write_text(
        json.dumps(
            [
                {
                    k: v
                    for k, v in item.items()
                    if k not in {"max_df", "min_df"}
                }
                for item in all_results
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pdf_path = write_pdf_report(all_results)
    print("\nSummary:", summary_path)
    print("PDF report:", pdf_path)


if __name__ == "__main__":
    main()
