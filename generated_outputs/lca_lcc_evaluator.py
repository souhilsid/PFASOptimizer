from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path(os.environ.get("PFAS_LCA_DATA_DIR", ROOT.parent)).expanduser()
GUIDE_XLSX = DOWNLOADS / "biochar_pfas_openlca_lca_lcc_complete_guide.xlsx"
NORMALIZATION_CSV = DOWNLOADS / "recipe2016_midpoint_h_normalization_values.csv"
COUNTRY_COST_XLSX = DOWNLOADS / "biochar_pfas_real_country_cost_profiles_recreated.xlsx"

DEFAULT_PRODUCT_SYSTEMS = {
    "incineration": "PS_Biochar_PFAS_Incineration",
    "landfill": "PS_Biochar_PFAS_Landfill",
}
DEFAULT_IMPACT_METHOD = "ReCiPe Midpoint (H)"
DEFAULT_IPC_PORT = 8080
DEFAULT_CLOUD_TIMEOUT_SECONDS = float(os.environ.get("OPENLCA_EVALUATOR_TIMEOUT", "180"))

CANDIDATE_TO_GLOBAL = {
    "PFAS": "PFAS_name",
    "Pyrolysis temperature": "T_pyro_C",
    "Pyrolysis time": "t_pyro_min",
    "Heating rated": "heating_rate_C_min",
    "C": "C_percent",
    "Ash": "ash_percent",
    "H/C": "H_C",
    "O/C": "O_C",
    "(O+N)/C": "ON_C",
    "Surface area": "surface_area_m2_g",
    "Average pore size": "avg_pore_size_nm",
    "Pore volume": "pore_volume_cm3_g",
    "Solution pH": "pH_solution",
    "Adsorption time": "t_ads_min",
    "Adsorption temperature": "T_ads_C",
    "S/L": "SLR_g_L",
    "Initial concentration": "C0_mg_L",
    "CaCl2": "CaCl2_mM",
    "NaCl": "NaCl_mM",
    "HA": "humic_mg_L",
}
CANDIDATE_DRIVEN_GLOBALS = set(CANDIDATE_TO_GLOBAL.values()) | {"RE_pred"}

COUNTRY_COST_PARAMETER_MAP = {
    "price_CaCO3_USD_kg": "price_caco3_USD_kg",
}

FALLBACK_GLOBAL_ROWS = [
    ("V_water_m3", 1.0, "m3", "Platform/LCA", "Functional unit volume"),
    ("V_water_L", 1000.0, "L", "Platform/LCA", "Water volume in liters"),
    ("PFAS_name", "PFOA", "-", "Platform/PKL", "PFAS selection"),
    ("T_pyro_C", 600.0, "deg C", "Platform/PKL/LCA driver", "Pyrolysis temperature"),
    ("t_pyro_min", 90.0, "min", "Platform/PKL/LCA driver", "Pyrolysis time"),
    ("heating_rate_C_min", 8.5, "deg C/min", "Platform/PKL/LCA driver", "Heating rate"),
    ("C_percent", 67.62, "%", "PKL", "Carbon content"),
    ("ash_percent", 3.61, "%", "PKL", "Ash content"),
    ("H_C", 0.245, "-", "PKL", "H/C ratio"),
    ("O_C", 0.208, "-", "PKL", "O/C ratio"),
    ("ON_C", 0.219, "-", "PKL", "(O+N)/C ratio"),
    ("surface_area_m2_g", 206.97, "m2/g", "PKL", "Surface area"),
    ("avg_pore_size_nm", 3.31, "nm", "PKL", "Average pore size"),
    ("pore_volume_cm3_g", 0.125, "cm3/g", "PKL", "Pore volume"),
    ("pH_solution", 6.5, "-", "Platform/PKL", "Solution pH"),
    ("t_ads_min", 720.0, "min", "Platform/PKL/LCA", "Adsorption time"),
    ("T_ads_C", 25.0, "deg C", "Platform/PKL", "Adsorption temperature"),
    ("SLR_g_L", 0.25, "g/L", "Platform/PKL/LCA", "Solid/liquid ratio"),
    ("C0_mg_L", 1.0, "mg/L", "Platform/PKL/LCA", "Initial PFAS concentration"),
    ("CaCl2_mM", 0.0, "mM", "Platform/PKL/LCA", "Calcium chloride concentration"),
    ("NaCl_mM", 0.0, "mM", "Platform/PKL/LCA", "Sodium chloride concentration"),
    ("humic_mg_L", 0.0, "mg/L", "Platform/PKL/LCA", "Humic acid concentration"),
    ("RE_pred", 0.8, "fraction", "PKL output", "Predicted removal efficiency"),
    ("biochar_yield", 0.3, "kg/kg", "LCA/LCC", "Biochar yield from dry biomass"),
    ("biomass_moisture_factor", 1.25, "kg wet/kg dry", "LCA/LCC", "Wet biomass input per dry biomass"),
    ("biomass_transport_km", 50.0, "km", "LCA/LCC", "Biomass transport distance"),
    ("prep_elec_kWh_kgDry", 0.02, "kWh/kg dry", "LCA/LCC", "Preparation electricity"),
    ("prep_heat_MJ_kgDry", 0.5, "MJ/kg dry", "LCA/LCC", "Preparation heat"),
    ("pyrolysis_elec_kWh_kgBC", 0.2, "kWh/kg BC", "LCA/LCC", "Pyrolysis electricity"),
    ("pyrolysis_heat_MJ_kgBC", 5.0, "MJ/kg BC", "LCA/LCC", "Pyrolysis heat"),
    ("N2_kg_kgBC", 0.0, "kg/kg BC", "LCA/LCC", "Nitrogen gas"),
    ("wash_water_kg_kgBC", 5.0, "kg/kg BC", "LCA/LCC", "Washing water"),
    ("washing_elec_kWh_kgBC", 0.1, "kWh/kg BC", "LCA/LCC", "Washing electricity"),
    ("washing_heat_MJ_kgBC", 1.0, "MJ/kg BC", "LCA/LCC", "Washing/drying heat"),
    ("KOH_kg_kgBC", 0.0, "kg/kg BC", "LCA/LCC", "KOH activation input"),
    ("HCl_kg_kgBC", 0.0, "kg/kg BC", "LCA/LCC", "HCl washing/neutralization"),
    ("FeCl3_kg_kgBC", 0.0, "kg/kg BC", "LCA/LCC", "FeCl3 modification"),
    ("urea_kg_kgBC", 0.0, "kg/kg BC", "LCA/LCC", "Urea/N-doping input"),
    ("adsorption_elec_kWh_m3_h", 0.0067, "kWh/m3/h", "LCA/LCC", "Adsorption mixing/pumping rate"),
    ("filter_media_kg_m3", 0.01, "kg/m3", "LCA/LCC", "Filter media for post-separation"),
    ("ash_fraction_spent", 0.05, "kg ash/kg spent", "LCA/LCC", "Incineration ash fraction"),
]

FALLBACK_COST_ROWS = [
    ("price_wet_biomass_USD_kg", 0.05, "USD/kg", "Wet biomass price"),
    ("price_electricity_USD_kWh", 0.12, "USD/kWh", "Electricity tariff"),
    ("price_heat_USD_MJ", 0.015, "USD/MJ", "Heat/natural gas cost"),
    ("price_transport_USD_tkm", 0.10, "USD/tkm", "Road transport cost"),
    ("price_N2_USD_kg", 0.25, "USD/kg", "Nitrogen price"),
    ("price_water_USD_kg", 0.001, "USD/kg", "Process water price"),
    ("price_wastewater_USD_m3", 0.50, "USD/m3", "Wastewater treatment cost"),
    ("price_KOH_USD_kg", 1.50, "USD/kg", "KOH price"),
    ("price_HCl_USD_kg", 0.20, "USD/kg", "HCl price"),
    ("price_FeCl3_USD_kg", 0.80, "USD/kg", "FeCl3 price"),
    ("price_urea_USD_kg", 0.40, "USD/kg", "Urea price"),
    ("price_CaCl2_USD_kg", 0.35, "USD/kg", "CaCl2 price"),
    ("price_NaCl_USD_kg", 0.10, "USD/kg", "NaCl price"),
    ("price_humic_USD_kg", 5.00, "USD/kg", "Humic acid proxy price"),
    ("price_filter_media_USD_kg", 2.00, "USD/kg", "Filter media price"),
    ("price_incin_USD_kg", 2.00, "USD/kg", "Hazardous incineration service"),
    ("price_landfill_USD_kg", 0.25, "USD/kg", "Hazardous landfill service"),
    ("price_caco3_USD_kg", 0.05, "USD/kg", "CaCO3/limestone price"),
    ("price_HCl30_USD_kg", 0.12, "USD/kg", "30% HCl solution price"),
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _clean(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else None
    return value


def _as_float(values: dict[str, Any], name: str, fallback: float = 0.0) -> float:
    value = values.get(name, fallback)
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return float(fallback)


def _row_dict(name: Any, default: Any, unit: Any, role: Any, meaning: Any, source: str, editable: bool) -> dict[str, Any] | None:
    name = _clean(name)
    if not name:
        return None
    return {
        "name": str(name),
        "default": _clean(default),
        "unit": _clean(unit) or "",
        "role": _clean(role) or "",
        "meaning": _clean(meaning) or "",
        "source": source,
        "editable": bool(editable),
    }


def _fallback_global_rows() -> list[dict[str, Any]]:
    rows = []
    for name, default, unit, role, meaning in FALLBACK_GLOBAL_ROWS:
        rows.append(_row_dict(name, default, unit, role, meaning, "candidate" if name in CANDIDATE_DRIVEN_GLOBALS else "constant", name not in CANDIDATE_DRIVEN_GLOBALS))
    return [row for row in rows if row]


def _fallback_cost_rows() -> list[dict[str, Any]]:
    rows = []
    for name, default, unit, meaning in FALLBACK_COST_ROWS:
        rows.append(_row_dict(name, default, unit, "Cost", meaning, "constant", True))
    return [row for row in rows if row]


def _read_parameter_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not GUIDE_XLSX.exists():
        return _fallback_global_rows(), _fallback_cost_rows()
    try:
        raw = pd.read_excel(GUIDE_XLSX, sheet_name="Global_Params", header=None)
    except Exception:
        return _fallback_global_rows(), _fallback_cost_rows()

    global_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for i in range(4, len(raw)):
        row = raw.iloc[i]
        global_row = _row_dict(
            row.get(0),
            row.get(1),
            row.get(2),
            row.get(3),
            row.get(4),
            "candidate" if str(_clean(row.get(0)) or "") in CANDIDATE_DRIVEN_GLOBALS else "constant",
            str(_clean(row.get(0)) or "") not in CANDIDATE_DRIVEN_GLOBALS,
        )
        if global_row:
            global_rows.append(global_row)
        cost_row = _row_dict(row.get(7), row.get(8), row.get(9), row.get(10), row.get(11), "constant", True)
        if cost_row:
            cost_rows.append(cost_row)
    return global_rows or _fallback_global_rows(), cost_rows or _fallback_cost_rows()


def _read_dependent_rows() -> list[dict[str, Any]]:
    if not GUIDE_XLSX.exists():
        return []
    try:
        raw = pd.read_excel(GUIDE_XLSX, sheet_name="Dependent_Params_P04", header=None)
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for i in range(4, len(raw)):
        name = _clean(raw.iloc[i].get(0))
        formula = _clean(raw.iloc[i].get(1))
        if not name or not formula:
            continue
        rows.append(
            {
                "name": str(name),
                "formula": str(formula),
                "unit": _clean(raw.iloc[i].get(2)) or "",
                "location": _clean(raw.iloc[i].get(3)) or "",
                "meaning": _clean(raw.iloc[i].get(4)) or "",
            }
        )
    return rows


def _read_normalization_rows() -> list[dict[str, Any]]:
    if not NORMALIZATION_CSV.exists():
        return []
    try:
        df = pd.read_csv(NORMALIZATION_CSV)
    except Exception:
        return []
    rows = []
    for _, row in df.iterrows():
        name = _clean(row.get("openlca_recipe2016_midpoint_h_category"))
        value = _clean(row.get("normalization_value"))
        if not name or value is None:
            continue
        rows.append(
            {
                "category": str(name),
                "normalization_value": float(value),
                "unit": _clean(row.get("normalization_unit")) or "",
                "perspective": _clean(row.get("perspective")) or "Hierarchic",
                "weight": 1.0,
            }
        )
    return rows


def _read_country_profiles() -> list[dict[str, Any]]:
    if not COUNTRY_COST_XLSX.exists():
        return []
    try:
        df = pd.read_excel(COUNTRY_COST_XLSX, sheet_name="Country_Profiles")
    except Exception:
        return []
    profiles: list[dict[str, Any]] = []
    cost_columns = [col for col in df.columns if str(col).startswith("price_")]
    quality_columns = [col for col in df.columns if str(col).endswith("_quality")]
    for _, row in df.iterrows():
        country = _clean(row.get("country"))
        iso3 = _clean(row.get("iso3"))
        if not country:
            continue
        costs: dict[str, float] = {}
        for col in cost_columns:
            target_name = COUNTRY_COST_PARAMETER_MAP.get(str(col), str(col))
            value = _clean(row.get(col))
            try:
                numeric = float(value)
            except Exception:
                continue
            if np.isfinite(numeric):
                costs[target_name] = numeric
        profiles.append(
            {
                "country": str(country),
                "iso3": str(iso3 or country)[:3].upper(),
                "region": _clean(row.get("region_proxy")) or "",
                "costs": costs,
                "qualities": {str(col): _clean(row.get(col)) or "" for col in quality_columns},
                "data_quality_overall": _clean(row.get("data_quality_overall")) or "",
                "notes": _clean(row.get("notes")) or "",
            }
        )
    return sorted(profiles, key=lambda item: (str(item.get("region", "")), str(item.get("country", ""))))


@dataclass
class LcaLccConfig:
    global_rows: list[dict[str, Any]]
    cost_rows: list[dict[str, Any]]
    dependent_rows: list[dict[str, Any]]
    normalization_rows: list[dict[str, Any]]
    country_profiles: list[dict[str, Any]]

    @property
    def default_constants(self) -> dict[str, Any]:
        constants: dict[str, Any] = {}
        for row in self.global_rows + self.cost_rows:
            if row.get("editable"):
                constants[row["name"]] = row.get("default")
        return constants

    @property
    def normalization_map(self) -> dict[str, float]:
        return {row["category"]: float(row["normalization_value"]) for row in self.normalization_rows}


def load_lca_lcc_config() -> LcaLccConfig:
    global_rows, cost_rows = _read_parameter_rows()
    return LcaLccConfig(
        global_rows=global_rows,
        cost_rows=cost_rows,
        dependent_rows=_read_dependent_rows(),
        normalization_rows=_read_normalization_rows(),
        country_profiles=_read_country_profiles(),
    )


def lca_lcc_metadata(config: LcaLccConfig | None = None) -> dict[str, Any]:
    cfg = config or load_lca_lcc_config()
    cloud_url = os.environ.get("OPENLCA_EVALUATOR_URL", "").strip()
    default_environmental_mode = os.environ.get("PFAS_ENVIRONMENTAL_MODE", "").strip().lower()
    if not default_environmental_mode:
        default_environmental_mode = "openlca_cloud" if cloud_url else "proxy"
    return _json_safe(
        {
            "enabled_for_dataset": "dataset1",
            "guide_path": str(GUIDE_XLSX),
            "normalization_path": str(NORMALIZATION_CSV),
            "country_cost_path": str(COUNTRY_COST_XLSX),
            "guide_available": GUIDE_XLSX.exists(),
            "normalization_available": NORMALIZATION_CSV.exists(),
            "country_cost_available": COUNTRY_COST_XLSX.exists(),
            "ipc_default_port": DEFAULT_IPC_PORT,
            "cloud_evaluator_configured": bool(cloud_url),
            "cloud_evaluator_label": os.environ.get("OPENLCA_EVALUATOR_LABEL", "Cloud OpenLCA evaluator"),
            "default_environmental_mode": default_environmental_mode,
            "impact_method_default": DEFAULT_IMPACT_METHOD,
            "product_systems": DEFAULT_PRODUCT_SYSTEMS,
            "candidate_mapping": CANDIDATE_TO_GLOBAL,
            "global_parameters": cfg.global_rows,
            "cost_parameters": cfg.cost_rows,
            "dependent_parameters": cfg.dependent_rows,
            "normalization_values": cfg.normalization_rows,
            "country_profiles": cfg.country_profiles,
            "default_constants": cfg.default_constants,
        }
    )


class BiocharLcaLccEvaluator:
    def __init__(self, config: LcaLccConfig | None = None):
        self.config = config or load_lca_lcc_config()

    def metadata(self) -> dict[str, Any]:
        return lca_lcc_metadata(self.config)

    def country_costs(self, country_or_iso3: str | None) -> dict[str, Any]:
        if not country_or_iso3:
            return {}
        key = str(country_or_iso3).strip().lower()
        for profile in self.config.country_profiles:
            if str(profile.get("country", "")).lower() == key or str(profile.get("iso3", "")).lower() == key:
                return profile
        return {}

    def build_global_parameters(
        self,
        candidate: dict[str, Any],
        prediction_percent: float,
        constants: dict[str, Any] | None = None,
        country: str | None = None,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for row in self.config.global_rows + self.config.cost_rows:
            values[row["name"]] = row.get("default")
        values.update(self.config.default_constants)
        country_profile = self.country_costs(country)
        if country_profile:
            values.update(country_profile.get("costs") or {})
        if constants:
            values.update({k: v for k, v in constants.items() if v not in ("", None)})
        for candidate_key, global_name in CANDIDATE_TO_GLOBAL.items():
            if candidate_key in candidate and candidate[candidate_key] not in ("", None):
                values[global_name] = candidate[candidate_key]
        values["RE_pred"] = float(np.clip(prediction_percent, 0, 100)) / 100.0
        return values

    def calculate_lci(self, params: dict[str, Any]) -> dict[str, float]:
        v_water_m3 = _as_float(params, "V_water_m3", 1.0)
        v_water_l = _as_float(params, "V_water_L", 1000.0)
        slr = _as_float(params, "SLR_g_L", 0.25)
        c0 = _as_float(params, "C0_mg_L", 1.0)
        re_pred = np.clip(_as_float(params, "RE_pred", 0.8), 0, 1)
        biochar_yield = max(_as_float(params, "biochar_yield", 0.3), 1e-9)

        m_biochar_kg = slr * v_water_l / 1000.0
        m_pfas_in_kg = c0 * v_water_l * 0.000001
        m_pfas_removed_kg = m_pfas_in_kg * re_pred
        m_pfas_residual_kg = m_pfas_in_kg * (1.0 - re_pred)
        m_spent_biochar_kg = m_biochar_kg + m_pfas_removed_kg
        m_dry_biomass_kg = m_biochar_kg / biochar_yield
        m_wet_biomass_kg = m_dry_biomass_kg * _as_float(params, "biomass_moisture_factor", 1.25)
        m_cacl2_kg = _as_float(params, "CaCl2_mM", 0.0) * 0.001 * v_water_l * 0.11098
        m_nacl_kg = _as_float(params, "NaCl_mM", 0.0) * 0.001 * v_water_l * 0.05844
        m_humic_kg = _as_float(params, "humic_mg_L", 0.0) * v_water_l * 0.000001
        adsorption_elec_kwh = _as_float(params, "adsorption_elec_kWh_m3_h", 0.0067) * v_water_m3 * _as_float(params, "t_ads_min", 720.0) / 60.0
        m_filter_media_kg = _as_float(params, "filter_media_kg_m3", 0.01) * v_water_m3
        m_ash_residue_kg = m_spent_biochar_kg * _as_float(params, "ash_fraction_spent", 0.05)
        return {
            "m_biochar_kg": m_biochar_kg,
            "m_PFAS_in_kg": m_pfas_in_kg,
            "m_PFAS_removed_kg": m_pfas_removed_kg,
            "m_PFAS_residual_kg": m_pfas_residual_kg,
            "m_spent_biochar_kg": m_spent_biochar_kg,
            "m_dry_biomass_kg": m_dry_biomass_kg,
            "m_wet_biomass_kg": m_wet_biomass_kg,
            "m_CaCl2_kg": m_cacl2_kg,
            "m_NaCl_kg": m_nacl_kg,
            "m_humic_kg": m_humic_kg,
            "adsorption_elec_kWh": adsorption_elec_kwh,
            "m_filter_media_kg": m_filter_media_kg,
            "m_ash_residue_kg": m_ash_residue_kg,
        }

    def calculate_lcc(self, params: dict[str, Any], lci: dict[str, float], eol_mode: str) -> dict[str, float]:
        p = lambda name, fallback=0.0: _as_float(params, name, fallback)
        cost_p01 = lci["m_dry_biomass_kg"] * (
            p("biomass_moisture_factor", 1.25) * p("price_wet_biomass_USD_kg", 0.05)
            + p("prep_elec_kWh_kgDry", 0.02) * p("price_electricity_USD_kWh", 0.12)
            + p("prep_heat_MJ_kgDry", 0.5) * p("price_heat_USD_MJ", 0.015)
            + p("biomass_moisture_factor", 1.25) * p("biomass_transport_km", 50.0) / 1000.0 * p("price_transport_USD_tkm", 0.10)
        )
        cost_p02 = lci["m_biochar_kg"] * (
            p("pyrolysis_elec_kWh_kgBC", 0.2) * p("price_electricity_USD_kWh", 0.12)
            + p("pyrolysis_heat_MJ_kgBC", 5.0) * p("price_heat_USD_MJ", 0.015)
            + p("N2_kg_kgBC", 0.0) * p("price_N2_USD_kg", 0.25)
        )
        cost_p03 = lci["m_biochar_kg"] * (
            p("wash_water_kg_kgBC", 5.0) * p("price_water_USD_kg", 0.001)
            + p("washing_elec_kWh_kgBC", 0.1) * p("price_electricity_USD_kWh", 0.12)
            + p("washing_heat_MJ_kgBC", 1.0) * p("price_heat_USD_MJ", 0.015)
            + p("KOH_kg_kgBC", 0.0) * p("price_KOH_USD_kg", 1.5)
            + p("HCl_kg_kgBC", 0.0) * p("price_HCl_USD_kg", 0.2)
            + p("FeCl3_kg_kgBC", 0.0) * p("price_FeCl3_USD_kg", 0.8)
            + p("urea_kg_kgBC", 0.0) * p("price_urea_USD_kg", 0.4)
            + p("wash_water_kg_kgBC", 5.0) / 1000.0 * p("price_wastewater_USD_m3", 0.5)
        )
        cost_p04 = (
            lci["adsorption_elec_kWh"] * p("price_electricity_USD_kWh", 0.12)
            + lci["m_CaCl2_kg"] * p("price_CaCl2_USD_kg", 0.35)
            + lci["m_NaCl_kg"] * p("price_NaCl_USD_kg", 0.1)
            + lci["m_humic_kg"] * p("price_humic_USD_kg", 5.0)
            + lci["m_filter_media_kg"] * p("price_filter_media_USD_kg", 2.0)
        )
        cost_p05 = lci["m_spent_biochar_kg"] * p("price_incin_USD_kg", 2.0)
        cost_p06 = lci["m_spent_biochar_kg"] * p("price_landfill_USD_kg", 0.25)
        total = cost_p01 + cost_p02 + cost_p03 + cost_p04 + (cost_p06 if eol_mode == "landfill" else cost_p05)
        removed = max(lci["m_PFAS_removed_kg"], 1e-12)
        return {
            "cost_p01_USD": cost_p01,
            "cost_p02_USD": cost_p02,
            "cost_p03_USD": cost_p03,
            "cost_p04_USD": cost_p04,
            "cost_p05_USD": cost_p05,
            "cost_p06_USD": cost_p06,
            "LCC_total_incin_USD_m3": cost_p01 + cost_p02 + cost_p03 + cost_p04 + cost_p05,
            "LCC_total_landfill_USD_m3": cost_p01 + cost_p02 + cost_p03 + cost_p04 + cost_p06,
            "LCC_total_USD_m3": total,
            "LCC_USD_per_kg_PFAS_removed": total / removed,
        }

    def proxy_impact_vector(self, params: dict[str, Any], lci: dict[str, float], eol_mode: str) -> dict[str, float]:
        p = lambda name, fallback=0.0: _as_float(params, name, fallback)
        elec_kwh = (
            lci["m_dry_biomass_kg"] * p("prep_elec_kWh_kgDry", 0.02)
            + lci["m_biochar_kg"] * (p("pyrolysis_elec_kWh_kgBC", 0.2) + p("washing_elec_kWh_kgBC", 0.1))
            + lci["adsorption_elec_kWh"]
        )
        heat_mj = lci["m_dry_biomass_kg"] * p("prep_heat_MJ_kgDry", 0.5) + lci["m_biochar_kg"] * (
            p("pyrolysis_heat_MJ_kgBC", 5.0) + p("washing_heat_MJ_kgBC", 1.0)
        )
        transport_tkm = lci["m_wet_biomass_kg"] / 1000.0 * p("biomass_transport_km", 50.0)
        water_kg = lci["m_biochar_kg"] * p("wash_water_kg_kgBC", 5.0)
        chemical_kg = (
            lci["m_CaCl2_kg"]
            + lci["m_NaCl_kg"]
            + lci["m_humic_kg"]
            + lci["m_filter_media_kg"]
            + lci["m_biochar_kg"] * (p("KOH_kg_kgBC", 0.0) + p("HCl_kg_kgBC", 0.0) + p("FeCl3_kg_kgBC", 0.0) + p("urea_kg_kgBC", 0.0))
        )
        incin_kg = lci["m_spent_biochar_kg"] if eol_mode != "landfill" else 0.0
        landfill_kg = lci["m_spent_biochar_kg"] if eol_mode == "landfill" else 0.0

        gwp = elec_kwh * 0.45 + heat_mj * 0.056 + transport_tkm * 0.12 + chemical_kg * 1.8 + incin_kg * 1.1 + landfill_kg * 0.08
        fossil = elec_kwh * 0.09 + heat_mj * 0.018 + transport_tkm * 0.03 + chemical_kg * 0.35
        water_m3 = water_kg / 1000.0 + lci["m_filter_media_kg"] * 0.02 + chemical_kg * 0.003
        return {
            "Global warming": gwp,
            "Stratospheric ozone depletion": (chemical_kg + incin_kg) * 1e-7,
            "Ionizing radiation": elec_kwh * 0.03,
            "Fine particulate matter formation": heat_mj * 0.00012 + incin_kg * 0.00018,
            "Ozone formation, Human health": heat_mj * 0.00008 + transport_tkm * 0.0003,
            "Human carcinogenic toxicity": chemical_kg * 0.02 + incin_kg * 0.006,
            "Human non-carcinogenic toxicity": chemical_kg * 0.09 + landfill_kg * 0.01,
            "Water consumption": water_m3,
            "Ozone formation, Terrestrial ecosystems": heat_mj * 0.00007 + transport_tkm * 0.00025,
            "Terrestrial acidification": heat_mj * 0.00016 + incin_kg * 0.00022,
            "Terrestrial ecotoxicity": chemical_kg * 0.04,
            "Land use": lci["m_wet_biomass_kg"] * 0.03,
            "Freshwater eutrophication": water_kg * 0.000001 + chemical_kg * 0.00004,
            "Freshwater ecotoxicity": chemical_kg * 0.018 + lci["m_PFAS_residual_kg"] * 10.0,
            "Marine ecotoxicity": chemical_kg * 0.015 + lci["m_PFAS_residual_kg"] * 8.0,
            "Marine eutrophication": water_kg * 0.0000008 + chemical_kg * 0.000025,
            "Mineral resource scarcity": chemical_kg * 0.25 + lci["m_filter_media_kg"] * 0.4,
            "Fossil resource scarcity": fossil,
        }

    def cloud_status(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        options = options or {}
        base_url = str(options.get("cloud_evaluator_url") or os.environ.get("OPENLCA_EVALUATOR_URL", "")).strip().rstrip("/")
        if not base_url:
            return {"available": False, "mode": "openlca_cloud", "message": "OPENLCA_EVALUATOR_URL is not configured."}
        headers = {"Accept": "application/json"}
        api_key = str(options.get("api_key") or os.environ.get("OPENLCA_EVALUATOR_API_KEY", "")).strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            req = Request(f"{base_url}/health", headers=headers, method="GET")
            with urlopen(req, timeout=float(options.get("timeout") or 20)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            data.setdefault("available", True)
            data.setdefault("mode", "openlca_cloud")
            return _json_safe(data)
        except Exception as exc:
            return {"available": False, "mode": "openlca_cloud", "url": base_url, "message": str(exc)}

    def cloud_evaluate(self, candidate: dict[str, Any], prediction_percent: float, options: dict[str, Any]) -> dict[str, Any]:
        base_url = str(options.get("cloud_evaluator_url") or os.environ.get("OPENLCA_EVALUATOR_URL", "")).strip().rstrip("/")
        if not base_url:
            raise RuntimeError("OPENLCA_EVALUATOR_URL is not configured.")
        payload = {
            "candidate": candidate,
            "prediction_percent": float(prediction_percent),
            "options": {k: v for k, v in options.items() if k not in {"api_key", "cloud_evaluator_url"}},
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        api_key = str(options.get("api_key") or os.environ.get("OPENLCA_EVALUATOR_API_KEY", "")).strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(
            f"{base_url}/evaluate",
            data=json.dumps(_json_safe(payload)).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=float(options.get("timeout") or DEFAULT_CLOUD_TIMEOUT_SECONDS)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Cloud OpenLCA evaluator returned HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Cloud OpenLCA evaluator is unreachable: {exc}") from exc
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(str(data.get("error")))
        evaluation = data.get("evaluation") if isinstance(data, dict) and "evaluation" in data else data
        if not isinstance(evaluation, dict):
            raise RuntimeError("Cloud OpenLCA evaluator returned an invalid response.")
        evaluation.setdefault("environmental_source", "openlca_cloud")
        evaluation.setdefault("cloud_evaluator_url", base_url)
        return _json_safe(evaluation)

    def openlca_rest_status(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        options = dict(options or {})
        base_url = str(options.get("openlca_rest_url") or os.environ.get("OPENLCA_REST_URL", "")).strip().rstrip("/")
        if not base_url:
            return {"available": False, "mode": "openlca_rest", "message": "OPENLCA_REST_URL is not configured."}
        try:
            req = Request(f"{base_url}/api/version", headers={"Accept": "application/json"}, method="GET")
            with urlopen(req, timeout=float(options.get("timeout") or 20)) as resp:
                version_text = resp.read().decode("utf-8", errors="replace")
            return {"available": True, "mode": "openlca_rest", "url": base_url, "version": version_text.strip(), "message": "openLCA REST/gdt-server responded."}
        except Exception as exc:
            return {"available": False, "mode": "openlca_rest", "url": base_url, "message": str(exc)}

    def openlca_rest_impact_vector(self, params: dict[str, Any], options: dict[str, Any]) -> dict[str, float]:
        import olca_ipc.rest as rest
        import olca_schema as o

        base_url = str(options.get("openlca_rest_url") or os.environ.get("OPENLCA_REST_URL", "")).strip().rstrip("/")
        if not base_url:
            raise RuntimeError("OPENLCA_REST_URL is not configured.")
        product_system_name = str(options.get("product_system") or DEFAULT_PRODUCT_SYSTEMS.get(options.get("eol_mode", "incineration"), DEFAULT_PRODUCT_SYSTEMS["incineration"]))
        impact_method_name = str(options.get("impact_method") or DEFAULT_IMPACT_METHOD)
        client = rest.RestClient(base_url)

        product_system = None
        for item in client.get_descriptors(o.ProductSystem):
            if str(getattr(item, "name", "")).strip() == product_system_name:
                product_system = item
                break
        if product_system is None:
            raise RuntimeError(f"OpenLCA product system not found via REST: {product_system_name}")

        impact_method = None
        for item in client.get_descriptors(o.ImpactMethod):
            if str(getattr(item, "name", "")).strip() == impact_method_name:
                impact_method = item
                break
        if impact_method is None:
            raise RuntimeError(f"OpenLCA impact method not found via REST: {impact_method_name}")

        parameter_redefs = []
        for name, value in params.items():
            try:
                numeric = float(value)
            except Exception:
                continue
            if np.isfinite(numeric):
                parameter_redefs.append(o.ParameterRedef(name=str(name), value=float(numeric)))
        setup = o.CalculationSetup(
            target=product_system,
            impact_method=impact_method,
            amount=float(_as_float(params, "V_water_m3", 1.0)),
            parameters=parameter_redefs,
            with_costs=False,
        )
        result = client.calculate(setup)
        try:
            result.wait_until_ready()
            impacts = {}
            for item in result.get_total_impacts():
                ref = getattr(item, "impact_category", None)
                name = getattr(ref, "name", None) or getattr(ref, "id", None)
                amount = getattr(item, "amount", None)
                if name is not None and amount is not None:
                    impacts[str(name)] = float(amount)
            return impacts
        finally:
            try:
                result.dispose()
            except Exception:
                pass

    def openlca_impact_vector(self, params: dict[str, Any], options: dict[str, Any]) -> dict[str, float]:
        import olca_ipc as ipc
        import olca_schema as o

        port = int(options.get("ipc_port") or DEFAULT_IPC_PORT)
        product_system_name = str(options.get("product_system") or DEFAULT_PRODUCT_SYSTEMS.get(options.get("eol_mode", "incineration"), DEFAULT_PRODUCT_SYSTEMS["incineration"]))
        impact_method_name = str(options.get("impact_method") or DEFAULT_IMPACT_METHOD)
        client = ipc.Client(port)
        product_system = client.find(o.ProductSystem, product_system_name)
        if product_system is None:
            raise RuntimeError(f"OpenLCA product system not found: {product_system_name}")
        impact_method = client.find(o.ImpactMethod, impact_method_name)
        if impact_method is None:
            raise RuntimeError(f"OpenLCA impact method not found: {impact_method_name}")
        parameter_redefs = []
        for name, value in params.items():
            try:
                numeric = float(value)
            except Exception:
                continue
            if not np.isfinite(numeric):
                continue
            parameter_redefs.append(o.ParameterRedef(name=str(name), value=float(numeric)))
        setup = o.CalculationSetup(
            target=product_system,
            impact_method=impact_method,
            amount=float(_as_float(params, "V_water_m3", 1.0)),
            parameters=parameter_redefs,
            with_costs=False,
        )
        result = client.calculate(setup)
        try:
            state = result.wait_until_ready()
            if getattr(state, "error", None):
                raise RuntimeError(str(state.error))
            impacts = {}
            for item in result.get_total_impacts():
                ref = getattr(item, "impact_category", None)
                name = getattr(ref, "name", None) or getattr(ref, "id", None)
                amount = getattr(item, "amount", None)
                if name is not None and amount is not None:
                    impacts[str(name)] = float(amount)
            return impacts
        finally:
            try:
                result.dispose()
            except Exception:
                pass

    def normalize_impacts(self, impacts: dict[str, float], weights: dict[str, float] | None = None) -> tuple[float, dict[str, float]]:
        weights = weights or {}
        norm_map = self.config.normalization_map
        normalized: dict[str, float] = {}
        ebi = 0.0
        for category, amount in impacts.items():
            norm = norm_map.get(category)
            if norm is None:
                category_lower = category.lower()
                for known, known_norm in norm_map.items():
                    known_lower = known.lower()
                    if category_lower == known_lower or category_lower in known_lower or known_lower in category_lower:
                        norm = known_norm
                        category = known
                        break
            if norm is None or abs(norm) < 1e-30:
                continue
            value = float(amount) / float(norm)
            normalized[category] = value
            ebi += float(weights.get(category, 1.0)) * value
        return float(ebi), normalized

    def evaluate(self, candidate: dict[str, Any], prediction_percent: float, options: dict[str, Any] | None = None) -> dict[str, Any]:
        options = dict(options or {})
        eol_mode = str(options.get("eol_mode") or "incineration").lower()
        if eol_mode not in {"incineration", "landfill"}:
            eol_mode = "incineration"
        params = self.build_global_parameters(
            candidate,
            prediction_percent,
            options.get("constants") or {},
            country=options.get("country") or options.get("country_iso3"),
        )
        country_profile = self.country_costs(options.get("country") or options.get("country_iso3"))
        lci = self.calculate_lci(params)
        costs = self.calculate_lcc(params, lci, eol_mode)
        evaluator_mode = str(options.get("environmental_mode") or "proxy").lower()
        fallback_to_proxy = bool(options.get("fallback_to_proxy", True))
        warning = ""
        source = "proxy"
        if evaluator_mode in {"openlca_cloud", "cloud", "remote"}:
            try:
                return self.cloud_evaluate(candidate, prediction_percent, {**options, "eol_mode": eol_mode})
            except Exception as exc:
                if not fallback_to_proxy:
                    raise
                warning = f"Cloud OpenLCA evaluator unavailable, used proxy impacts: {exc}"
                impacts = self.proxy_impact_vector(params, lci, eol_mode)
                source = "proxy_fallback"
        elif evaluator_mode in {"openlca_rest", "gdt_server", "openlca_gdt"}:
            try:
                impacts = self.openlca_rest_impact_vector(params, {**options, "eol_mode": eol_mode})
                source = "openlca_rest"
            except Exception as exc:
                if not fallback_to_proxy:
                    raise
                warning = f"OpenLCA REST evaluator unavailable, used proxy impacts: {exc}"
                impacts = self.proxy_impact_vector(params, lci, eol_mode)
                source = "proxy_fallback"
        elif evaluator_mode == "openlca_ipc":
            try:
                impacts = self.openlca_impact_vector(params, {**options, "eol_mode": eol_mode})
                source = "openlca_ipc"
            except Exception as exc:
                if not fallback_to_proxy:
                    raise
                warning = f"OpenLCA IPC unavailable, used proxy impacts: {exc}"
                impacts = self.proxy_impact_vector(params, lci, eol_mode)
                source = "proxy_fallback"
        else:
            impacts = self.proxy_impact_vector(params, lci, eol_mode)
        ebi, normalized = self.normalize_impacts(impacts, options.get("impact_weights") or {})
        product_system = str(options.get("product_system") or DEFAULT_PRODUCT_SYSTEMS[eol_mode])
        return _json_safe(
            {
                "lcc_total_usd_m3": costs["LCC_total_USD_m3"],
                "ebi": ebi,
                "environmental_source": source,
                "lca_warning": warning,
                "eol_mode": eol_mode,
                "product_system": product_system,
                "impact_method": str(options.get("impact_method") or DEFAULT_IMPACT_METHOD),
                "ipc_port": int(options.get("ipc_port") or DEFAULT_IPC_PORT),
                "cost_breakdown": costs,
                "lci": lci,
                "impact_vector": impacts,
                "normalized_impacts": normalized,
                "country_profile": country_profile,
                "country": country_profile.get("country") if country_profile else options.get("country"),
                "country_iso3": country_profile.get("iso3") if country_profile else options.get("country_iso3"),
                "global_parameters_sent": {k: v for k, v in params.items() if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v))},
            }
        )
