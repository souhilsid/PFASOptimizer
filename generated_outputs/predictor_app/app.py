from __future__ import annotations

import argparse
import gzip
import io
import json
import mimetypes
import os
import pickle
import sys
import threading
import time
import traceback
import uuid
import warnings
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

from pfas_feature_builders import build_dataset1_row, build_dataset2_row, load_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inverse_design_engine import run_inverse_design
from lca_lcc_evaluator import BiocharLcaLccEvaluator
from run_pso_optimization import load_search_spaces, parameter_ranges, run_pso

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parents[1]
ASSETS_PATH = APP_DIR / "app_assets.json"


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PFAS Removal Predictor</title>
  <link rel="icon" type="image/png" href="/files/generated_outputs/predictor_app/assets/aiscia_logo.png">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root {
      --bg: #eef4f6;
      --panel: #ffffff;
      --panel-soft: #f7fbfc;
      --text: #101828;
      --muted: #667085;
      --line: #d8e2e8;
      --accent: #0a6f7d;
      --accent-dark: #063941;
      --teal: #13b8b0;
      --orange: #d97830;
      --blue: #246bfe;
      --red: #b42318;
      --green: #157f3b;
      --ink: #071216;
      --cyan-soft: #dff9fb;
      --shadow: 0 16px 42px rgba(9, 28, 36, 0.10);
      --shadow-soft: 0 8px 22px rgba(9, 28, 36, 0.07);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 16% -8%, rgba(19, 184, 176, 0.14), transparent 34%),
        radial-gradient(circle at 86% 0%, rgba(36, 107, 254, 0.10), transparent 30%),
        var(--bg);
      color: var(--text);
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      font-size: 14px;
    }
    header {
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 14px 22px;
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto;
      gap: 16px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    .subtitle {
      color: var(--muted);
      font-size: 13px;
      margin-top: 3px;
    }
    .tabs {
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: var(--panel);
    }
    .tab {
      min-width: 148px;
      height: 38px;
      border: 0;
      background: transparent;
      color: var(--muted);
      font-weight: 750;
      cursor: pointer;
    }
    .tab.active {
      color: #fff;
      background: var(--accent);
    }
    .app-header {
      position: sticky;
      top: 0;
      z-index: 25;
      background:
        linear-gradient(135deg, rgba(7, 18, 22, 0.98), rgba(6, 57, 65, 0.97)),
        var(--ink);
      color: #fff;
      border-bottom: 1px solid rgba(255, 255, 255, 0.10);
      padding: 14px 22px 12px;
      display: grid;
      grid-template-columns: minmax(330px, 1fr) auto;
      gap: 16px;
      align-items: center;
      box-shadow: 0 18px 40px rgba(5, 18, 24, 0.20);
    }
    .brand-cluster {
      display: grid;
      grid-template-columns: auto minmax(230px, 1fr);
      gap: 16px;
      align-items: center;
      min-width: 0;
    }
    .brand-logo {
      width: min(300px, 28vw);
      height: auto;
      display: block;
    }
    .product-kicker {
      color: rgba(223, 249, 251, 0.88);
      font-size: 11px;
      line-height: 1.2;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 850;
    }
    .app-header h1 {
      margin: 3px 0 0;
      color: #fff;
      font-size: 22px;
      line-height: 1.15;
      letter-spacing: 0;
    }
    .app-header .subtitle {
      color: rgba(255, 255, 255, 0.70);
      max-width: 760px;
      font-size: 13px;
      margin-top: 5px;
    }
    .header-actions {
      display: grid;
      gap: 10px;
      justify-items: end;
    }
    .system-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      justify-content: flex-end;
      max-width: 620px;
    }
    .system-pill {
      min-height: 25px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 3px 9px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.14);
      color: rgba(255, 255, 255, 0.82);
      font-size: 11px;
      font-weight: 800;
      white-space: nowrap;
    }
    .system-pill::before {
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--teal);
      box-shadow: 0 0 0 3px rgba(19, 184, 176, 0.16);
    }
    .main-tabs {
      background: rgba(255, 255, 255, 0.08);
      border-color: rgba(255, 255, 255, 0.14);
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
    }
    .main-tabs .tab {
      color: rgba(255, 255, 255, 0.72);
      min-width: 130px;
      height: 40px;
      padding: 0 15px;
    }
    .main-tabs .tab.active {
      color: var(--ink);
      background: linear-gradient(135deg, #e8ffff, var(--teal));
    }
    .tour-launch {
      height: 36px;
      border: 1px solid rgba(255, 255, 255, 0.20);
      border-radius: 999px;
      padding: 0 13px;
      background: rgba(255, 255, 255, 0.10);
      color: #fff;
      font-weight: 850;
      cursor: pointer;
    }
    .workspace-intro {
      max-width: 1500px;
      margin: 18px auto 0;
      padding: 0 18px;
    }
    .workflow-guide {
      display: grid;
      grid-template-columns: minmax(280px, 0.95fr) repeat(4, minmax(160px, 1fr));
      gap: 12px;
      align-items: stretch;
    }
    .workflow-summary {
      border-radius: 8px;
      border: 1px solid rgba(10, 111, 125, 0.18);
      background: linear-gradient(135deg, #ffffff, #f3fbfc);
      box-shadow: var(--shadow-soft);
      padding: 16px;
    }
    .workflow-summary strong {
      display: block;
      font-size: 15px;
      margin-bottom: 5px;
    }
    .workflow-summary span {
      display: block;
      color: var(--muted);
      line-height: 1.45;
      font-size: 12px;
    }
    .workflow-card {
      min-height: 92px;
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 13px;
      box-shadow: var(--shadow-soft);
      color: var(--text);
      cursor: pointer;
      display: grid;
      align-content: start;
      gap: 6px;
    }
    .workflow-card.active {
      border-color: rgba(19, 184, 176, 0.65);
      box-shadow: 0 0 0 3px rgba(19, 184, 176, 0.14), var(--shadow-soft);
    }
    .workflow-card .step {
      width: 26px;
      height: 26px;
      display: inline-grid;
      place-items: center;
      border-radius: 8px;
      background: #e8f8f8;
      color: var(--accent-dark);
      font-weight: 950;
      font-size: 12px;
    }
    .workflow-card strong {
      font-size: 13px;
    }
    .workflow-card span {
      color: var(--muted);
      line-height: 1.35;
      font-size: 12px;
    }
    main {
      max-width: 1500px;
      margin: 0 auto;
      padding: 18px;
      display: grid;
      grid-template-columns: minmax(0, 1.08fr) minmax(420px, 0.92fr);
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .panel-header {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    .panel-title {
      font-size: 15px;
      font-weight: 850;
      margin: 0;
    }
    .panel-note {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    .form-body { padding: 18px; }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(240px, 1fr) auto auto;
      gap: 10px;
      align-items: end;
      margin-bottom: 16px;
    }
    .optimizer-panel {
      grid-column: 1 / -1;
    }
    .hidden {
      display: none !important;
    }
    .disabled-section {
      opacity: 0.62;
    }
    .phase-main {
      max-width: 1500px;
      margin: 0 auto;
      padding: 18px;
    }
    #predictionPhase {
      display: grid;
      grid-template-columns: minmax(0, 1.08fr) minmax(420px, 0.92fr);
      gap: 18px;
    }
    #optimizationPhase {
      display: block;
    }
    #inversePhase {
      display: block;
    }
    #documentationPhase {
      display: block;
    }
    .prediction-usecase-tabs {
      flex-shrink: 0;
    }
    .optimizer-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 12px;
      align-items: end;
      padding: 18px;
    }
    .inverse-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(130px, 1fr));
      gap: 12px;
      align-items: end;
      padding: 18px;
    }
    .optimizer-output {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      padding: 0 18px 18px;
    }
    .inverse-output {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      padding: 0 18px 18px;
    }
    .optimizer-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
      overflow: hidden;
    }
    .optimizer-card h3 {
      margin: 0;
      padding: 12px 14px;
      font-size: 14px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    .table-wrap {
      overflow: auto;
      max-height: 360px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      white-space: nowrap;
    }
    th, td {
      padding: 8px 10px;
      border-bottom: 1px solid #e8edf3;
      text-align: left;
    }
    th {
      position: sticky;
      top: 0;
      background: #f8fafc;
      z-index: 1;
      color: #344054;
      font-weight: 850;
    }
    .optimizer-bars {
      padding: 14px;
      display: grid;
      gap: 9px;
    }
    .inverse-history {
      padding: 14px;
      display: grid;
      gap: 9px;
    }
    .progress-wrap {
      padding: 14px 18px 18px;
      display: grid;
      gap: 10px;
    }
    .progress-track {
      height: 14px;
      border-radius: 999px;
      background: #e8edf3;
      overflow: hidden;
      border: 1px solid var(--line);
    }
    .progress-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--teal), var(--blue));
      border-radius: 999px;
      transition: width 220ms ease;
    }
    .progress-meta {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    .stage-list {
      display: grid;
      gap: 6px;
      font-size: 12px;
    }
    .stage-item {
      display: grid;
      grid-template-columns: 74px 1fr 112px;
      gap: 8px;
      align-items: center;
      padding: 6px 0;
      border-bottom: 1px solid #eef1f5;
    }
    .stage-item:last-child { border-bottom: 0; }
    .viz {
      padding: 12px;
      min-height: 230px;
    }
    .viz svg {
      width: 100%;
      height: 230px;
      display: block;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      font-size: 12px;
      color: var(--muted);
      margin-top: 8px;
    }
    .legend span {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--blue);
      display: inline-block;
    }
    .dot.sobol { background: var(--orange); }
    .dot.bo { background: var(--blue); }
    .dot.best { background: var(--green); }
    .mini-bar-row {
      display: grid;
      grid-template-columns: 78px 1fr 58px;
      gap: 10px;
      align-items: center;
      font-size: 12px;
    }
    .mini-fill {
      height: 12px;
      border-radius: 999px;
      background: var(--green);
    }
    .mini-fill.minimize {
      background: var(--red);
    }
    .optimizer-links {
      padding: 0 18px 18px;
      color: var(--muted);
      font-size: 12px;
    }
    .optimizer-links a {
      color: var(--accent-dark);
      font-weight: 800;
      text-decoration: none;
      margin-right: 12px;
    }
    .optimizer-range-section {
      padding: 0 18px 18px;
      display: grid;
      gap: 12px;
    }
    .range-section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .range-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .range-editor {
      display: grid;
      gap: 9px;
      max-height: 310px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
    }
    .lca-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(130px, 1fr));
      gap: 12px;
      align-items: end;
    }
    .checkline {
      grid-column: span 2;
      min-height: 38px;
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .checkline input {
      width: 16px;
      height: 16px;
      margin: 0;
    }
    .lci-table {
      max-height: 330px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .lci-table input {
      width: 120px;
      height: 30px;
      padding: 5px 7px;
      font-size: 12px;
    }
    .source-pill {
      display: inline-flex;
      align-items: center;
      height: 22px;
      padding: 0 8px;
      border-radius: 999px;
      background: #e8f4f6;
      color: var(--accent-dark);
      font-size: 11px;
      font-weight: 850;
    }
    .source-pill.fixed {
      background: #eef2f7;
      color: #475467;
    }
    .pareto-viz svg {
      height: 300px;
    }
    .pareto-point {
      stroke: #fff;
      stroke-width: 1.2;
    }
    .country-map {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
      min-height: 360px;
      overflow: hidden;
    }
    .map-stage {
      display: grid;
      grid-template-columns: minmax(520px, 1.35fr) minmax(280px, 0.65fr);
      gap: 14px;
      align-items: stretch;
    }
    .real-map-stage {
      grid-template-columns: minmax(560px, 1.45fr) minmax(300px, 0.55fr);
    }
    .leaflet-country-map {
      width: 100%;
      height: 420px;
      min-height: 360px;
      border-radius: 8px;
      border: 1px solid #d7dde7;
      overflow: hidden;
      background: #eef4f7;
    }
    .leaflet-container {
      font: inherit;
    }
    .leaflet-control-attribution {
      font-size: 10px;
    }
    .map-loading {
      height: 420px;
      display: grid;
      place-items: center;
      color: var(--muted);
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
      font-size: 13px;
    }
    .map-panel-side {
      display: grid;
      gap: 10px;
      align-content: start;
    }
    .country-badge {
      border-radius: 8px;
      background: linear-gradient(135deg, #e8f4f6, #eef2f7);
      border: 1px solid #d7dde7;
      padding: 12px;
      display: grid;
      align-content: center;
      gap: 5px;
    }
    .country-badge strong {
      font-size: 18px;
    }
    .country-badge span {
      color: var(--muted);
      font-size: 12px;
    }
    .cost-bars {
      display: grid;
      gap: 7px;
    }
    .cost-row {
      display: grid;
      grid-template-columns: 150px 1fr 72px;
      gap: 8px;
      align-items: center;
      font-size: 12px;
    }
    .cost-row .track {
      height: 10px;
    }
    .map-note {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.35;
    }
    .range-row {
      display: grid;
      grid-template-columns: minmax(190px, 1fr) 150px 150px;
      gap: 10px;
      align-items: end;
      padding: 8px;
      border-bottom: 1px solid #eef1f5;
    }
    .range-row:last-child { border-bottom: 0; }
    .range-row label { margin-bottom: 4px; }
    .category-editor {
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      gap: 12px;
    }
    .category-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
      display: grid;
      gap: 8px;
    }
    .category-card select {
      min-height: 150px;
      padding: 6px;
    }
    .category-actions {
      display: flex;
      gap: 8px;
    }
    .category-actions .button {
      height: 32px;
      padding: 0 10px;
      font-size: 12px;
    }
    label {
      display: block;
      font-size: 12px;
      font-weight: 800;
      color: #344054;
      margin-bottom: 6px;
    }
    input, select, textarea {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
      padding: 8px 10px;
      outline: none;
    }
    textarea {
      min-height: 78px;
      resize: vertical;
      font-family: Consolas, Menlo, monospace;
      font-size: 13px;
    }
    input:focus, select:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 109, 122, 0.14);
    }
    .button {
      height: 40px;
      padding: 0 16px;
      border: 0;
      border-radius: 6px;
      font-weight: 850;
      cursor: pointer;
      white-space: nowrap;
    }
    .button.primary {
      background: var(--teal);
      color: #062422;
    }
    .button.secondary {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
    }
    .button:disabled {
      opacity: 0.58;
      cursor: wait;
    }
    .section-title {
      margin: 18px 0 10px;
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0;
      color: var(--muted);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(145px, 1fr));
      gap: 12px;
    }
    .wide { grid-column: 1 / -1; }
    .hint {
      margin-top: 4px;
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .insights {
      display: grid;
      gap: 18px;
    }
    .result-card {
      padding: 18px;
      display: grid;
      grid-template-columns: 180px minmax(0, 1fr);
      gap: 18px;
      align-items: center;
    }
    .gauge {
      width: 164px;
      height: 164px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: conic-gradient(var(--teal) 0deg, var(--teal) 0deg, #e8edf3 0deg 360deg);
      position: relative;
    }
    .gauge::after {
      content: "";
      width: 118px;
      height: 118px;
      background: #fff;
      border-radius: 50%;
      position: absolute;
      box-shadow: inset 0 0 0 1px var(--line);
    }
    .gauge-value {
      position: relative;
      z-index: 1;
      font-size: 34px;
      font-weight: 900;
      color: var(--accent-dark);
      letter-spacing: 0;
    }
    .result-meta h2 {
      margin: 0 0 8px;
      font-size: 22px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 26px;
      border-radius: 999px;
      padding: 4px 10px;
      background: #eef7f6;
      color: #105c57;
      border: 1px solid #cbe8e5;
      font-size: 12px;
      font-weight: 750;
    }
    .chart-card {
      padding: 16px 18px 18px;
    }
    .chart-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .chart-title h3 {
      margin: 0;
      font-size: 15px;
      letter-spacing: 0;
    }
    .chart-title span {
      color: var(--muted);
      font-size: 12px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 160px minmax(140px, 1fr) 70px;
      gap: 10px;
      align-items: center;
      padding: 7px 0;
      border-bottom: 1px solid #eef1f5;
      font-size: 13px;
    }
    .bar-row:last-child { border-bottom: 0; }
    .bar-label {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #344054;
      font-weight: 750;
    }
    .track {
      height: 12px;
      background: #eef1f5;
      border-radius: 999px;
      overflow: hidden;
      position: relative;
    }
    .bar {
      height: 100%;
      width: 0%;
      border-radius: 999px;
      background: var(--blue);
    }
    .range-dot {
      position: absolute;
      top: -2px;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--accent);
      border: 2px solid #fff;
      box-shadow: 0 0 0 1px rgba(0,0,0,0.18);
      transform: translateX(-50%);
    }
    .range-dot.warn { background: var(--orange); }
    .range-dot.bad { background: var(--red); }
    .bar-value {
      text-align: right;
      font-weight: 850;
      color: var(--text);
    }
    .scenario-bars {
      display: grid;
      grid-template-columns: 120px minmax(150px, 1fr) 64px;
      gap: 10px;
      align-items: center;
      margin-top: 8px;
      font-size: 13px;
    }
    .scenario-label { color: #344054; font-weight: 800; }
    .scenario-fill {
      height: 18px;
      border-radius: 999px;
      background: var(--teal);
    }
    .scenario-fill.baseline { background: #98a2b3; }
    .status-list {
      display: grid;
      gap: 8px;
    }
    .status-item {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 9px 10px;
      border: 1px solid #edf0f4;
      border-radius: 6px;
      background: var(--panel-soft);
      font-size: 13px;
    }
    .status-name { color: #344054; font-weight: 750; }
    .status-pill {
      font-size: 11px;
      font-weight: 850;
      border-radius: 999px;
      padding: 3px 8px;
      background: #e8f6ef;
      color: var(--green);
      white-space: nowrap;
    }
    .status-pill.warn { background: #fff2df; color: var(--warn); }
    .status-pill.bad { background: #fee4e2; color: var(--red); }
    .error {
      margin-top: 12px;
      color: var(--red);
      white-space: pre-wrap;
      font-weight: 750;
    }
    .empty {
      color: var(--muted);
      padding: 8px 0;
      font-size: 13px;
    }
    .docs-hero {
      overflow: hidden;
    }
    .docs-banner {
      padding: 20px;
      display: grid;
      grid-template-columns: minmax(280px, 1fr) auto;
      gap: 18px;
      align-items: center;
      border-bottom: 1px solid var(--line);
      background:
        linear-gradient(135deg, rgba(7, 18, 22, 0.96), rgba(10, 111, 125, 0.94)),
        var(--ink);
      color: #fff;
    }
    .docs-banner h2 {
      margin: 0 0 7px;
      font-size: 22px;
      line-height: 1.15;
      letter-spacing: 0;
    }
    .docs-banner p {
      margin: 0;
      max-width: 840px;
      color: rgba(255, 255, 255, 0.74);
      line-height: 1.5;
      font-size: 13px;
    }
    .docs-badge {
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 8px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.08);
      min-width: 210px;
      display: grid;
      gap: 4px;
    }
    .docs-badge strong {
      font-size: 20px;
    }
    .docs-badge span {
      color: rgba(255, 255, 255, 0.70);
      font-size: 12px;
    }
    .doc-grid {
      padding: 18px;
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      gap: 14px;
    }
    .doc-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 15px;
      display: grid;
      gap: 9px;
      align-content: start;
      min-height: 176px;
    }
    .doc-card h3 {
      margin: 0;
      font-size: 15px;
      letter-spacing: 0;
    }
    .doc-card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.48;
      font-size: 12px;
    }
    .doc-card ul {
      margin: 0;
      padding-left: 18px;
      color: #344054;
      font-size: 12px;
      line-height: 1.55;
    }
    .doc-callout {
      margin: 0 18px 18px;
      border-radius: 8px;
      border: 1px solid rgba(19, 184, 176, 0.28);
      background: #ecfbfb;
      padding: 14px 16px;
      color: #18424a;
      line-height: 1.45;
      font-size: 13px;
    }
    .tour-backdrop {
      position: fixed;
      inset: 0;
      z-index: 80;
      background: rgba(4, 12, 16, 0.58);
      pointer-events: auto;
    }
    .tour-spotlight {
      position: fixed;
      border-radius: 10px;
      box-shadow: 0 0 0 9999px rgba(4, 12, 16, 0.58), 0 0 0 3px rgba(19, 184, 176, 0.90);
      background: transparent;
      transition: top 180ms ease, left 180ms ease, width 180ms ease, height 180ms ease;
      pointer-events: none;
    }
    .tour-card {
      position: fixed;
      width: min(390px, calc(100vw - 28px));
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 22px 70px rgba(0, 0, 0, 0.28);
      border: 1px solid rgba(255, 255, 255, 0.70);
      padding: 16px;
      display: grid;
      gap: 11px;
      transition: top 180ms ease, left 180ms ease;
    }
    .tour-card h3 {
      margin: 0;
      font-size: 17px;
      line-height: 1.2;
    }
    .tour-card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
      font-size: 13px;
    }
    .tour-progress {
      height: 7px;
      border-radius: 999px;
      background: #e8edf3;
      overflow: hidden;
    }
    .tour-progress div {
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--teal), var(--blue));
    }
    .tour-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .tour-actions .left,
    .tour-actions .right {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .tour-count {
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }
    @media (max-width: 1120px) {
      .app-header { grid-template-columns: 1fr; }
      .header-actions { justify-items: start; }
      .system-pills { justify-content: flex-start; }
      .workflow-guide { grid-template-columns: repeat(2, minmax(220px, 1fr)); }
      .workflow-summary { grid-column: 1 / -1; }
      main { grid-template-columns: 1fr; }
      #predictionPhase { grid-template-columns: 1fr; }
      .grid { grid-template-columns: repeat(2, minmax(145px, 1fr)); }
      .optimizer-grid { grid-template-columns: repeat(2, minmax(145px, 1fr)); }
      .inverse-grid { grid-template-columns: repeat(2, minmax(145px, 1fr)); }
      .lca-grid { grid-template-columns: repeat(2, minmax(145px, 1fr)); }
      .optimizer-output { grid-template-columns: 1fr; }
      .inverse-output { grid-template-columns: 1fr; }
      .doc-grid { grid-template-columns: repeat(2, minmax(220px, 1fr)); }
      .category-editor { grid-template-columns: 1fr; }
      .range-row { grid-template-columns: 1fr 1fr; }
      .range-row > div:first-child { grid-column: 1 / -1; }
      .map-stage { grid-template-columns: 1fr; }
      .leaflet-country-map { height: 330px; min-height: 330px; }
    }
    @media (max-width: 620px) {
      header { grid-template-columns: 1fr; }
      .app-header { padding: 12px; }
      .brand-cluster { grid-template-columns: 1fr; gap: 10px; }
      .brand-logo { width: min(250px, 80vw); }
      .main-tabs { width: 100%; overflow-x: auto; }
      .main-tabs .tab { min-width: 118px; }
      .workspace-intro { padding: 0 12px; margin-top: 12px; }
      .workflow-guide { grid-template-columns: 1fr; }
      main { padding: 12px; }
      .toolbar { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
      .lca-grid { grid-template-columns: 1fr; }
      .doc-grid { grid-template-columns: 1fr; padding: 12px; }
      .docs-banner { grid-template-columns: 1fr; padding: 16px; }
      .checkline { grid-column: span 1; }
      .result-card { grid-template-columns: 1fr; justify-items: center; }
      .bar-row { grid-template-columns: 1fr; }
      .bar-value { text-align: left; }
      .tabs { width: 100%; }
      .tab { flex: 1; min-width: 0; }
      .main-tabs .tab { flex: 0 0 auto; min-width: 118px; }
    }
  </style>
</head>
<body>
  <div id="tourBackdrop" class="tour-backdrop hidden" aria-hidden="true">
    <div id="tourSpotlight" class="tour-spotlight"></div>
    <div id="tourCard" class="tour-card" role="dialog" aria-modal="true" aria-labelledby="tourTitle">
      <div class="tour-count" id="tourCount">Step 1 of 7</div>
      <h3 id="tourTitle">Welcome</h3>
      <p id="tourText">This guide explains the PFAS workflow.</p>
      <div class="tour-progress"><div id="tourProgress"></div></div>
      <div class="tour-actions">
        <div class="left">
          <button id="tourSkip" class="button secondary" type="button">Skip</button>
        </div>
        <div class="right">
          <button id="tourPrev" class="button secondary" type="button">Back</button>
          <button id="tourNext" class="button primary" type="button">Next</button>
        </div>
      </div>
    </div>
  </div>

  <header class="app-header">
    <div class="brand-cluster">
      <img class="brand-logo" src="/files/generated_outputs/predictor_app/assets/aiscia_logo.png" alt="AISCIA">
      <div>
        <div class="product-kicker">AISCIA Platform Use Case Module</div>
        <h1>PFAS Removal Decision Engine</h1>
        <div class="subtitle">Prediction, optimization, inverse design, and LCA/LCC screening for biochar and resin PFAS removal workflows.</div>
      </div>
    </div>
    <div class="header-actions">
      <div class="system-pills" aria-label="System status">
        <span class="system-pill">Local PKL evaluators</span>
        <span class="system-pill">Country LCC profiles</span>
        <span class="system-pill">Cloud LCA evaluator</span>
      </div>
      <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap; justify-content:flex-end;">
        <button id="tourStartButton" class="tour-launch" type="button">Guided tour</button>
        <nav class="tabs main-tabs" aria-label="PFAS workflow navigation">
          <button class="tab active" data-phase="prediction">Prediction</button>
          <button class="tab" data-phase="optimization">Optimization</button>
          <button class="tab" data-phase="inverse">Inverse Design</button>
          <button class="tab" data-phase="documentation">Documentation</button>
        </nav>
      </div>
    </div>
  </header>

  <section class="workspace-intro" aria-label="Workflow summary">
    <div class="workflow-guide" data-tour="workflow">
      <div class="workflow-summary">
        <strong>End-to-end PFAS treatment workflow</strong>
        <span>Use the saved ML models to estimate removal, search operating windows, and screen target-matching designs with cost and environmental objectives.</span>
      </div>
      <button class="workflow-card active" data-phase="prediction" type="button">
        <span class="step">1</span>
        <strong>Predict</strong>
        <span>Estimate removal efficiency for one biochar or resin case.</span>
      </button>
      <button class="workflow-card" data-phase="optimization" type="button">
        <span class="step">2</span>
        <strong>Optimize</strong>
        <span>Use PSO to find top minimum or maximum removal candidates.</span>
      </button>
      <button class="workflow-card" data-phase="inverse" type="button">
        <span class="step">3</span>
        <strong>Inverse Design</strong>
        <span>Use BoTorch to hit a target while minimizing cost and EBI.</span>
      </button>
      <button class="workflow-card" data-phase="documentation" type="button">
        <span class="step">?</span>
        <strong>Documentation</strong>
        <span>Read how inputs, outputs, reports, and limits should be used.</span>
      </button>
    </div>
  </section>

  <main id="predictionPhase" class="phase-main">
    <section class="panel">
      <div class="panel-header">
        <div>
          <h2 class="panel-title" id="formTitle">Inputs</h2>
          <div class="panel-note" id="formNote">Enter the case conditions and run a prediction.</div>
        </div>
        <div class="tabs prediction-usecase-tabs">
          <button class="tab active" data-dataset="dataset1">Biochar use case</button>
          <button class="tab" data-dataset="dataset2">Resin use case</button>
        </div>
      </div>
      <div class="form-body">
        <div class="toolbar">
          <div class="panel-note">Use the controls below to define a prediction case.</div>
          <button id="resetButton" class="button secondary">Reset defaults</button>
          <button id="predictButton" class="button primary">Predict</button>
        </div>
        <form id="form"></form>
        <div id="error" class="error"></div>
      </div>
    </section>

    <section class="insights">
      <div class="panel result-card">
        <div class="gauge" id="gauge"><div class="gauge-value" id="prediction">--</div></div>
        <div class="result-meta">
          <h2 id="resultTitle">Run a prediction</h2>
          <div class="panel-note" id="resultNote">The charts update after prediction.</div>
          <div class="chips" id="contextChips"></div>
          <div class="optimizer-links" id="predictionDownloads" style="padding:8px 0 0;"></div>
        </div>
      </div>

      <div class="panel chart-card">
        <div class="chart-title">
          <h3>Scenario Comparison</h3>
          <span>Current case vs default baseline</span>
        </div>
        <div id="scenarioChart"></div>
      </div>

      <div class="panel chart-card">
        <div class="chart-title">
          <h3>Operating Sensitivity</h3>
          <span>Prediction swing across practical ranges</span>
        </div>
        <div id="sensitivityChart" class="empty">Run a prediction to see sensitivity.</div>
      </div>

      <div class="panel chart-card">
        <div class="chart-title">
          <h3>Input Range Check</h3>
          <span>Compared with the training data range</span>
        </div>
        <div id="rangeChart" class="status-list"></div>
      </div>
    </section>
  </main>

  <main id="optimizationPhase" class="phase-main hidden">
    <section class="panel optimizer-panel">
      <div class="panel-header">
        <div>
          <h2 class="panel-title">PSO Optimization</h2>
          <div class="panel-note">Run a single-objective Particle Swarm Optimization search against the saved PKL model.</div>
        </div>
      </div>
      <div class="optimizer-grid">
        <div>
          <label for="optDataset">Use case</label>
          <select id="optDataset">
            <option value="dataset1">Biochar</option>
            <option value="dataset2">Resin</option>
          </select>
        </div>
        <div>
          <label for="optDirection">Objective</label>
          <select id="optDirection">
            <option value="max">Maximize removal</option>
            <option value="min">Minimize removal</option>
          </select>
        </div>
        <div>
          <label for="optParticles">Particle size</label>
          <input id="optParticles" type="number" min="4" max="160" step="1" value="24">
        </div>
        <div>
          <label for="optIterations">Iterations</label>
          <input id="optIterations" type="number" min="1" max="300" step="1" value="30">
        </div>
        <button id="optRunButton" class="button primary">Run PSO</button>
      </div>
      <div class="optimizer-range-section">
        <div class="range-section-header">
          <div>
            <div class="section-title" style="margin:0 0 4px;">Categorical choices</div>
            <div class="panel-note">Select which categorical values PSO is allowed to search.</div>
          </div>
        </div>
        <div id="optCategoryEditor" class="category-editor"></div>
      </div>
      <div class="optimizer-range-section">
        <div class="range-section-header">
          <div>
            <div class="section-title" style="margin:0 0 4px;">Numeric search ranges</div>
            <div class="panel-note">Set the numeric lower and upper bounds that PSO is allowed to search.</div>
          </div>
          <div class="range-actions">
            <button id="optRobustRanges" class="button secondary" type="button">Robust defaults</button>
            <button id="optFullRanges" class="button secondary" type="button">Full training range</button>
          </div>
        </div>
        <div id="optRangeEditor" class="range-editor"></div>
        <div id="optError" class="error"></div>
      </div>
      <div class="optimizer-links" id="optLinks">Ranges and candidates will appear after running PSO.</div>
      <div class="optimizer-output">
        <div class="optimizer-card">
          <h3>Run progress</h3>
          <div class="progress-wrap">
            <div class="progress-track"><div id="optProgressFill" class="progress-fill"></div></div>
            <div class="progress-meta">
              <span id="optProgressText">Waiting for a run.</span>
              <span id="optProgressPct">0%</span>
            </div>
          </div>
        </div>
        <div class="optimizer-card">
          <h3>Convergence</h3>
          <div id="optConvergence" class="viz"><div class="empty">No optimization run yet.</div></div>
        </div>
      </div>
      <div class="optimizer-output">
        <div class="optimizer-card">
          <h3>Top 5 candidates</h3>
          <div id="optTable" class="table-wrap"><div class="empty" style="padding:14px;">No optimization run yet.</div></div>
        </div>
        <div class="optimizer-card">
          <h3>Top candidates chart</h3>
          <div id="optBars" class="optimizer-bars"><div class="empty">No optimization run yet.</div></div>
        </div>
      </div>
      <div class="inverse-output">
        <div class="optimizer-card">
          <h3>Candidate scatter</h3>
          <div id="optScatter" class="viz"><div class="empty">No candidates yet.</div></div>
        </div>
        <div class="optimizer-card">
          <h3>Optimization parallel coordinates</h3>
          <div id="optParallel" class="viz"><div class="empty">No candidates yet.</div></div>
        </div>
      </div>
    </section>
  </main>

  <main id="inversePhase" class="phase-main hidden">
    <section class="panel optimizer-panel">
      <div class="panel-header">
        <div>
          <h2 class="panel-title">Inverse Design</h2>
          <div class="panel-note">Search for PFAS treatment conditions that hit a target removal value or target removal range using the saved PKL model as evaluator.</div>
        </div>
      </div>
      <div class="inverse-grid">
        <div>
          <label for="invDataset">Use case</label>
          <select id="invDataset">
            <option value="dataset1">Biochar</option>
            <option value="dataset2">Resin</option>
          </select>
        </div>
        <div>
          <label for="invTargetMode">Target type</label>
          <select id="invTargetMode">
            <option value="target_value">Target value</option>
            <option value="target_range">Target range</option>
          </select>
        </div>
        <div class="inv-target-value-field">
          <label for="invTargetValue">Target removal (%)</label>
          <input id="invTargetValue" type="number" min="0" max="100" step="any" value="90">
        </div>
        <div class="inv-target-value-field">
          <label for="invTolerance">Tolerance</label>
          <input id="invTolerance" type="number" min="0" step="any" value="2">
        </div>
        <div class="inv-target-value-field">
          <label for="invToleranceMode">Tolerance mode</label>
          <select id="invToleranceMode">
            <option value="absolute">Absolute</option>
            <option value="percent">Percent</option>
          </select>
        </div>
        <div class="inv-target-range-field hidden">
          <label for="invTargetMin">Target min (%)</label>
          <input id="invTargetMin" type="number" min="0" max="100" step="any" value="85">
        </div>
        <div class="inv-target-range-field hidden">
          <label for="invTargetMax">Target max (%)</label>
          <input id="invTargetMax" type="number" min="0" max="100" step="any" value="95">
        </div>
      </div>
      <div class="inverse-grid" style="padding-top:0;">
        <div>
          <label for="invInitStrategy">Initialization</label>
          <select id="invInitStrategy">
            <option value="sobol">Sobol</option>
            <option value="uniform">Uniform</option>
            <option value="none">None</option>
          </select>
        </div>
        <div>
          <label for="invInitTrials">Initialization trials</label>
          <input id="invInitTrials" type="number" min="0" max="256" step="1" value="16">
        </div>
        <div>
          <label for="invExecutionMode">Execution mode</label>
          <select id="invExecutionMode">
            <option value="batch">Batch</option>
            <option value="sequential">Sequential</option>
          </select>
        </div>
        <div>
          <label for="invIterations">Iterations</label>
          <input id="invIterations" type="number" min="1" max="300" step="1" value="20">
        </div>
        <div>
          <label for="invBatchSize">Batch size</label>
          <input id="invBatchSize" type="number" min="1" max="50" step="1" value="5">
        </div>
        <div>
          <label for="invParticles">BoTorch raw samples</label>
          <input id="invParticles" type="number" min="16" max="512" step="1" value="64">
        </div>
        <button id="invRunButton" class="button primary">Run Inverse Design</button>
      </div>

      <div class="optimizer-range-section" id="lcaLccSection">
        <div class="range-section-header">
          <div>
            <div class="section-title" style="margin:0 0 4px;">Sustainability objectives</div>
            <div class="panel-note">For the Biochar use case, add LCC and normalized ReCiPe EBI as minimization objectives beside target-distance.</div>
          </div>
        </div>
        <div class="lca-grid">
          <div>
            <label>Country cost profile</label>
            <input id="countrySearch" type="text" placeholder="Search country on map">
            <input id="invCountry" type="hidden" value="QAT">
          </div>
          <div>
            <label for="invObjectiveMode">Objective mode</label>
            <select id="invObjectiveMode">
              <option value="target_lca_lcc_pareto">Target + LCC + EBI Pareto</option>
              <option value="target_only">Target distance only</option>
            </select>
          </div>
          <div>
            <label for="invEnvironmentalMode">Environmental evaluator</label>
            <select id="invEnvironmentalMode">
              <option value="openlca_cloud">Cloud OpenLCA evaluator</option>
              <option value="proxy">Offline proxy EBI</option>
              <option value="openlca_ipc">Advanced: local OpenLCA IPC</option>
            </select>
          </div>
          <div id="localIpcField">
            <label for="invIpcPort">Local IPC port</label>
            <input id="invIpcPort" type="number" min="1" max="65535" step="1" value="8080">
          </div>
          <div>
            <label for="invEolMode">End-of-life route</label>
            <select id="invEolMode">
              <option value="incineration">Incineration</option>
              <option value="landfill">Landfill</option>
            </select>
          </div>
          <div>
            <label for="invProductSystem">Product system</label>
            <input id="invProductSystem" type="text" value="PS_Biochar_PFAS_Incineration">
          </div>
          <div>
            <label for="invImpactMethod">LCIA method</label>
            <input id="invImpactMethod" type="text" value="ReCiPe Midpoint (H)">
          </div>
          <label class="checkline">
            <input id="invFallbackProxy" type="checkbox" checked>
            <span>Fallback to proxy EBI if evaluator is unavailable</span>
          </label>
          <button id="invTestOpenLca" class="button secondary" type="button">Test evaluator</button>
        </div>
        <div id="countryMapPanel" class="country-map"><div class="empty">Select a country to load cost profile.</div></div>
        <div class="panel-note" id="lcaLccNote">Constants below are sent with every candidate. Candidate-driven values come from the search space and PKL prediction.</div>
        <div id="lcaConstantEditor" class="lci-table"><div class="empty" style="padding:14px;">Loading LCI/LCC constants...</div></div>
      </div>

      <div class="optimizer-range-section">
        <div class="range-section-header">
          <div>
            <div class="section-title" style="margin:0 0 4px;">Categorical choices</div>
            <div class="panel-note">Select which categorical values inverse design can use.</div>
          </div>
        </div>
        <div id="invCategoryEditor" class="category-editor"></div>
      </div>
      <div class="optimizer-range-section">
        <div class="range-section-header">
          <div>
            <div class="section-title" style="margin:0 0 4px;">Numeric search ranges</div>
            <div class="panel-note">Set the numeric lower and upper bounds for inverse design.</div>
          </div>
          <div class="range-actions">
            <button id="invRobustRanges" class="button secondary" type="button">Robust defaults</button>
            <button id="invFullRanges" class="button secondary" type="button">Full training range</button>
          </div>
        </div>
        <div id="invRangeEditor" class="range-editor"></div>
        <div id="invError" class="error"></div>
      </div>
      <div class="optimizer-links" id="invLinks">Configure the target and search space, then run inverse design.</div>
      <div class="optimizer-output">
        <div class="optimizer-card">
          <h3>Run progress</h3>
          <div class="progress-wrap">
            <div class="progress-track"><div id="invProgressFill" class="progress-fill"></div></div>
            <div class="progress-meta">
              <span id="invProgressText">Waiting for a run.</span>
              <span id="invProgressPct">0%</span>
            </div>
            <div id="invStageList" class="stage-list"><div class="empty">No stages yet.</div></div>
          </div>
        </div>
        <div class="optimizer-card">
          <h3>Source contribution</h3>
          <div id="invSourceChart" class="viz"><div class="empty">No candidates yet.</div></div>
        </div>
      </div>
      <div class="inverse-output">
        <div class="optimizer-card">
          <h3>Best matching candidates</h3>
          <div id="invTable" class="table-wrap"><div class="empty" style="padding:14px;">No inverse-design run yet.</div></div>
        </div>
        <div class="optimizer-card">
          <h3>Distance evolution</h3>
          <div id="invHistory" class="inverse-history"><div class="empty">No inverse-design run yet.</div></div>
        </div>
      </div>
      <div class="inverse-output">
        <div class="optimizer-card">
          <h3>Sobol vs BoTorch exploration</h3>
          <div id="invScatter" class="viz"><div class="empty">No candidates yet.</div></div>
        </div>
        <div class="optimizer-card">
          <h3>Prediction target fit</h3>
          <div id="invPredictionChart" class="viz"><div class="empty">No candidates yet.</div></div>
        </div>
      </div>
      <div class="optimizer-output">
        <div class="optimizer-card">
          <h3>3-objective Pareto front</h3>
          <div id="invPareto3D" class="viz pareto-viz"><div class="empty">Enable LCC/EBI objectives to see the Pareto front.</div></div>
        </div>
        <div class="optimizer-card">
          <h3>LCA/LCC objective summary</h3>
          <div id="invLcaSummary" class="optimizer-bars"><div class="empty">No sustainability run yet.</div></div>
        </div>
      </div>
      <div class="inverse-output">
        <div class="optimizer-card">
          <h3>Environmental impact breakdown</h3>
          <div id="invImpactBreakdown" class="optimizer-bars"><div class="empty">No environmental impact data yet.</div></div>
        </div>
        <div class="optimizer-card">
          <h3>Objective scatter</h3>
          <div id="invObjectiveScatter" class="viz"><div class="empty">No objective scatter yet.</div></div>
        </div>
      </div>
      <div class="optimizer-card" style="margin:0 18px 18px;">
        <h3>Parallel coordinates</h3>
        <div id="invParallel" class="viz"><div class="empty">No candidate data yet.</div></div>
      </div>
    </section>
  </main>

  <main id="documentationPhase" class="phase-main hidden">
    <section class="panel docs-hero" data-tour="documentation">
      <div class="docs-banner">
        <div>
          <h2>Documentation Guide</h2>
          <p>This guide explains how to use the PFAS Removal Decision Engine, what each section does, and how to interpret the generated prediction, optimization, inverse-design, LCC, and environmental outputs.</p>
        </div>
        <div class="docs-badge">
          <strong>4 workflows</strong>
          <span>Prediction, PSO optimization, BoTorch inverse design, and LCA/LCC reporting.</span>
        </div>
      </div>
      <div class="doc-grid">
        <article class="doc-card">
          <h3>1. Prediction</h3>
          <p>Use this when you already know the operating case and want the model to estimate PFAS removal efficiency.</p>
          <ul>
            <li>Select Biochar or Resin.</li>
            <li>Enter PFAS/material and operating inputs.</li>
            <li>Check scenario, sensitivity, and range diagnostics.</li>
          </ul>
        </article>
        <article class="doc-card">
          <h3>2. Optimization</h3>
          <p>Use this when the goal is simply to maximize or minimize predicted removal within selected search bounds.</p>
          <ul>
            <li>Set particle size and iterations.</li>
            <li>Choose categorical values allowed in the search.</li>
            <li>Review top 5 candidates and convergence plots.</li>
          </ul>
        </article>
        <article class="doc-card">
          <h3>3. Inverse Design</h3>
          <p>Use this when you need conditions that hit a target removal value or a target removal range.</p>
          <ul>
            <li>Set target and tolerance.</li>
            <li>Use Sobol initialization and BoTorch acquisition.</li>
            <li>Inspect distance evolution and candidate fit.</li>
          </ul>
        </article>
        <article class="doc-card">
          <h3>4. Sustainability Objectives</h3>
          <p>For the Biochar use case, inverse design can search for candidates that balance removal target distance, LCC, and normalized EBI.</p>
          <ul>
            <li>Distance is minimized against the target removal.</li>
            <li>LCC is minimized using country cost profiles.</li>
            <li>EBI is minimized using ReCiPe normalization values.</li>
          </ul>
        </article>
        <article class="doc-card">
          <h3>Country Cost Map</h3>
          <p>The real map drives the country-specific cost constants sent to the LCC evaluator.</p>
          <ul>
            <li>Click a country polygon to select its profile.</li>
            <li>Search narrows visual emphasis on the map.</li>
            <li>Constants update before optimization runs.</li>
          </ul>
        </article>
        <article class="doc-card">
          <h3>Cloud OpenLCA</h3>
          <p>The deployed platform calls a private evaluator API, which can connect to a headless openLCA/gdt-server engine or use proxy fallback for screening runs.</p>
          <ul>
            <li>No local openLCA install is required for users.</li>
            <li>Use Test cloud evaluator before a full run.</li>
            <li>Keep fallback enabled for demos.</li>
          </ul>
        </article>
        <article class="doc-card">
          <h3>Search Bounds</h3>
          <p>Bounds protect the model from unrealistic extrapolation and help it generalize instead of memorizing observed rows.</p>
          <ul>
            <li>Robust defaults use practical trimmed ranges.</li>
            <li>Full training range uses all observed ranges.</li>
            <li>Categorical choices should match feasible materials.</li>
          </ul>
        </article>
        <article class="doc-card">
          <h3>Reports and Exports</h3>
          <p>Each major workflow can generate files that are easier to share outside the live demo.</p>
          <ul>
            <li>Prediction reports summarize one case.</li>
            <li>Optimization exports include ranked candidates.</li>
            <li>Inverse exports include Pareto and LCA/LCC outputs.</li>
          </ul>
        </article>
        <article class="doc-card">
          <h3>Model Use Limits</h3>
          <p>These models are decision-support tools. The UI highlights range checks and candidate diagnostics so users can avoid over-trusting extrapolated results.</p>
          <ul>
            <li>Use range warnings seriously.</li>
            <li>Validate high-value candidates experimentally.</li>
            <li>Use LCA/LCC outputs for screening, not final procurement.</li>
          </ul>
        </article>
      </div>
      <div class="doc-callout">
        Recommended demo flow: start with Prediction to explain model inputs, move to Optimization for top candidate discovery, then use Inverse Design with country cost profiles and LCA/LCC objectives for decision-ready PFAS treatment screening.
      </div>
    </section>
  </main>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const labels = {
      "Pyrolysis temperature": "Pyrolysis temperature",
      "Pyrolysis time": "Pyrolysis time",
      "Heating rated": "Heating rate",
      "C": "Carbon",
      "Ash": "Ash",
      "H/C": "H/C",
      "O/C": "O/C",
      "(O+N)/C": "(O+N)/C",
      "Surface area": "Surface area",
      "Average pore size": "Average pore size",
      "Pore volume": "Pore volume",
      "Solution pH": "Solution pH",
      "Adsorption time": "Adsorption time",
      "Adsorption temperature": "Adsorption temperature",
      "S/L": "Solid/liquid ratio",
      "Initial concentration": "Initial concentration",
      "CaCl2": "CaCl2",
      "NaCl": "NaCl",
      "HA": "Humic acid",
      "PFAS": "PFAS",
      "Solution": "Water matrix",
      "Resin": "Resin",
      "Polymer_matrix ": "Polymer matrix",
      "Porosity": "Porosity",
      "Functional group": "Functional group",
      "Resin_type": "Resin type",
      "Resin profile": "Resin profile",
      "initial_pfas_concentration_ug_L": "Initial PFAS concentration",
      "resin_dosage_mg_L": "Resin dosage",
      "temperature_C": "Temperature",
      "contact_time_h": "Contact time",
      "stirring_rate_rpm": "Stirring rate",
      "CDOC_mg_L": "DOC",
      "distance": "Target distance",
      "target_error": "Target error",
      "lcc_total_usd_m3": "LCC",
      "ebi": "Normalized EBI",
      "pareto_front": "Pareto front",
      "environmental_source": "Environmental source",
      "eol_mode": "End of life",
      "cost_p01_USD": "P01 cost",
      "cost_p02_USD": "P02 cost",
      "cost_p03_USD": "P03 cost",
      "cost_p04_USD": "P04 cost",
      "cost_p05_USD": "P05 incineration cost",
      "cost_p06_USD": "P06 landfill cost",
      "m_biochar_kg": "Biochar dose",
      "m_PFAS_removed_kg": "PFAS removed",
      "m_spent_biochar_kg": "Spent biochar"
    };
    const units = {
      "Pyrolysis temperature": "C",
      "Pyrolysis time": "min",
      "Heating rated": "C/min",
      "C": "%",
      "Ash": "%",
      "Surface area": "m2/g",
      "Average pore size": "nm",
      "Pore volume": "cm3/g",
      "Solution pH": "",
      "Adsorption time": "min",
      "Adsorption temperature": "C",
      "Initial concentration": "mg/L",
      "CaCl2": "mM",
      "NaCl": "mM",
      "HA": "mg/L",
      "initial_pfas_concentration_ug_L": "ug/L",
      "resin_dosage_mg_L": "mg/L",
      "temperature_C": "C",
      "contact_time_h": "h",
      "stirring_rate_rpm": "rpm",
      "CDOC_mg_L": "mg/L",
      "lcc_total_usd_m3": "USD/m3",
      "cost_p01_USD": "USD",
      "cost_p02_USD": "USD",
      "cost_p03_USD": "USD",
      "cost_p04_USD": "USD",
      "cost_p05_USD": "USD",
      "cost_p06_USD": "USD",
      "m_biochar_kg": "kg",
      "m_PFAS_removed_kg": "kg",
      "m_spent_biochar_kg": "kg"
    };
    let assets = null;
    let optimizationSpace = null;
    let activeDataset = "dataset1";
    let lastValues = {};

    const form = document.getElementById("form");
    const errorBox = document.getElementById("error");

    function fmt(value, digits = 2) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
      return Number(value).toFixed(digits).replace(/\.?0+$/, "");
    }
    function labelText(key) {
      const unit = units[key];
      return `${labels[key] || key}${unit ? " (" + unit + ")" : ""}`;
    }
    function fieldId(key) {
      return "field_" + key.replace(/[^a-zA-Z0-9]/g, "_");
    }
    function rangeHint(key, ds) {
      const r = ds.ranges?.[key];
      if (!r || r.p05 === null || r.p95 === null) return "";
      return `Typical range: ${fmt(r.p05)} - ${fmt(r.p95)}`;
    }
    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[s]));
    }
    function inputField(key, value, ds) {
      const hint = rangeHint(key, ds);
      return `<div><label for="${fieldId(key)}">${labelText(key)}</label><input data-key="${escapeHtml(key)}" id="${fieldId(key)}" type="number" step="any" value="${fmt(value, 6)}">${hint ? `<div class="hint">${hint}</div>` : ""}</div>`;
    }
    function selectField(key, options, value) {
      const opts = (options || []).map(v => `<option value="${escapeHtml(v)}" ${String(v) === String(value) ? "selected" : ""}>${escapeHtml(v)}</option>`).join("");
      return `<div><label for="${fieldId(key)}">${labelText(key)}</label><select data-key="${escapeHtml(key)}" id="${fieldId(key)}">${opts}</select></div>`;
    }
    function datasetDefaults(name) {
      const ds = assets.datasets[name];
      if (name === "dataset1") return {...ds.defaults};
      return {
        ...ds.defaults,
        initial_pfas_concentration_ug_L: (ds.defaults.initial_concentration_mg_L ?? 0) * 1000,
        resin_dosage_mg_L: (ds.defaults.resin_dosage_g_L ?? 0) * 1000,
        temperature_C: ds.defaults["temperature\n(℃)"],
        contact_time_h: ds.defaults["contact_time\n(h)"],
        stirring_rate_rpm: ds.defaults.Stirring_rate_numeric,
        CDOC_mg_L: ds.defaults["CDOC\n(mg/L)"]
      };
    }
    function finiteOr(value, fallback) {
      const n = Number(value);
      return Number.isFinite(n) ? n : fallback;
    }
    function rangeDefaults(r) {
      const min = finiteOr(r?.min, 0);
      const max = finiteOr(r?.max, min);
      let lower = finiteOr(r?.p05, min);
      let upper = finiteOr(r?.p95, max);
      if (lower > upper) [lower, upper] = [upper, lower];
      return {min, max, lower, upper};
    }
    function optimizationRangeRow(key, ds) {
      const r = rangeDefaults(ds.ranges?.[key] || {});
      return `
        <div class="range-row" data-opt-field="${escapeHtml(key)}">
          <div>
            <label>${labelText(key)}</label>
            <div class="hint">Training: ${fmt(r.min, 4)} - ${fmt(r.max, 4)} | robust: ${fmt(r.lower, 4)} - ${fmt(r.upper, 4)}</div>
          </div>
          <div>
            <label>Lower bound</label>
            <input class="opt-lower" type="number" step="any" value="${fmt(r.lower, 8)}">
          </div>
          <div>
            <label>Upper bound</label>
            <input class="opt-upper" type="number" step="any" value="${fmt(r.upper, 8)}">
          </div>
        </div>
      `;
    }
    function renderOptimizationRanges(name) {
      if (!assets) return;
      const ds = assets.datasets[name];
      const fields = ds.numeric_fields || [];
      document.getElementById("optRangeEditor").innerHTML = fields.map(key => optimizationRangeRow(key, ds)).join("");
      document.getElementById("optError").textContent = "";
      document.getElementById("optLinks").textContent = "Edit the search ranges, then run PSO.";
    }
    function renderOptimizationCategories(space) {
      optimizationSpace = space;
      const cats = space?.categorical || [];
      const root = document.getElementById("optCategoryEditor");
      if (!cats.length) {
        root.innerHTML = '<div class="empty" style="padding:14px;">No categorical parameters for this use case.</div>';
        return;
      }
      root.innerHTML = cats.map(cat => {
        const options = (cat.choices || []).map(choice => `<option value="${escapeHtml(choice.id)}" selected>${escapeHtml(choice.label)}</option>`).join("");
        const size = Math.max(4, Math.min(9, (cat.choices || []).length));
        return `
          <div class="category-card" data-category-card="${escapeHtml(cat.name)}">
            <div>
              <label>${escapeHtml(cat.label || labelText(cat.name))}</label>
              <div class="hint"><span class="category-count">${(cat.choices || []).length}</span> of ${(cat.choices || []).length} selected</div>
            </div>
            <select class="opt-category-select" data-opt-category="${escapeHtml(cat.name)}" multiple size="${size}">${options}</select>
            <div class="category-actions">
              <button class="button secondary category-all" type="button">All</button>
              <button class="button secondary category-none" type="button">None</button>
            </div>
          </div>
        `;
      }).join("");
      root.querySelectorAll(".opt-category-select").forEach(select => {
        select.addEventListener("change", () => updateCategoryCount(select));
        updateCategoryCount(select);
      });
      root.querySelectorAll(".category-all").forEach(button => {
        button.addEventListener("click", () => {
          const select = button.closest(".category-card").querySelector(".opt-category-select");
          Array.from(select.options).forEach(option => option.selected = true);
          updateCategoryCount(select);
        });
      });
      root.querySelectorAll(".category-none").forEach(button => {
        button.addEventListener("click", () => {
          const select = button.closest(".category-card").querySelector(".opt-category-select");
          Array.from(select.options).forEach(option => option.selected = false);
          updateCategoryCount(select);
        });
      });
    }
    function updateCategoryCount(select) {
      const card = select.closest(".category-card");
      const count = Array.from(select.selectedOptions).length;
      const total = select.options.length;
      card.querySelector(".category-count").textContent = count;
      card.querySelector(".hint").lastChild.textContent = ` of ${total} selected`;
    }
    async function loadOptimizationSpace(name) {
      document.getElementById("optCategoryEditor").innerHTML = '<div class="empty" style="padding:14px;">Loading categorical choices...</div>';
      const res = await fetch(`/api/optimization-space?dataset=${encodeURIComponent(name)}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Could not load optimization search space");
      renderOptimizationCategories(data);
    }
    function renderOptimizationSetup(name) {
      renderOptimizationRanges(name);
      loadOptimizationSpace(name).catch(err => {
        document.getElementById("optCategoryEditor").innerHTML = `<div class="error" style="padding:14px;">${escapeHtml(err.message || err)}</div>`;
      });
    }
    function setOptimizationRangeMode(mode) {
      const ds = assets.datasets[document.getElementById("optDataset").value];
      document.querySelectorAll("#optRangeEditor [data-opt-field]").forEach(row => {
        const key = row.dataset.optField;
        const r = rangeDefaults(ds.ranges?.[key] || {});
        const lower = mode === "full" ? r.min : r.lower;
        const upper = mode === "full" ? r.max : r.upper;
        row.querySelector(".opt-lower").value = fmt(lower, 8);
        row.querySelector(".opt-upper").value = fmt(upper, 8);
      });
      document.getElementById("optError").textContent = "";
    }
    function collectOptimizationRanges() {
      const ranges = {};
      const errors = [];
      document.querySelectorAll("#optRangeEditor [data-opt-field]").forEach(row => {
        const key = row.dataset.optField;
        const lowerText = row.querySelector(".opt-lower").value;
        const upperText = row.querySelector(".opt-upper").value;
        const lower = Number(lowerText);
        const upper = Number(upperText);
        if (!Number.isFinite(lower) || !Number.isFinite(upper)) {
          errors.push(`${labelText(key)} needs numeric lower and upper bounds.`);
          return;
        }
        if (lower > upper) {
          errors.push(`${labelText(key)} lower bound is greater than upper bound.`);
          return;
        }
        ranges[key] = {lower, upper};
      });
      if (errors.length) throw new Error(errors.join(" "));
      return ranges;
    }
    function collectOptimizationCategories() {
      const categories = {};
      const errors = [];
      document.querySelectorAll("#optCategoryEditor .opt-category-select").forEach(select => {
        const key = select.dataset.optCategory;
        const selected = Array.from(select.selectedOptions).map(option => option.value);
        if (!selected.length) {
          errors.push(`${labelText(key)} needs at least one selected value.`);
          return;
        }
        categories[key] = selected;
      });
      if (errors.length) throw new Error(errors.join(" "));
      return categories;
    }
    function inverseRangeRow(key, ds) {
      const r = rangeDefaults(ds.ranges?.[key] || {});
      return `
        <div class="range-row" data-inv-field="${escapeHtml(key)}">
          <div>
            <label>${labelText(key)}</label>
            <div class="hint">Training: ${fmt(r.min, 4)} - ${fmt(r.max, 4)} | robust: ${fmt(r.lower, 4)} - ${fmt(r.upper, 4)}</div>
          </div>
          <div>
            <label>Lower bound</label>
            <input class="inv-lower" type="number" step="any" value="${fmt(r.lower, 8)}">
          </div>
          <div>
            <label>Upper bound</label>
            <input class="inv-upper" type="number" step="any" value="${fmt(r.upper, 8)}">
          </div>
        </div>
      `;
    }
    function renderInverseRanges(name) {
      if (!assets) return;
      const ds = assets.datasets[name];
      document.getElementById("invRangeEditor").innerHTML = (ds.numeric_fields || []).map(key => inverseRangeRow(key, ds)).join("");
      document.getElementById("invError").textContent = "";
      document.getElementById("invLinks").textContent = "Configure the target and search space, then run inverse design.";
    }
    function setInverseRangeMode(mode) {
      const ds = assets.datasets[document.getElementById("invDataset").value];
      document.querySelectorAll("#invRangeEditor [data-inv-field]").forEach(row => {
        const key = row.dataset.invField;
        const r = rangeDefaults(ds.ranges?.[key] || {});
        const lower = mode === "full" ? r.min : r.lower;
        const upper = mode === "full" ? r.max : r.upper;
        row.querySelector(".inv-lower").value = fmt(lower, 8);
        row.querySelector(".inv-upper").value = fmt(upper, 8);
      });
      document.getElementById("invError").textContent = "";
    }
    function collectInverseRanges() {
      const ranges = {};
      const errors = [];
      document.querySelectorAll("#invRangeEditor [data-inv-field]").forEach(row => {
        const key = row.dataset.invField;
        const lower = Number(row.querySelector(".inv-lower").value);
        const upper = Number(row.querySelector(".inv-upper").value);
        if (!Number.isFinite(lower) || !Number.isFinite(upper)) {
          errors.push(`${labelText(key)} needs numeric lower and upper bounds.`);
          return;
        }
        if (lower > upper) {
          errors.push(`${labelText(key)} lower bound is greater than upper bound.`);
          return;
        }
        ranges[key] = {lower, upper};
      });
      if (errors.length) throw new Error(errors.join(" "));
      return ranges;
    }
    function renderInverseCategories(space) {
      const cats = space?.categorical || [];
      const root = document.getElementById("invCategoryEditor");
      if (!cats.length) {
        root.innerHTML = '<div class="empty" style="padding:14px;">No categorical parameters for this use case.</div>';
        return;
      }
      root.innerHTML = cats.map(cat => {
        const options = (cat.choices || []).map(choice => `<option value="${escapeHtml(choice.id)}" selected>${escapeHtml(choice.label)}</option>`).join("");
        const size = Math.max(4, Math.min(9, (cat.choices || []).length));
        return `
          <div class="category-card" data-category-card="${escapeHtml(cat.name)}">
            <div>
              <label>${escapeHtml(cat.label || labelText(cat.name))}</label>
              <div class="hint"><span class="category-count">${(cat.choices || []).length}</span> of ${(cat.choices || []).length} selected</div>
            </div>
            <select class="inv-category-select" data-inv-category="${escapeHtml(cat.name)}" multiple size="${size}">${options}</select>
            <div class="category-actions">
              <button class="button secondary category-all" type="button">All</button>
              <button class="button secondary category-none" type="button">None</button>
            </div>
          </div>
        `;
      }).join("");
      root.querySelectorAll(".inv-category-select").forEach(select => {
        select.addEventListener("change", () => updateCategoryCount(select));
        updateCategoryCount(select);
      });
      root.querySelectorAll(".category-all").forEach(button => {
        button.addEventListener("click", () => {
          const select = button.closest(".category-card").querySelector(".inv-category-select");
          Array.from(select.options).forEach(option => option.selected = true);
          updateCategoryCount(select);
        });
      });
      root.querySelectorAll(".category-none").forEach(button => {
        button.addEventListener("click", () => {
          const select = button.closest(".category-card").querySelector(".inv-category-select");
          Array.from(select.options).forEach(option => option.selected = false);
          updateCategoryCount(select);
        });
      });
    }
    async function loadInverseSpace(name) {
      document.getElementById("invCategoryEditor").innerHTML = '<div class="empty" style="padding:14px;">Loading categorical choices...</div>';
      const res = await fetch(`/api/optimization-space?dataset=${encodeURIComponent(name)}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Could not load inverse-design search space");
      renderInverseCategories(data);
    }
    function renderInverseSetup(name) {
      renderInverseRanges(name);
      syncLcaLccControls();
      loadInverseSpace(name).catch(err => {
        document.getElementById("invCategoryEditor").innerHTML = `<div class="error" style="padding:14px;">${escapeHtml(err.message || err)}</div>`;
      });
    }
    function collectInverseCategories() {
      const categories = {};
      const errors = [];
      document.querySelectorAll("#invCategoryEditor .inv-category-select").forEach(select => {
        const key = select.dataset.invCategory;
        const selected = Array.from(select.selectedOptions).map(option => option.value);
        if (!selected.length) {
          errors.push(`${labelText(key)} needs at least one selected value.`);
          return;
        }
        categories[key] = selected;
      });
      if (errors.length) throw new Error(errors.join(" "));
      return categories;
    }
    function syncInverseTargetMode() {
      const mode = document.getElementById("invTargetMode").value;
      document.querySelectorAll(".inv-target-value-field").forEach(el => el.classList.toggle("hidden", mode !== "target_value"));
      document.querySelectorAll(".inv-target-range-field").forEach(el => el.classList.toggle("hidden", mode !== "target_range"));
    }
    function syncInverseExecutionMode() {
      const sequential = document.getElementById("invExecutionMode").value === "sequential";
      const batchInput = document.getElementById("invBatchSize");
      batchInput.disabled = sequential;
      if (sequential) batchInput.value = 1;
    }
    function lcaMeta() {
      return assets?.lca_lcc || {};
    }
    function countryProfiles() {
      return lcaMeta().country_profiles || [];
    }
    function selectedCountryProfile() {
      const value = document.getElementById("invCountry")?.value || "";
      return countryProfiles().find(profile => profile.iso3 === value || profile.country === value) || null;
    }
    const COUNTRY_GEOJSON_URL = "https://cdn.jsdelivr.net/gh/johan/world.geo.json@master/countries.geo.json";
    let countryMap = null;
    let countryGeoLayer = null;
    let countryGeoReady = false;
    let countryGeoLoading = false;
    let countryMapResizeObserver = null;
    const countryNameAliases = {
      "Bahamas": "BHS",
      "Bolivia": "BOL",
      "Bosnia and Herzegovina": "BIH",
      "Brunei": "BRN",
      "Burma": "MMR",
      "Congo": "COG",
      "Democratic Republic of the Congo": "COD",
      "Cote d'Ivoire": "CIV",
      "Czech Republic": "CZE",
      "Dominican Rep.": "DOM",
      "Falkland Islands": "FLK",
      "Guinea Bissau": "GNB",
      "Iran": "IRN",
      "Laos": "LAO",
      "Macedonia": "MKD",
      "Moldova": "MDA",
      "North Korea": "PRK",
      "Russia": "RUS",
      "South Korea": "KOR",
      "Syria": "SYR",
      "Taiwan": "TWN",
      "Tanzania": "TZA",
      "United Kingdom": "GBR",
      "United States of America": "USA",
      "Venezuela": "VEN",
      "Vietnam": "VNM"
    };
    function profileLookup() {
      const lookup = {};
      countryProfiles().forEach(profile => {
        if (profile.iso3) lookup[String(profile.iso3).toUpperCase()] = profile;
        if (profile.country) lookup[String(profile.country).trim().toLowerCase()] = profile;
      });
      return lookup;
    }
    function featureIso(feature) {
      const props = feature?.properties || {};
      return String(feature?.id || props.iso_a3 || props.ISO_A3 || props.ADM0_A3 || props.ISO3 || "").toUpperCase();
    }
    function featureName(feature) {
      const props = feature?.properties || {};
      return String(props.name || props.NAME || props.ADMIN || props.name_long || props.sovereignt || featureIso(feature) || "");
    }
    function profileForFeature(feature) {
      const lookup = profileLookup();
      const iso = featureIso(feature);
      const name = featureName(feature);
      return lookup[iso] || lookup[countryNameAliases[name]] || lookup[String(name).trim().toLowerCase()] || null;
    }
    function countrySearchText() {
      return String(document.getElementById("countrySearch")?.value || "").trim().toLowerCase();
    }
    function countryMatchesSearch(feature, profile) {
      const search = countrySearchText();
      if (!search) return true;
      const name = featureName(feature).toLowerCase();
      const iso = featureIso(feature).toLowerCase();
      const country = String(profile?.country || "").toLowerCase();
      const profileIso = String(profile?.iso3 || "").toLowerCase();
      return name.includes(search) || iso.includes(search) || country.includes(search) || profileIso.includes(search);
    }
    function countryFeatureStyle(feature) {
      const profile = profileForFeature(feature);
      const selected = selectedCountryProfile();
      const hasProfile = Boolean(profile);
      const isSelected = Boolean(profile && selected && profile.iso3 === selected.iso3);
      const matches = countryMatchesSearch(feature, profile);
      return {
        color: isSelected ? "#d97830" : hasProfile ? "#0f6d7a" : "#98a2b3",
        weight: isSelected ? 2.4 : hasProfile ? 1.0 : 0.55,
        fillColor: isSelected ? "#d97830" : hasProfile ? "#16a39a" : "#d8dee6",
        fillOpacity: isSelected ? 0.58 : hasProfile ? (matches ? 0.34 : 0.08) : 0.05,
        opacity: matches ? 1 : 0.32
      };
    }
    function renderCountrySidePanel() {
      const side = document.getElementById("countryProfileSide");
      const profile = selectedCountryProfile();
      const profiles = countryProfiles();
      if (!side) return;
      if (!profiles.length || !profile) {
        side.innerHTML = '<div class="empty">Country cost workbook was not loaded.</div>';
        return;
      }
      const costKeys = [
        ["price_electricity_USD_kWh", "Electricity"],
        ["price_heat_USD_MJ", "Heat"],
        ["price_water_USD_kg", "Water"],
        ["price_transport_USD_tkm", "Transport"],
        ["price_wet_biomass_USD_kg", "Biomass"],
        ["price_incin_USD_kg", "Incineration"]
      ];
      const costs = profile.costs || {};
      const max = Math.max(...costKeys.map(([key]) => Number(costs[key]) || 0), 1e-9);
      const bars = costKeys.map(([key, label]) => {
        const value = Number(costs[key]);
        const width = Number.isFinite(value) ? Math.max(3, value / max * 100) : 0;
        return `<div class="cost-row"><div>${escapeHtml(label)}</div><div class="track"><div class="bar" style="width:${width}%"></div></div><div class="bar-value">${fmt(value, 4)}</div></div>`;
      }).join("");
      side.innerHTML = `
        <div class="country-badge">
          <strong>${escapeHtml(profile.country)} (${escapeHtml(profile.iso3)})</strong>
          <span>${escapeHtml(profile.region || "Region not specified")}</span>
          <span>${escapeHtml(profile.data_quality_overall || "Screening cost profile")}</span>
        </div>
        <div class="cost-bars">${bars}</div>
        <div class="map-note">${escapeHtml(profile.notes || "Click a country with a cost profile to update the LCA/LCC constants.")}</div>
      `;
    }
    function updateCountryMapSelection(fitSelected = false) {
      renderCountrySidePanel();
      if (!countryGeoLayer || !countryMap) return;
      countryGeoLayer.setStyle(countryFeatureStyle);
      const selected = selectedCountryProfile();
      let selectedLayer = null;
      countryGeoLayer.eachLayer(layer => {
        const profile = profileForFeature(layer.feature);
        if (profile && selected && profile.iso3 === selected.iso3) selectedLayer = layer;
      });
      if (selectedLayer) {
        selectedLayer.bringToFront();
        if (fitSelected) countryMap.fitBounds(selectedLayer.getBounds(), { maxZoom: 4, padding: [24, 24] });
      }
    }
    async function loadCountryGeoJson() {
      if (countryGeoReady || countryGeoLoading || !countryMap) return;
      countryGeoLoading = true;
      const side = document.getElementById("countryProfileSide");
      if (side) side.innerHTML = '<div class="map-loading">Loading country boundaries...</div>';
      try {
        const response = await fetch(COUNTRY_GEOJSON_URL);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const geojson = await response.json();
        countryGeoLayer = L.geoJSON(geojson, {
          style: countryFeatureStyle,
          onEachFeature: (feature, layer) => {
            const profile = profileForFeature(feature);
            const label = profile ? `${profile.country} (${profile.iso3})` : featureName(feature);
            layer.bindTooltip(label, { sticky: true });
            layer.on({
              click: () => {
                if (!profile) return;
                document.getElementById("invCountry").value = profile.iso3;
                applyCountryCostsToConstants();
                updateCountryMapSelection(true);
              },
              mouseover: event => event.target.setStyle({ weight: 2.2, opacity: 1, fillOpacity: profile ? 0.52 : 0.12 }),
              mouseout: event => {
                countryGeoLayer.resetStyle(event.target);
                updateCountryMapSelection(false);
              }
            });
          }
        }).addTo(countryMap);
        countryGeoReady = true;
        updateCountryMapSelection(true);
      } catch (error) {
        const sidePanel = document.getElementById("countryProfileSide");
        if (sidePanel) sidePanel.innerHTML = `<div class="error">Could not load country boundaries from the map service. The selected country cost profile is still applied. ${escapeHtml(error.message || error)}</div>`;
        renderCountrySidePanel();
      } finally {
        countryGeoLoading = false;
      }
    }
    function ensureCountryMap() {
      const root = document.getElementById("countryMapPanel");
      const profiles = countryProfiles();
      if (!profiles.length) {
        root.innerHTML = '<div class="empty">Country cost workbook was not loaded.</div>';
        return;
      }
      if (!countryMap) {
        if (!document.getElementById("leafletCountryMap")) {
          root.innerHTML = `
            <div class="map-stage real-map-stage">
              <div id="leafletCountryMap" class="leaflet-country-map"></div>
              <div id="countryProfileSide" class="map-panel-side"></div>
            </div>
          `;
        }
        renderCountrySidePanel();
        if (!window.L) {
          document.getElementById("leafletCountryMap").innerHTML = '<div class="map-loading">Leaflet did not load. Check internet access, then refresh.</div>';
          return;
        }
        const mapElement = document.getElementById("leafletCountryMap");
        if (!mapElement.clientWidth || !mapElement.clientHeight) return;
        countryMap = L.map("leafletCountryMap", {
          center: [22, 15],
          zoom: 2,
          minZoom: 2,
          maxZoom: 6,
          worldCopyJump: true,
          zoomControl: true
        });
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          maxZoom: 6,
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(countryMap);
        if (window.ResizeObserver) {
          countryMapResizeObserver = new ResizeObserver(() => {
            if (countryMap) countryMap.invalidateSize();
          });
          countryMapResizeObserver.observe(mapElement);
        }
        loadCountryGeoJson();
      } else {
        renderCountrySidePanel();
        updateCountryMapSelection(false);
      }
      setTimeout(() => {
        if (countryMap) countryMap.invalidateSize();
      }, 50);
    }
    function refreshCountryMapAfterReveal() {
      setTimeout(() => {
        ensureCountryMap();
        if (countryMap) {
          countryMap.invalidateSize(true);
          updateCountryMapSelection(false);
          setTimeout(() => countryMap.invalidateSize(true), 250);
        }
      }, 80);
    }
    function renderCountrySelector() {
      const profiles = countryProfiles();
      if (!profiles.length) {
        document.getElementById("invCountry").value = "";
        ensureCountryMap();
        return;
      }
      const current = selectedCountryProfile();
      const qatar = profiles.find(profile => profile.iso3 === "QAT");
      document.getElementById("invCountry").value = current?.iso3 || qatar?.iso3 || profiles[0].iso3;
      applyCountryCostsToConstants();
      ensureCountryMap();
      updateCountryMapSelection(true);
    }
    function applyCountryCostsToConstants() {
      const profile = selectedCountryProfile();
      if (!profile || !profile.costs) return;
      Object.entries(profile.costs).forEach(([key, value]) => {
        const input = document.querySelector(`#lcaConstantEditor [data-lca-param="${CSS.escape(key)}"]`);
        if (input && Number.isFinite(Number(value))) input.value = value;
      });
    }
    function renderCountryMap() {
      ensureCountryMap();
      updateCountryMapSelection(false);
    }
    function renderLcaConstants() {
      const meta = lcaMeta();
      const rows = [...(meta.global_parameters || []), ...(meta.cost_parameters || [])];
      const root = document.getElementById("lcaConstantEditor");
      if (!rows.length) {
        root.innerHTML = '<div class="empty" style="padding:14px;">No LCI/LCC parameter metadata was found.</div>';
        return;
      }
      const body = rows.map(row => {
        const editable = Boolean(row.editable);
        const value = row.default ?? "";
        const source = row.source === "candidate" ? "candidate/PKL" : "constant";
        const valueCell = editable
          ? `<input data-lca-param="${escapeHtml(row.name)}" type="number" step="any" value="${escapeHtml(value)}">`
          : `<span class="source-pill fixed">${escapeHtml(source)}</span>`;
        const sourceClass = editable ? "" : "fixed";
        return `
          <tr>
            <td>${escapeHtml(row.name)}</td>
            <td>${valueCell}</td>
            <td>${escapeHtml(row.unit || "")}</td>
            <td><span class="source-pill ${sourceClass}">${escapeHtml(source)}</span></td>
            <td>${escapeHtml(row.meaning || row.role || "")}</td>
          </tr>
        `;
      }).join("");
      root.innerHTML = `
        <table>
          <thead><tr><th>Parameter</th><th>Value sent</th><th>Unit</th><th>Source</th><th>Meaning</th></tr></thead>
          <tbody>${body}</tbody>
        </table>
      `;
      applyCountryCostsToConstants();
    }
    function syncLcaProductSystem() {
      const meta = lcaMeta();
      const eol = document.getElementById("invEolMode").value;
      const defaults = meta.product_systems || {};
      const product = defaults[eol] || defaults.incineration || "";
      if (product) document.getElementById("invProductSystem").value = product;
    }
    function syncLcaEvaluatorMode() {
      const mode = document.getElementById("invEnvironmentalMode").value;
      const localField = document.getElementById("localIpcField");
      if (localField) localField.classList.toggle("hidden", mode !== "openlca_ipc");
      const button = document.getElementById("invTestOpenLca");
      if (button) button.textContent = mode === "openlca_ipc" ? "Test local IPC" : mode === "proxy" ? "Test proxy" : "Test cloud evaluator";
    }
    function syncLcaLccControls() {
      const dataset = document.getElementById("invDataset")?.value || "dataset1";
      const enabled = dataset === "dataset1";
      const section = document.getElementById("lcaLccSection");
      section.classList.toggle("disabled-section", !enabled);
      ["countrySearch","invObjectiveMode","invEnvironmentalMode","invIpcPort","invEolMode","invProductSystem","invImpactMethod","invFallbackProxy"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.disabled = !enabled;
      });
      if (!enabled) {
        document.getElementById("invObjectiveMode").value = "target_only";
        document.getElementById("lcaLccNote").textContent = "LCA/LCC Pareto objectives are configured for the Biochar OpenLCA model. Resin remains target-distance inverse design only.";
        document.getElementById("lcaLccNote").style.color = "var(--muted)";
      } else {
        document.getElementById("lcaLccNote").textContent = "Constants below are sent with every candidate. Candidate-driven values come from the search space and PKL prediction.";
        document.getElementById("lcaLccNote").style.color = "var(--muted)";
      }
      syncLcaEvaluatorMode();
    }
    function collectLcaConstants() {
      const constants = {};
      const errors = [];
      document.querySelectorAll("#lcaConstantEditor [data-lca-param]").forEach(input => {
        const key = input.dataset.lcaParam;
        const value = Number(input.value);
        if (!Number.isFinite(value)) {
          errors.push(`${key} must be numeric.`);
          return;
        }
        constants[key] = value;
      });
      if (errors.length) throw new Error(errors.join(" "));
      return constants;
    }
    async function testOpenLca() {
      const button = document.getElementById("invTestOpenLca");
      const note = document.getElementById("lcaLccNote");
      const mode = document.getElementById("invEnvironmentalMode").value;
      button.disabled = true;
      button.textContent = "Testing...";
      const params = new URLSearchParams({
        mode,
        port: document.getElementById("invIpcPort").value || "8080",
        product_system: document.getElementById("invProductSystem").value || "",
        impact_method: document.getElementById("invImpactMethod").value || ""
      });
      try {
        const res = await fetch(`/api/openlca-status?${params.toString()}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "OpenLCA status check failed");
        const found = mode === "openlca_ipc" ? (data.product_system_found && data.impact_method_found) : Boolean(data.available);
        if (mode === "proxy") {
          note.textContent = "Offline proxy EBI/LCC evaluator is available. This does not call openLCA.";
        } else if (mode === "openlca_ipc") {
          note.textContent = data.available
            ? `${data.message} Port ${data.port}. Product system: ${data.product_system_found ? "found" : "not found"}. LCIA method: ${data.impact_method_found ? "found" : "not found"}.`
            : `OpenLCA IPC is not reachable on port ${data.port}: ${data.message}`;
        } else {
          note.textContent = data.available
            ? `${data.message || "Cloud evaluator responded."} ${data.url ? "Endpoint configured." : ""}`
            : `Cloud evaluator is not reachable: ${data.message || "No endpoint configured."}`;
        }
        note.style.color = found ? "var(--green)" : "var(--orange)";
      } catch (err) {
        note.textContent = String(err.message || err);
        note.style.color = "var(--red)";
      } finally {
        button.disabled = false;
        syncLcaEvaluatorMode();
      }
    }
    function collectInverseTarget() {
      const mode = document.getElementById("invTargetMode").value;
      const payload = {target_mode: mode};
      if (mode === "target_value") {
        const target = Number(document.getElementById("invTargetValue").value);
        const tolerance = Number(document.getElementById("invTolerance").value || 0);
        if (!Number.isFinite(target)) throw new Error("Target removal needs a numeric value.");
        if (!Number.isFinite(tolerance) || tolerance < 0) throw new Error("Tolerance must be a non-negative number.");
        payload.target_value = target;
        payload.tolerance = tolerance;
        payload.tolerance_mode = document.getElementById("invToleranceMode").value;
      } else {
        const low = Number(document.getElementById("invTargetMin").value);
        const high = Number(document.getElementById("invTargetMax").value);
        if (!Number.isFinite(low) || !Number.isFinite(high)) throw new Error("Target range needs numeric min and max values.");
        if (low > high) throw new Error("Target range min cannot be greater than max.");
        payload.target_min = low;
        payload.target_max = high;
      }
      return payload;
    }
    function renderDataset(name) {
      activeDataset = name;
      document.querySelectorAll("[data-dataset]").forEach(b => b.classList.toggle("active", b.dataset.dataset === name));
      const ds = assets.datasets[name];
      const defaults = datasetDefaults(name);
      document.getElementById("formTitle").textContent = ds.title;
      document.getElementById("formNote").textContent = name === "dataset1"
        ? "Biochar properties, adsorption conditions, and PFAS selection."
        : "Known PFAS/resin screening with operating conditions.";
      const parts = [];
      if (name === "dataset1") {
        parts.push('<div class="section-title">PFAS selection</div><div class="grid">');
        parts.push(selectField("PFAS", ds.pfas_options || [], defaults.PFAS));
        parts.push('</div><div class="section-title">Biochar properties</div><div class="grid">');
        ["Pyrolysis temperature","Pyrolysis time","Heating rated","C","Ash","H/C","O/C","(O+N)/C","Surface area","Average pore size","Pore volume"].forEach(key => parts.push(inputField(key, defaults[key], ds)));
        parts.push('</div><div class="section-title">Adsorption conditions</div><div class="grid">');
        ["Solution pH","Adsorption time","Adsorption temperature","S/L","Initial concentration","CaCl2","NaCl","HA"].forEach(key => parts.push(inputField(key, defaults[key], ds)));
        parts.push("</div>");
      } else {
        parts.push('<div class="section-title">PFAS and resin selection</div><div class="grid">');
        ds.categorical_fields.forEach(key => parts.push(selectField(key, ds.options[key] || [], defaults[key])));
        parts.push('</div><div class="section-title">Operating conditions</div><div class="grid">');
        ds.numeric_fields.forEach(key => parts.push(inputField(key, defaults[key], ds)));
        parts.push("</div>");
      }
      form.innerHTML = parts.join("");
      resetToDefaults();
      setEmptyState();
    }
    function resetToDefaults() {
      const defaults = datasetDefaults(activeDataset);
      Object.entries(defaults).forEach(([key, value]) => {
        const el = form.querySelector(`[data-key="${CSS.escape(key)}"]`);
        if (el) el.value = value ?? "";
      });
      lastValues = collectValues();
    }
    function collectValues() {
      const values = {};
      form.querySelectorAll("[data-key]").forEach(el => {
        const key = el.dataset.key;
        values[key] = el.type === "number" ? (el.value === "" ? null : Number(el.value)) : el.value;
      });
      return values;
    }
    function setEmptyState() {
      errorBox.textContent = "";
      document.getElementById("prediction").textContent = "--";
      document.getElementById("gauge").style.background = "conic-gradient(var(--teal) 0deg, var(--teal) 0deg, #e8edf3 0deg 360deg)";
      document.getElementById("resultTitle").textContent = "Run a prediction";
      document.getElementById("resultNote").textContent = "Charts update after prediction.";
      document.getElementById("contextChips").innerHTML = "";
      document.getElementById("predictionDownloads").innerHTML = "";
      document.getElementById("scenarioChart").innerHTML = '<div class="empty">No prediction yet.</div>';
      document.getElementById("sensitivityChart").innerHTML = '<div class="empty">Run a prediction to see sensitivity.</div>';
      document.getElementById("rangeChart").innerHTML = '<div class="empty">Run a prediction to check input ranges.</div>';
    }
    function drawGauge(value) {
      const v = Math.max(0, Math.min(100, Number(value) || 0));
      const deg = v * 3.6;
      const color = v >= 80 ? "var(--green)" : v >= 50 ? "var(--teal)" : v >= 25 ? "var(--orange)" : "var(--red)";
      document.getElementById("gauge").style.background = `conic-gradient(${color} 0deg, ${color} ${deg}deg, #e8edf3 ${deg}deg 360deg)`;
      document.getElementById("prediction").textContent = fmt(v, 1);
    }
    function drawScenario(data) {
      const max = 100;
      const rows = [
        ["Default baseline", data.baseline_prediction, "baseline"],
        ["Current input", data.prediction, ""]
      ];
      document.getElementById("scenarioChart").innerHTML = rows.map(([label, value, cls]) => {
        const width = Math.max(0, Math.min(100, value));
        return `<div class="scenario-bars"><div class="scenario-label">${label}</div><div class="track"><div class="scenario-fill ${cls}" style="width:${width / max * 100}%"></div></div><div class="bar-value">${fmt(value, 1)}%</div></div>`;
      }).join("");
    }
    function drawSensitivity(items) {
      const root = document.getElementById("sensitivityChart");
      if (!items || !items.length) {
        root.innerHTML = '<div class="empty">No sensitivity data available for this input.</div>';
        return;
      }
      const maxSwing = Math.max(...items.map(x => x.swing), 1);
      root.innerHTML = items.map(item => {
        const width = Math.max(4, item.swing / maxSwing * 100);
        return `<div class="bar-row"><div class="bar-label" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</div><div class="track"><div class="bar" style="width:${width}%"></div></div><div class="bar-value">${fmt(item.swing, 1)} pts</div></div>`;
      }).join("");
    }
    function drawRanges(items) {
      const root = document.getElementById("rangeChart");
      if (!items || !items.length) {
        root.innerHTML = '<div class="empty">No range metadata available.</div>';
        return;
      }
      root.innerHTML = items.map(item => {
        const statusClass = item.status === "inside" || item.status === "known" ? "" : item.status === "near_edge" ? "warn" : "bad";
        const statusText = item.status === "inside" ? "In range" : item.status === "known" ? "Known" : item.status === "near_edge" ? "Near edge" : item.status === "unknown_category" ? "New value" : "Out of range";
        const position = Math.max(0, Math.min(100, item.position ?? 0));
        const bar = item.kind === "numeric"
          ? `<div class="track"><span class="range-dot ${statusClass}" style="left:${position}%"></span></div>`
          : "";
        return `<div class="status-item"><div><div class="status-name">${escapeHtml(item.label)}</div>${bar}</div><div class="status-pill ${statusClass}">${statusText}</div></div>`;
      }).join("");
    }
    function drawContext(values, data) {
      const chips = [];
      if (activeDataset === "dataset1") {
        chips.push(["PFAS", values.PFAS || "--"]);
        chips.push(["Use case", "Biochar adsorption"]);
        chips.push(["Model", data.model_name || "Model"]);
      } else {
        chips.push(["PFAS", values.PFAS || "--"]);
        chips.push(["Resin", values.Resin || "--"]);
        chips.push(["Water", values.Solution || "--"]);
      }
      document.getElementById("contextChips").innerHTML = chips.map(([k, v]) => `<span class="chip">${escapeHtml(k)}: ${escapeHtml(v)}</span>`).join("");
    }
    async function predict() {
      const button = document.getElementById("predictButton");
      button.disabled = true;
      errorBox.textContent = "";
      const values = collectValues();
      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({dataset: activeDataset, values})
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Prediction failed");
        drawGauge(data.prediction);
        drawScenario(data);
        drawSensitivity(data.sensitivity);
        drawRanges(data.range_status);
        drawContext(values, data);
        const delta = data.prediction - data.baseline_prediction;
        document.getElementById("resultTitle").textContent = `${fmt(data.prediction, 1)}% estimated removal`;
        document.getElementById("resultNote").textContent = `Compared with the default baseline: ${delta >= 0 ? "+" : ""}${fmt(delta, 1)} percentage points.`;
        document.getElementById("predictionDownloads").innerHTML = `
          <a href="/api/prediction-xlsx?dataset=${encodeURIComponent(data.dataset)}" target="_blank">Download prediction Excel</a>
          <a href="/api/prediction-report?dataset=${encodeURIComponent(data.dataset)}" target="_blank">Download prediction report</a>
        `;
        lastValues = values;
      } catch (err) {
        errorBox.textContent = String(err.message || err);
      } finally {
        button.disabled = false;
      }
    }
    function shortValue(value) {
      if (value === null || value === undefined) return "";
      if (typeof value === "number") return fmt(value, Math.abs(value) >= 100 ? 1 : 3);
      return String(value);
    }
    function renderOptimizationTable(rows) {
      if (!rows || !rows.length) {
        return '<div class="empty" style="padding:14px;">No candidates returned.</div>';
      }
      const preferred = ["rank","prediction","PFAS","Solution","Resin","Polymer_matrix ","Functional group","resin_dosage_mg_L","initial_pfas_concentration_ug_L","contact_time_h","stirring_rate_rpm","pH","temperature_C","CDOC_mg_L","Initial concentration","Adsorption time","S/L"];
      const scalarKey = k => rows.some(row => row[k] !== null && row[k] !== undefined && typeof row[k] !== "object");
      const keys = preferred.filter(k => Object.prototype.hasOwnProperty.call(rows[0], k) && scalarKey(k)).concat(Object.keys(rows[0]).filter(k => !preferred.includes(k) && scalarKey(k)).slice(0, 8));
      const head = keys.map(k => `<th>${escapeHtml(labels[k] || k)}</th>`).join("");
      const body = rows.map(row => `<tr>${keys.map(k => `<td>${escapeHtml(shortValue(row[k]))}</td>`).join("")}</tr>`).join("");
      return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    }
    function renderOptimizationBars(rows, direction) {
      if (!rows || !rows.length) return '<div class="empty">No candidates returned.</div>';
      return rows.map((row, i) => {
        const value = Math.max(0, Math.min(100, Number(row.prediction) || 0));
        const cls = direction === "min" ? "minimize" : "";
        return `<div class="mini-bar-row"><div>Rank ${i + 1}</div><div class="track"><div class="mini-fill ${cls}" style="width:${value}%"></div></div><div class="bar-value">${fmt(value, 1)}%</div></div>`;
      }).join("");
    }
    function renderOptConvergence(history) {
      const data = (history || []).filter(row => Number.isFinite(Number(row.best_prediction)));
      if (!data.length) return '<div class="empty">No convergence history yet.</div>';
      const w = 640, h = 230, l = 46, r = 18, t = 18, b = 38;
      const xs = data.map(row => Number(row.iteration));
      const ys = data.map(row => Number(row.best_prediction));
      const minX = Math.min(...xs), maxX = Math.max(...xs, minX + 1);
      const minY = Math.min(...ys, 0), maxY = Math.max(...ys, 100);
      const x = v => svgPoint(v, minX, maxX, l, w-r);
      const y = v => svgPoint(v, minY, maxY, t, h-b, true);
      const path = data.map((row, i) => `${i ? "L" : "M"} ${x(Number(row.iteration))} ${y(Number(row.best_prediction))}`).join(" ");
      const dots = data.map(row => `<circle cx="${x(Number(row.iteration))}" cy="${y(Number(row.best_prediction))}" r="3.5" fill="#157f3b"><title>Iteration ${row.iteration}: ${fmt(row.best_prediction,2)}%</title></circle>`).join("");
      return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="PSO convergence"><line x1="${l}" y1="${h-b}" x2="${w-r}" y2="${h-b}" stroke="#d7dde7"/><line x1="${l}" y1="${t}" x2="${l}" y2="${h-b}" stroke="#d7dde7"/><path d="${path}" fill="none" stroke="#157f3b" stroke-width="3"/>${dots}<text x="${l}" y="12" font-size="11" fill="#667085">Best removal %</text><text x="${w-r}" y="${h-8}" font-size="11" text-anchor="end" fill="#667085">Iteration</text></svg>`;
    }
    function renderOptScatter(rows) {
      const data = (rows || []).filter(row => Number.isFinite(Number(row.prediction)));
      if (!data.length) return '<div class="empty">No candidates yet.</div>';
      const w = 640, h = 230, l = 46, r = 18, t = 18, b = 38;
      const yVals = data.map(row => Number(row.prediction));
      const minY = Math.min(...yVals, 0), maxY = Math.max(...yVals, 100);
      const x = i => svgPoint(i, 0, Math.max(data.length - 1, 1), l, w-r);
      const y = v => svgPoint(v, minY, maxY, t, h-b, true);
      const points = data.slice(0, 500).map((row, i) => `<circle cx="${x(i)}" cy="${y(Number(row.prediction))}" r="4" fill="#2f6fed" opacity="0.55"><title>${fmt(row.prediction,2)}% removal</title></circle>`).join("");
      return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Optimization candidate scatter"><line x1="${l}" y1="${h-b}" x2="${w-r}" y2="${h-b}" stroke="#d7dde7"/><line x1="${l}" y1="${t}" x2="${l}" y2="${h-b}" stroke="#d7dde7"/>${points}<text x="${l}" y="12" font-size="11" fill="#667085">Removal efficiency %</text><text x="${w-r}" y="${h-8}" font-size="11" text-anchor="end" fill="#667085">Candidate order</text></svg>`;
    }
    function renderOptParallel(rows) {
      return renderParallelCoordinates(rows || []);
    }
    function renderOptProgress(status) {
      const progress = Math.max(0, Math.min(1, Number(status?.progress) || 0));
      document.getElementById("optProgressFill").style.width = `${progress * 100}%`;
      document.getElementById("optProgressPct").textContent = `${Math.round(progress * 100)}%`;
      document.getElementById("optProgressText").textContent = status?.message || "Running optimization.";
      document.getElementById("optConvergence").innerHTML = renderOptConvergence(status?.history || []);
      document.getElementById("optScatter").innerHTML = renderOptScatter(status?.all_candidates || []);
      document.getElementById("optParallel").innerHTML = renderOptParallel(status?.all_candidates || []);
    }
    function renderInverseTable(rows) {
      if (!rows || !rows.length) {
        return '<div class="empty" style="padding:14px;">No candidates returned.</div>';
      }
      const preferred = ["rank","prediction","distance","lcc_total_usd_m3","ebi","pareto_front","target_error","meets_target","environmental_source","eol_mode","PFAS","Initial concentration","S/L","Adsorption time","Solution pH","Pyrolysis temperature","Surface area","CaCl2","NaCl","HA","Solution","Resin","Polymer_matrix ","Functional group","resin_dosage_mg_L","initial_pfas_concentration_ug_L","contact_time_h","stirring_rate_rpm","pH","temperature_C","CDOC_mg_L"];
      const scalarKey = k => rows.some(row => row[k] !== null && row[k] !== undefined && typeof row[k] !== "object");
      const keys = preferred.filter(k => Object.prototype.hasOwnProperty.call(rows[0], k) && scalarKey(k)).concat(Object.keys(rows[0]).filter(k => !preferred.includes(k) && scalarKey(k)).slice(0, 8));
      const head = keys.map(k => `<th>${escapeHtml(labels[k] || k)}</th>`).join("");
      const body = rows.map(row => `<tr>${keys.map(k => `<td>${escapeHtml(shortValue(row[k]))}</td>`).join("")}</tr>`).join("");
      return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    }
    function rankedCandidates(rows, top = 10) {
      return [...(rows || [])]
        .sort((a, b) => {
          const ap = a.pareto_front ? 0 : 1;
          const bp = b.pareto_front ? 0 : 1;
          return ap - bp
            || (Number(a.composite_objective) || 0) - (Number(b.composite_objective) || 0)
            || (Number(a.distance) || 0) - (Number(b.distance) || 0)
            || (Number(a.lcc_total_usd_m3) || 0) - (Number(b.lcc_total_usd_m3) || 0)
            || (Number(a.ebi) || 0) - (Number(b.ebi) || 0)
            || (Number(a.target_error) || 0) - (Number(b.target_error) || 0);
        })
        .slice(0, top)
        .map((row, index) => ({rank: index + 1, ...row}));
    }
    function sourceType(row) {
      const s = String(row?.source || row?.generation_strategy || "").toLowerCase();
      if (s.includes("sobol") || s.includes("initial")) return "Sobol";
      if (s.includes("botorch") || s.includes("qlog") || s.includes("qei")) return "BoTorch";
      return "Other";
    }
    function sourceColor(name) {
      if (name === "Sobol") return "#d97830";
      if (name === "BoTorch") return "#2f6fed";
      return "#667085";
    }
    function svgPoint(value, min, max, low, high, invert = false) {
      if (!Number.isFinite(value)) value = min;
      if (!Number.isFinite(min) || !Number.isFinite(max) || Math.abs(max - min) < 1e-12) return (low + high) / 2;
      const t = Math.max(0, Math.min(1, (value - min) / (max - min)));
      return invert ? high - t * (high - low) : low + t * (high - low);
    }
    function targetWindow(details) {
      const d = details || {};
      const mode = d.target_mode || d.targetMode || document.getElementById("invTargetMode")?.value || "target_value";
      if (mode === "target_range") {
        return {
          mode,
          min: Number(d.target_min ?? document.getElementById("invTargetMin")?.value),
          max: Number(d.target_max ?? document.getElementById("invTargetMax")?.value)
        };
      }
      const value = Number(d.target_value ?? document.getElementById("invTargetValue")?.value);
      const tol = Number(d.tolerance ?? document.getElementById("invTolerance")?.value ?? 0);
      const tolMode = d.tolerance_mode || document.getElementById("invToleranceMode")?.value || "absolute";
      const effectiveTol = tolMode === "percent" ? Math.abs(value) * tol / 100 : tol;
      return {mode, value, min: value - effectiveTol, max: value + effectiveTol};
    }
    function renderInverseHistory(history) {
      if (!history || !history.length) return '<div class="empty">No convergence history returned.</div>';
      const w = 640, h = 230, l = 54, r = 18, t = 18, b = 38;
      const xs = history.map((_, i) => i);
      const bestVals = history.map(item => Number(item.best_overall_distance)).filter(Number.isFinite);
      const batchVals = history.map(item => Number(item.batch_min_distance)).filter(Number.isFinite);
      const allVals = bestVals.concat(batchVals);
      const yMax = Math.max(...allVals, 1e-9);
      const yMin = Math.min(...allVals, 0);
      const maxX = Math.max(xs.length - 1, 1);
      const x = i => svgPoint(i, 0, maxX, l, w - r);
      const y = v => svgPoint(v, yMin, yMax, t, h - b, true);
      const bestPath = history.map((item, i) => `${i ? "L" : "M"} ${x(i)} ${y(Number(item.best_overall_distance) || 0)}`).join(" ");
      const batchPath = history.map((item, i) => `${i ? "L" : "M"} ${x(i)} ${y(Number(item.batch_min_distance) || 0)}`).join(" ");
      const dots = history.map((item, i) => `<circle cx="${x(i)}" cy="${y(Number(item.best_overall_distance) || 0)}" r="4" fill="#157f3b"><title>${escapeHtml(item.generation_strategy || "")}: ${fmt(item.best_overall_distance, 5)}</title></circle>`).join("");
      return `
        <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Distance evolution">
          <line x1="${l}" y1="${h-b}" x2="${w-r}" y2="${h-b}" stroke="#d7dde7"/>
          <line x1="${l}" y1="${t}" x2="${l}" y2="${h-b}" stroke="#d7dde7"/>
          <line x1="${l}" y1="${y(0)}" x2="${w-r}" y2="${y(0)}" stroke="#9aa4b2" stroke-dasharray="4 4"/>
          <path d="${batchPath}" fill="none" stroke="#2f6fed" stroke-width="2" opacity="0.55"/>
          <path d="${bestPath}" fill="none" stroke="#157f3b" stroke-width="3"/>
          ${dots}
          <text x="${l}" y="12" font-size="11" fill="#667085">Distance</text>
          <text x="${w-r}" y="${h-8}" font-size="11" text-anchor="end" fill="#667085">Iteration</text>
          <text x="${l}" y="${h-8}" font-size="11" fill="#667085">${escapeHtml(history[0]?.generation_strategy || "init")}</text>
          <text x="${w-r}" y="12" font-size="11" text-anchor="end" fill="#667085">Best ${fmt(bestVals[bestVals.length - 1], 5)}</text>
        </svg>
        <div class="legend"><span><i class="dot best"></i>Best overall distance</span><span><i class="dot bo"></i>Batch minimum distance</span></div>
      `;
    }
    function renderInverseScatter(rows, details) {
      const data = (rows || []).filter(r => Number.isFinite(Number(r.prediction)) && Number.isFinite(Number(r.distance)));
      if (!data.length) return '<div class="empty">No candidate scatter data yet.</div>';
      const w = 640, h = 230, l = 46, r = 18, t = 18, b = 38;
      const distances = data.map(r => Number(r.distance));
      const yMax = Math.max(...distances, 1e-9);
      const yMin = Math.min(...distances, 0);
      const x = v => svgPoint(v, 0, 100, l, w-r);
      const y = v => svgPoint(v, yMin, yMax, t, h-b, true);
      const target = targetWindow(details);
      let targetSvg = "";
      if (Number.isFinite(target.min) && Number.isFinite(target.max)) {
        const x1 = x(Math.max(0, target.min));
        const x2 = x(Math.min(100, target.max));
        targetSvg += `<rect x="${Math.min(x1,x2)}" y="${t}" width="${Math.abs(x2-x1)}" height="${h-b-t}" fill="#16a39a" opacity="0.09"/>`;
      }
      if (Number.isFinite(target.value)) {
        targetSvg += `<line x1="${x(target.value)}" y1="${t}" x2="${x(target.value)}" y2="${h-b}" stroke="#157f3b" stroke-dasharray="4 4"/>`;
      }
      const best = [...data].sort((a,b) => Number(a.distance) - Number(b.distance))[0];
      const points = data.map(row => {
        const src = sourceType(row);
        const isBest = row === best;
        return `<circle cx="${x(Number(row.prediction))}" cy="${y(Number(row.distance))}" r="${isBest ? 6 : 4}" fill="${isBest ? "#157f3b" : sourceColor(src)}" opacity="${isBest ? 0.95 : 0.72}"><title>${src}: pred ${fmt(row.prediction,2)}%, distance ${fmt(row.distance,5)}</title></circle>`;
      }).join("");
      return `
        <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Sobol and BoTorch candidate scatter">
          ${targetSvg}
          <line x1="${l}" y1="${h-b}" x2="${w-r}" y2="${h-b}" stroke="#d7dde7"/>
          <line x1="${l}" y1="${t}" x2="${l}" y2="${h-b}" stroke="#d7dde7"/>
          ${points}
          <text x="${l}" y="12" font-size="11" fill="#667085">Distance</text>
          <text x="${w-r}" y="${h-8}" font-size="11" text-anchor="end" fill="#667085">Predicted removal (%)</text>
          <text x="${l}" y="${h-8}" font-size="11" fill="#667085">0</text>
          <text x="${w-r}" y="${h-8}" font-size="11" fill="#667085">100</text>
        </svg>
        <div class="legend"><span><i class="dot sobol"></i>Sobol</span><span><i class="dot bo"></i>BoTorch</span><span><i class="dot best"></i>Best</span></div>
      `;
    }
    function renderPredictionFit(rows, details) {
      const data = [...(rows || [])].filter(r => Number.isFinite(Number(r.prediction))).sort((a,b) => Number(a.batch || 0) - Number(b.batch || 0));
      if (!data.length) return '<div class="empty">No prediction exploration data yet.</div>';
      const w = 640, h = 230, l = 44, r = 18, t = 18, b = 38;
      const x = i => svgPoint(i, 0, Math.max(data.length - 1, 1), l, w-r);
      const y = v => svgPoint(v, 0, 100, t, h-b, true);
      const target = targetWindow(details);
      let targetSvg = "";
      if (Number.isFinite(target.min) && Number.isFinite(target.max)) {
        const y1 = y(Math.max(0, target.min));
        const y2 = y(Math.min(100, target.max));
        targetSvg += `<rect x="${l}" y="${Math.min(y1,y2)}" width="${w-r-l}" height="${Math.abs(y2-y1)}" fill="#16a39a" opacity="0.1"/>`;
      }
      if (Number.isFinite(target.value)) {
        targetSvg += `<line x1="${l}" y1="${y(target.value)}" x2="${w-r}" y2="${y(target.value)}" stroke="#157f3b" stroke-dasharray="4 4"/>`;
      }
      const points = data.map((row, i) => {
        const src = sourceType(row);
        return `<circle cx="${x(i)}" cy="${y(Number(row.prediction))}" r="4" fill="${sourceColor(src)}" opacity="0.72"><title>${src}: prediction ${fmt(row.prediction,2)}%</title></circle>`;
      }).join("");
      return `
        <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Prediction target fit">
          ${targetSvg}
          <line x1="${l}" y1="${h-b}" x2="${w-r}" y2="${h-b}" stroke="#d7dde7"/>
          <line x1="${l}" y1="${t}" x2="${l}" y2="${h-b}" stroke="#d7dde7"/>
          ${points}
          <text x="${l}" y="12" font-size="11" fill="#667085">Removal %</text>
          <text x="${w-r}" y="${h-8}" font-size="11" text-anchor="end" fill="#667085">Candidate order</text>
          <text x="${l}" y="${h-8}" font-size="11" fill="#667085">0</text>
          <text x="${l}" y="${t+4}" font-size="11" fill="#667085">100</text>
        </svg>
        <div class="legend"><span><i class="dot sobol"></i>Sobol</span><span><i class="dot bo"></i>BoTorch</span><span><i class="dot best"></i>Target band/value</span></div>
      `;
    }
    function renderSourceChart(rows) {
      const data = rows || [];
      if (!data.length) return '<div class="empty">No candidate sources yet.</div>';
      const groups = {};
      data.forEach(row => {
        const src = sourceType(row);
        if (!groups[src]) groups[src] = {count: 0, best: Infinity};
        groups[src].count += 1;
        groups[src].best = Math.min(groups[src].best, Number(row.distance) || Infinity);
      });
      const entries = Object.entries(groups);
      const maxCount = Math.max(...entries.map(([, v]) => v.count), 1);
      return entries.map(([src, item]) => {
        const width = Math.max(5, item.count / maxCount * 100);
        return `<div class="bar-row"><div class="bar-label"><span class="dot ${src === "Sobol" ? "sobol" : src === "BoTorch" ? "bo" : ""}"></span> ${escapeHtml(src)}</div><div class="track"><div class="bar" style="width:${width}%; background:${sourceColor(src)}"></div></div><div class="bar-value">${item.count} | ${fmt(item.best, 4)}</div></div>`;
      }).join("");
    }
    function paretoProjection(row, bounds) {
      const dx = (Number(row.distance) - bounds.distance.min) / Math.max(bounds.distance.max - bounds.distance.min, 1e-12);
      const cx = (Number(row.lcc_total_usd_m3) - bounds.lcc.min) / Math.max(bounds.lcc.max - bounds.lcc.min, 1e-12);
      const ez = (Number(row.ebi) - bounds.ebi.min) / Math.max(bounds.ebi.max - bounds.ebi.min, 1e-12);
      const x = 105 + dx * 285 + cx * 135;
      const y = 260 - ez * 145 - cx * 70;
      return {x, y, dx, cx, ez};
    }
    function renderPareto3D(rows) {
      const data = (rows || []).filter(row =>
        Number.isFinite(Number(row.distance)) &&
        Number.isFinite(Number(row.lcc_total_usd_m3)) &&
        Number.isFinite(Number(row.ebi))
      );
      if (!data.length) return '<div class="empty">No LCC/EBI candidate data yet.</div>';
      const values = {
        distance: data.map(row => Number(row.distance)),
        lcc: data.map(row => Number(row.lcc_total_usd_m3)),
        ebi: data.map(row => Number(row.ebi))
      };
      const bounds = {
        distance: {min: Math.min(...values.distance), max: Math.max(...values.distance)},
        lcc: {min: Math.min(...values.lcc), max: Math.max(...values.lcc)},
        ebi: {min: Math.min(...values.ebi), max: Math.max(...values.ebi)}
      };
      const grid = `
        <line x1="105" y1="260" x2="390" y2="260" stroke="#d7dde7"/>
        <line x1="105" y1="260" x2="240" y2="190" stroke="#d7dde7"/>
        <line x1="105" y1="260" x2="105" y2="115" stroke="#d7dde7"/>
        <line x1="390" y1="260" x2="525" y2="190" stroke="#eef1f5"/>
        <line x1="240" y1="190" x2="525" y2="190" stroke="#eef1f5"/>
        <line x1="105" y1="115" x2="240" y2="45" stroke="#eef1f5"/>
        <text x="392" y="284" font-size="11" fill="#667085">Distance min to max</text>
        <text x="235" y="183" font-size="11" fill="#667085">LCC min to max</text>
        <text x="70" y="112" font-size="11" fill="#667085">EBI min to max</text>
      `;
      const points = data.map(row => {
        const p = paretoProjection(row, bounds);
        const front = Boolean(row.pareto_front);
        const color = front ? "#157f3b" : sourceColor(sourceType(row));
        const radius = front ? 6 : 4;
        const opacity = front ? 0.95 : 0.58;
        return `<circle class="pareto-point" cx="${p.x}" cy="${p.y}" r="${radius}" fill="${color}" opacity="${opacity}">
          <title>${front ? "Pareto" : sourceType(row)} | distance ${fmt(row.distance, 5)} | LCC ${fmt(row.lcc_total_usd_m3, 4)} USD/m3 | EBI ${fmt(row.ebi, 8)}</title>
        </circle>`;
      }).join("");
      return `
        <svg viewBox="0 0 640 320" role="img" aria-label="3-objective Pareto front">
          ${grid}
          ${points}
          <text x="105" y="300" font-size="11" fill="#667085">Best corner: low distance, low LCC, low EBI</text>
        </svg>
        <div class="legend"><span><i class="dot best"></i>Pareto front</span><span><i class="dot bo"></i>BoTorch</span><span><i class="dot sobol"></i>Sobol</span></div>
      `;
    }
    function renderLcaSummary(details, rows) {
      if (!details?.lca_lcc_enabled) return '<div class="empty">LCA/LCC objectives were not enabled for this run.</div>';
      const sourceRows = (rows || []).filter(row => row.environmental_source);
      const sources = [...new Set(sourceRows.map(row => row.environmental_source))].join(", ") || "--";
      const front = Number(details.pareto_front_size || 0);
      const items = [
        ["Best target distance", details.best_distance, ""],
        ["Best LCC", details.best_lcc_total_usd_m3, "USD/m3"],
        ["Best normalized EBI", details.best_ebi, ""],
        ["Pareto candidates", front, ""],
        ["Environmental evaluator", sources, ""]
      ];
      return items.map(([label, value, unit]) => `
        <div class="status-item">
          <div class="status-name">${escapeHtml(label)}</div>
          <div class="status-pill">${typeof value === "number" ? fmt(value, label.includes("EBI") ? 8 : 4) : escapeHtml(value)} ${unit}</div>
        </div>
      `).join("");
    }
    function bestCandidateRow(rows) {
      const ranked = rankedCandidates(rows || [], 1);
      return ranked.length ? ranked[0] : null;
    }
    function renderImpactBreakdown(rows) {
      const row = bestCandidateRow((rows || []).filter(r => r.normalized_impacts && typeof r.normalized_impacts === "object"));
      if (!row || !row.normalized_impacts) return '<div class="empty">No normalized impact breakdown returned yet.</div>';
      const entries = Object.entries(row.normalized_impacts)
        .map(([name, value]) => [name, Number(value)])
        .filter(([, value]) => Number.isFinite(value))
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 10);
      if (!entries.length) return '<div class="empty">No normalized impact categories matched the ReCiPe normalization table.</div>';
      const max = Math.max(...entries.map(([, value]) => Math.abs(value)), 1e-12);
      return entries.map(([name, value]) => {
        const width = Math.max(3, Math.abs(value) / max * 100);
        return `<div class="bar-row"><div class="bar-label" title="${escapeHtml(name)}">${escapeHtml(name)}</div><div class="track"><div class="bar" style="width:${width}%; background:#0f6d7a"></div></div><div class="bar-value">${fmt(value, 8)}</div></div>`;
      }).join("");
    }
    function renderObjectiveScatter(rows) {
      const data = (rows || []).filter(r => Number.isFinite(Number(r.prediction)) && Number.isFinite(Number(r.lcc_total_usd_m3)));
      if (!data.length) return '<div class="empty">No removal/cost objective data yet.</div>';
      const w = 640, h = 250, l = 52, r = 20, t = 18, b = 42;
      const xVals = data.map(r => Number(r.prediction));
      const yVals = data.map(r => Number(r.lcc_total_usd_m3));
      const ebiVals = data.map(r => Number(r.ebi)).filter(Number.isFinite);
      const minX = Math.min(...xVals, 0), maxX = Math.max(...xVals, 100);
      const minY = Math.min(...yVals), maxY = Math.max(...yVals);
      const minE = ebiVals.length ? Math.min(...ebiVals) : 0;
      const maxE = ebiVals.length ? Math.max(...ebiVals) : 1;
      const x = v => svgPoint(v, minX, maxX, l, w-r);
      const y = v => svgPoint(v, minY, maxY, t, h-b, true);
      const points = data.map(row => {
        const e = Number(row.ebi);
        const eNorm = Number.isFinite(e) ? (e - minE) / Math.max(maxE - minE, 1e-12) : 0.5;
        const color = row.pareto_front ? "#157f3b" : `rgb(${Math.round(47 + eNorm * 150)}, ${Math.round(111 - eNorm * 45)}, ${Math.round(237 - eNorm * 120)})`;
        return `<circle cx="${x(Number(row.prediction))}" cy="${y(Number(row.lcc_total_usd_m3))}" r="${row.pareto_front ? 6 : 4}" fill="${color}" opacity="${row.pareto_front ? 0.95 : 0.62}"><title>Removal ${fmt(row.prediction,2)}%, LCC ${fmt(row.lcc_total_usd_m3,4)} USD/m3, EBI ${fmt(row.ebi,8)}</title></circle>`;
      }).join("");
      return `
        <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Removal cost objective scatter">
          <line x1="${l}" y1="${h-b}" x2="${w-r}" y2="${h-b}" stroke="#d7dde7"/>
          <line x1="${l}" y1="${t}" x2="${l}" y2="${h-b}" stroke="#d7dde7"/>
          ${points}
          <text x="${w-r}" y="${h-9}" font-size="11" text-anchor="end" fill="#667085">Predicted removal efficiency (%)</text>
          <text x="${l}" y="12" font-size="11" fill="#667085">LCC (USD/m3)</text>
          <text x="${l}" y="${h-9}" font-size="11" fill="#667085">${fmt(minX,1)}%</text>
          <text x="${w-r}" y="${h-9}" font-size="11" fill="#667085">${fmt(maxX,1)}%</text>
        </svg>
        <div class="legend"><span><i class="dot best"></i>Pareto</span><span><i class="dot bo"></i>Lower to higher EBI color scale</span></div>
      `;
    }
    function renderParallelCoordinates(rows) {
      const data = (rows || []).filter(r => Number.isFinite(Number(r.prediction))).slice(0, 80);
      if (!data.length) return '<div class="empty">No candidate data available for parallel coordinates.</div>';
      const axes = [
        ["prediction", "Removal %"],
        ["distance", "Distance"],
        ["lcc_total_usd_m3", "LCC"],
        ["ebi", "EBI"],
        ["S/L", "S/L"],
        ["Adsorption time", "Time"],
        ["Initial concentration", "C0"]
      ].filter(([key]) => data.some(row => Number.isFinite(Number(row[key]))));
      if (axes.length < 2) return '<div class="empty">Not enough numeric fields for parallel coordinates.</div>';
      const w = 900, h = 300, t = 28, b = 42, l = 42, r = 42;
      const axisX = i => l + i * ((w - l - r) / Math.max(axes.length - 1, 1));
      const bounds = {};
      axes.forEach(([key]) => {
        const values = data.map(row => Number(row[key])).filter(Number.isFinite);
        bounds[key] = {min: Math.min(...values), max: Math.max(...values)};
      });
      const y = (key, value) => svgPoint(Number(value), bounds[key].min, bounds[key].max, t, h-b, true);
      const lines = data.map(row => {
        const d = axes.map(([key], i) => `${i ? "L" : "M"} ${axisX(i)} ${y(key, row[key])}`).join(" ");
        const color = row.pareto_front ? "#157f3b" : sourceColor(sourceType(row));
        const opacity = row.pareto_front ? 0.72 : 0.22;
        return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${row.pareto_front ? 2.4 : 1.2}" opacity="${opacity}"/>`;
      }).join("");
      const axisSvg = axes.map(([key, label], i) => {
        const x = axisX(i);
        return `<g><line x1="${x}" y1="${t}" x2="${x}" y2="${h-b}" stroke="#d7dde7"/><text x="${x}" y="${h-18}" text-anchor="middle" font-size="11" fill="#344054">${escapeHtml(label)}</text><text x="${x}" y="${t-8}" text-anchor="middle" font-size="10" fill="#667085">${fmt(bounds[key].max, 3)}</text><text x="${x}" y="${h-b+14}" text-anchor="middle" font-size="10" fill="#667085">${fmt(bounds[key].min, 3)}</text></g>`;
      }).join("");
      return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Parallel coordinates">${axisSvg}${lines}</svg><div class="legend"><span><i class="dot best"></i>Pareto candidates</span><span><i class="dot bo"></i>BoTorch/other candidates</span></div>`;
    }
    function renderInverseProgress(status) {
      const progress = Math.max(0, Math.min(1, Number(status?.progress) || 0));
      document.getElementById("invProgressFill").style.width = `${progress * 100}%`;
      document.getElementById("invProgressPct").textContent = `${Math.round(progress * 100)}%`;
      document.getElementById("invProgressText").textContent = status?.message || "Running inverse design.";
      const history = status?.history || [];
      if (!history.length) {
        document.getElementById("invStageList").innerHTML = '<div class="empty">Preparing first stage...</div>';
        return;
      }
      document.getElementById("invStageList").innerHTML = history.slice(-6).map(item => {
        const label = Number(item.batch) === 0 ? "Init" : `Batch ${item.batch}`;
        return `<div class="stage-item"><strong>${label}</strong><span>${escapeHtml(item.generation_strategy || item.phase || "")}</span><span>${fmt(item.best_overall_prediction, 1)}% | D ${fmt(item.best_overall_distance, 4)}</span></div>`;
      }).join("");
    }
    function updateInverseVisuals(status, finalResult = null) {
      const details = finalResult?.details || status?.result?.details || status || {};
      const allRows = details.all_candidates || status?.all_candidates || [];
      const history = details.history || status?.history || [];
      renderInverseProgress(status || {progress: 0, message: "Waiting for a run.", history: []});
      document.getElementById("invHistory").innerHTML = renderInverseHistory(history);
      document.getElementById("invScatter").innerHTML = renderInverseScatter(allRows, details);
      document.getElementById("invPredictionChart").innerHTML = renderPredictionFit(allRows, details);
      document.getElementById("invSourceChart").innerHTML = renderSourceChart(allRows);
      document.getElementById("invPareto3D").innerHTML = renderPareto3D(allRows);
      document.getElementById("invLcaSummary").innerHTML = renderLcaSummary(details, allRows);
      document.getElementById("invImpactBreakdown").innerHTML = renderImpactBreakdown(allRows);
      document.getElementById("invObjectiveScatter").innerHTML = renderObjectiveScatter(allRows);
      document.getElementById("invParallel").innerHTML = renderParallelCoordinates(allRows);
      if (!finalResult && allRows.length) {
        document.getElementById("invTable").innerHTML = renderInverseTable(rankedCandidates(allRows, 10));
      }
    }
    async function runInverseDesign() {
      const button = document.getElementById("invRunButton");
      let payload = {};
      try {
        const selectedDataset = document.getElementById("invDataset").value;
        const includeLcaLcc = selectedDataset === "dataset1" && document.getElementById("invObjectiveMode").value === "target_lca_lcc_pareto";
        payload = {
          dataset: selectedDataset,
          ...collectInverseTarget(),
          initialization_strategy: document.getElementById("invInitStrategy").value,
          initialization_trials: Number(document.getElementById("invInitTrials").value || 0),
          execution_mode: document.getElementById("invExecutionMode").value,
          batch_iterations: Number(document.getElementById("invIterations").value || 20),
          batch_size: Number(document.getElementById("invBatchSize").value || 1),
          raw_samples: Number(document.getElementById("invParticles").value || 64),
          top_k: 10,
          ranges: collectInverseRanges(),
          categories: collectInverseCategories(),
          include_lca_lcc: includeLcaLcc,
          environmental_mode: document.getElementById("invEnvironmentalMode").value,
          ipc_port: Number(document.getElementById("invIpcPort").value || 8080),
          eol_mode: document.getElementById("invEolMode").value,
          country: document.getElementById("invCountry").value,
          product_system: document.getElementById("invProductSystem").value,
          impact_method: document.getElementById("invImpactMethod").value,
          fallback_to_proxy: document.getElementById("invFallbackProxy").checked,
          lca_constants: includeLcaLcc ? collectLcaConstants() : {}
        };
        document.getElementById("invError").textContent = "";
      } catch (err) {
        document.getElementById("invError").textContent = String(err.message || err);
        return;
      }
      button.disabled = true;
      button.textContent = "Running...";
      document.getElementById("invTable").innerHTML = '<div class="empty" style="padding:14px;">Running inverse design.</div>';
      document.getElementById("invHistory").innerHTML = '<div class="empty">Waiting for first progress update...</div>';
      document.getElementById("invScatter").innerHTML = '<div class="empty">Waiting for candidates...</div>';
      document.getElementById("invPredictionChart").innerHTML = '<div class="empty">Waiting for candidates...</div>';
      document.getElementById("invSourceChart").innerHTML = '<div class="empty">Waiting for candidates...</div>';
      document.getElementById("invPareto3D").innerHTML = payload.include_lca_lcc ? '<div class="empty">Waiting for Pareto candidates...</div>' : '<div class="empty">LCA/LCC objectives are disabled for this run.</div>';
      document.getElementById("invLcaSummary").innerHTML = payload.include_lca_lcc ? '<div class="empty">Waiting for LCA/LCC objective values...</div>' : '<div class="empty">LCA/LCC objectives are disabled for this run.</div>';
      document.getElementById("invImpactBreakdown").innerHTML = payload.include_lca_lcc ? '<div class="empty">Waiting for normalized impact breakdown...</div>' : '<div class="empty">LCA/LCC objectives are disabled for this run.</div>';
      document.getElementById("invObjectiveScatter").innerHTML = '<div class="empty">Waiting for objective scatter...</div>';
      document.getElementById("invParallel").innerHTML = '<div class="empty">Waiting for candidate trajectories...</div>';
      renderInverseProgress({progress: 0.02, message: "Submitting inverse-design job.", history: []});
      document.getElementById("invLinks").textContent = `BoTorch inverse design. Objectives: ${payload.include_lca_lcc ? "target distance + LCC + EBI" : "target distance"}, initialization: ${payload.initialization_strategy}, execution: ${payload.execution_mode}, iterations: ${payload.batch_iterations}, raw samples: ${payload.raw_samples}.`;
      try {
        const startRes = await fetch("/api/inverse-design-job", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const started = await startRes.json();
        if (!startRes.ok) throw new Error(started.error || "Could not start inverse design");
        let data = null;
        while (true) {
          await new Promise(resolve => setTimeout(resolve, 850));
          const statusRes = await fetch(`/api/inverse-status?job_id=${encodeURIComponent(started.job_id)}`);
          data = await statusRes.json();
          if (!statusRes.ok) throw new Error(data.error || "Could not read inverse-design status");
          updateInverseVisuals(data);
          if (data.status === "complete") {
            data = data.result;
            break;
          }
          if (data.status === "error") {
            throw new Error(data.error || data.message || "Inverse design failed");
          }
        }
        document.getElementById("invTable").innerHTML = renderInverseTable(data.candidates);
        updateInverseVisuals({status: "complete", progress: 1, message: "Inverse design complete.", history: data.details.history, all_candidates: data.details.all_candidates || []}, data);
        const warnings = [...(data.details.range_warnings || []), ...(data.details.category_warnings || [])].map(w => `<span>${escapeHtml(w)}</span>`).join(" ");
        const best = data.details.best_prediction === null || data.details.best_prediction === undefined ? "--" : `${fmt(data.details.best_prediction, 2)}%`;
        const bestLcc = data.details.best_lcc_total_usd_m3 === null || data.details.best_lcc_total_usd_m3 === undefined ? "" : `<span>Best LCC: ${fmt(data.details.best_lcc_total_usd_m3, 4)} USD/m3. </span>`;
        const bestEbi = data.details.best_ebi === null || data.details.best_ebi === undefined ? "" : `<span>Best EBI: ${fmt(data.details.best_ebi, 8)}. </span>`;
        document.getElementById("invLinks").innerHTML = `
          <span>Best prediction: ${best}. </span>
          <span>Best distance: ${fmt(data.details.best_distance, 5)}. </span>
          ${bestLcc}
          ${bestEbi}
          ${data.details.lca_lcc_enabled ? `<span>Pareto front size: ${fmt(data.details.pareto_front_size, 0)}. </span>` : ""}
          <span>Strategy: ${escapeHtml(data.details.strategy || "BoTorch Bayesian optimization")}. </span>
          <span>Acquisition: ${escapeHtml(data.details.acquisition || "qLogExpectedImprovement")}. </span>
          <span>Evaluated unique candidates: ${data.details.evaluated_unique_candidates}. </span>
          <span>Range mode: ${escapeHtml(data.details.range_mode)}. </span>
          <span>Category mode: ${escapeHtml(data.details.category_mode || "all_available")}. </span>
          ${warnings ? `<span>${warnings}</span>` : ""}
          <a href="/api/inverse-csv?dataset=${encodeURIComponent(data.dataset)}" target="_blank">Download inverse candidates CSV</a>
          <a href="/api/inverse-xlsx?dataset=${encodeURIComponent(data.dataset)}" target="_blank">Download inverse Excel</a>
          <a href="/api/inverse-report?dataset=${encodeURIComponent(data.dataset)}" target="_blank">Download inverse report</a>
        `;
      } catch (err) {
        document.getElementById("invTable").innerHTML = `<div class="error" style="padding:14px;">${escapeHtml(err.message || err)}</div>`;
        document.getElementById("invHistory").innerHTML = '<div class="empty">Inverse design failed.</div>';
        renderInverseProgress({progress: 1, message: String(err.message || err), history: []});
      } finally {
        button.disabled = false;
        button.textContent = "Run Inverse Design";
      }
    }
    async function runOptimization() {
      const button = document.getElementById("optRunButton");
      let ranges = {};
      let categories = {};
      try {
        ranges = collectOptimizationRanges();
        categories = collectOptimizationCategories();
        document.getElementById("optError").textContent = "";
      } catch (err) {
        document.getElementById("optError").textContent = String(err.message || err);
        return;
      }
      const payload = {
        dataset: document.getElementById("optDataset").value,
        direction: document.getElementById("optDirection").value,
        particle_size: Number(document.getElementById("optParticles").value || 24),
        iterations: Number(document.getElementById("optIterations").value || 30),
        top_k: 5,
        ranges,
        categories
      };
      button.disabled = true;
      button.textContent = "Running PSO...";
      document.getElementById("optTable").innerHTML = '<div class="empty" style="padding:14px;">Running PSO. This may take a minute.</div>';
      document.getElementById("optBars").innerHTML = '<div class="empty">Running PSO...</div>';
      document.getElementById("optConvergence").innerHTML = '<div class="empty">Waiting for first iteration...</div>';
      document.getElementById("optScatter").innerHTML = '<div class="empty">Waiting for candidates...</div>';
      document.getElementById("optParallel").innerHTML = '<div class="empty">Waiting for candidates...</div>';
      renderOptProgress({progress: 0.02, message: "Submitting optimization job.", history: [], all_candidates: []});
      document.getElementById("optLinks").textContent = `Particles: ${payload.particle_size}, iterations: ${payload.iterations}, objective: ${payload.direction}, numeric ranges: ${Object.keys(ranges).length}, categorical groups: ${Object.keys(categories).length}.`;
      try {
        const startRes = await fetch("/api/optimization-job", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const started = await startRes.json();
        if (!startRes.ok) throw new Error(started.error || "Could not start PSO optimization");
        let data = null;
        while (true) {
          await new Promise(resolve => setTimeout(resolve, 650));
          const statusRes = await fetch(`/api/optimization-status?job_id=${encodeURIComponent(started.job_id)}`);
          const status = await statusRes.json();
          if (!statusRes.ok) throw new Error(status.error || "Could not read optimization status");
          renderOptProgress(status);
          if (status.all_candidates?.length) {
            document.getElementById("optTable").innerHTML = renderOptimizationTable(status.all_candidates.slice(0, 10));
            document.getElementById("optBars").innerHTML = renderOptimizationBars(status.all_candidates.slice(0, 5), payload.direction);
          }
          if (status.status === "complete") {
            data = status.result;
            break;
          }
          if (status.status === "error") {
            throw new Error(status.error || status.message || "PSO optimization failed");
          }
        }
        document.getElementById("optTable").innerHTML = renderOptimizationTable(data.candidates);
        document.getElementById("optBars").innerHTML = renderOptimizationBars(data.candidates, data.direction);
        renderOptProgress({progress: 1, message: "Optimization complete.", history: data.details.history || [], all_candidates: data.candidates || []});
        const warnings = [...(data.details.range_warnings || []), ...(data.details.category_warnings || [])].map(w => `<span>${escapeHtml(w)}</span>`).join(" ");
        document.getElementById("optLinks").innerHTML = `
          <span>Best predicted removal: ${fmt(data.candidates?.[0]?.prediction, 2)}%. </span>
          <span>Evaluated unique candidates: ${data.details.evaluated_unique_candidates}. </span>
          <span>Range mode: ${escapeHtml(data.details.range_mode)}. </span>
          <span>Category mode: ${escapeHtml(data.details.category_mode || "all_available")}. </span>
          ${warnings ? `<span>${warnings}</span>` : ""}
          <a href="/api/optimization-ranges?dataset=${encodeURIComponent(data.dataset)}" target="_blank">Parameter ranges CSV</a>
          <a href="/api/optimization-csv?dataset=${encodeURIComponent(data.dataset)}&direction=${encodeURIComponent(data.direction)}" target="_blank">Download candidates CSV</a>
          <a href="/api/optimization-xlsx?dataset=${encodeURIComponent(data.dataset)}&direction=${encodeURIComponent(data.direction)}" target="_blank">Download Excel</a>
          <a href="/api/optimization-report?dataset=${encodeURIComponent(data.dataset)}&direction=${encodeURIComponent(data.direction)}" target="_blank">Download report</a>
        `;
      } catch (err) {
        document.getElementById("optTable").innerHTML = `<div class="error" style="padding:14px;">${escapeHtml(err.message || err)}</div>`;
        document.getElementById("optBars").innerHTML = '<div class="empty">Optimization failed.</div>';
        renderOptProgress({progress: 1, message: String(err.message || err), history: [], all_candidates: []});
      } finally {
        button.disabled = false;
        button.textContent = "Run PSO";
      }
    }
    const tourSteps = [
      {
        selector: ".main-tabs",
        title: "Choose the workflow",
        text: "The top navigation separates the module into Prediction, Optimization, Inverse Design, and Documentation so users always know which task they are running."
      },
      {
        selector: ".workflow-guide",
        title: "Follow the decision flow",
        text: "Start with a single prediction, move to optimization when you need top candidates, then use inverse design when you have a target removal and sustainability objectives."
      },
      {
        phase: "prediction",
        selector: "#predictionPhase .panel:first-child",
        title: "Prediction inputs",
        text: "Select the use case, enter PFAS/material/operating conditions, then run Predict. The model returns removal efficiency plus charts and range checks."
      },
      {
        phase: "prediction",
        selector: ".insights",
        title: "Prediction outputs",
        text: "The right side summarizes predicted removal, compares the case with a baseline, shows sensitivity, and warns if inputs are outside familiar training ranges."
      },
      {
        phase: "optimization",
        selector: "#optimizationPhase .optimizer-grid",
        title: "PSO optimization",
        text: "Use this section for a single objective: maximize or minimize predicted removal. Define particle size, iterations, numeric bounds, and allowed categorical choices."
      },
      {
        phase: "inverse",
        selector: "#inversePhase .inverse-grid:first-of-type",
        title: "Target-based inverse design",
        text: "Set a target removal value or range. The BoTorch engine evaluates candidates against the saved PKL model and ranks the best matches."
      },
      {
        phase: "inverse",
        selector: "#lcaLccSection .range-section-header",
        title: "LCA/LCC objectives",
        text: "For Biochar, select country costs from the real map and include LCC and normalized EBI as minimization objectives beside target distance."
      },
      {
        phase: "documentation",
        selector: "#documentationPhase",
        title: "Documentation guide",
        text: "Use this section during demos or handover. It explains the workflows, search bounds, OpenLCA IPC, exports, and model-use limits."
      }
    ];
    let activeTourIndex = 0;
    function storeTourSeen() {
      try { localStorage.setItem("pfasDecisionEngineTourSeen", "1"); } catch (error) {}
    }
    function tourElements() {
      return {
        backdrop: document.getElementById("tourBackdrop"),
        spotlight: document.getElementById("tourSpotlight"),
        card: document.getElementById("tourCard"),
        title: document.getElementById("tourTitle"),
        text: document.getElementById("tourText"),
        count: document.getElementById("tourCount"),
        progress: document.getElementById("tourProgress"),
        prev: document.getElementById("tourPrev"),
        next: document.getElementById("tourNext")
      };
    }
    function positionTour(target) {
      const els = tourElements();
      const pad = 10;
      const rect = target.getBoundingClientRect();
      const top = Math.max(8, rect.top - pad);
      const left = Math.max(8, rect.left - pad);
      const width = Math.min(window.innerWidth - left - 8, rect.width + pad * 2);
      const height = Math.min(window.innerHeight - top - 8, rect.height + pad * 2);
      els.spotlight.style.top = `${top}px`;
      els.spotlight.style.left = `${left}px`;
      els.spotlight.style.width = `${Math.max(80, width)}px`;
      els.spotlight.style.height = `${Math.max(54, height)}px`;
      const cardWidth = Math.min(390, window.innerWidth - 28);
      let cardLeft = rect.right + 18;
      if (cardLeft + cardWidth > window.innerWidth - 14) cardLeft = rect.left - cardWidth - 18;
      if (cardLeft < 14) cardLeft = Math.min(14, window.innerWidth - cardWidth - 14);
      let cardTop = Math.max(14, Math.min(rect.top, window.innerHeight - 270));
      if (window.innerWidth < 760) {
        cardLeft = 14;
        cardTop = Math.min(window.innerHeight - 260, rect.bottom + 16);
        if (cardTop < 14 || cardTop > window.innerHeight - 230) cardTop = 14;
      }
      els.card.style.left = `${cardLeft}px`;
      els.card.style.top = `${cardTop}px`;
    }
    function showTourStep(index) {
      activeTourIndex = Math.max(0, Math.min(tourSteps.length - 1, index));
      const step = tourSteps[activeTourIndex];
      if (step.phase) showPhase(step.phase);
      const els = tourElements();
      els.backdrop.classList.remove("hidden");
      els.backdrop.setAttribute("aria-hidden", "false");
      els.title.textContent = step.title;
      els.text.textContent = step.text;
      els.count.textContent = `Step ${activeTourIndex + 1} of ${tourSteps.length}`;
      els.progress.style.width = `${((activeTourIndex + 1) / tourSteps.length) * 100}%`;
      els.prev.disabled = activeTourIndex === 0;
      els.next.textContent = activeTourIndex === tourSteps.length - 1 ? "Finish" : "Next";
      setTimeout(() => {
        const target = document.querySelector(step.selector) || document.body;
        target.scrollIntoView({behavior: "smooth", block: "center", inline: "center"});
        setTimeout(() => positionTour(target), 260);
      }, 80);
    }
    function startTour() {
      showTourStep(0);
    }
    function endTour() {
      const els = tourElements();
      els.backdrop.classList.add("hidden");
      els.backdrop.setAttribute("aria-hidden", "true");
      storeTourSeen();
    }
    function showPhase(phase) {
      document.getElementById("predictionPhase").classList.toggle("hidden", phase !== "prediction");
      document.getElementById("optimizationPhase").classList.toggle("hidden", phase !== "optimization");
      document.getElementById("inversePhase").classList.toggle("hidden", phase !== "inverse");
      document.getElementById("documentationPhase").classList.toggle("hidden", phase !== "documentation");
      document.querySelectorAll("[data-phase]").forEach(btn => btn.classList.toggle("active", btn.dataset.phase === phase));
      if (phase === "inverse") refreshCountryMapAfterReveal();
    }
    document.querySelectorAll("[data-phase]").forEach(btn => btn.addEventListener("click", () => showPhase(btn.dataset.phase)));
    document.querySelectorAll("[data-dataset]").forEach(btn => btn.addEventListener("click", () => renderDataset(btn.dataset.dataset)));
    document.getElementById("tourStartButton").addEventListener("click", startTour);
    document.getElementById("tourSkip").addEventListener("click", endTour);
    document.getElementById("tourPrev").addEventListener("click", () => showTourStep(activeTourIndex - 1));
    document.getElementById("tourNext").addEventListener("click", () => {
      if (activeTourIndex >= tourSteps.length - 1) endTour();
      else showTourStep(activeTourIndex + 1);
    });
    window.addEventListener("resize", () => {
      if (!document.getElementById("tourBackdrop").classList.contains("hidden")) {
        const step = tourSteps[activeTourIndex];
        const target = document.querySelector(step.selector) || document.body;
        positionTour(target);
      }
    });
    document.addEventListener("keydown", event => {
      if (event.key === "Escape" && !document.getElementById("tourBackdrop").classList.contains("hidden")) endTour();
    });
    document.getElementById("optDataset").addEventListener("change", event => renderOptimizationSetup(event.target.value));
    document.getElementById("optRobustRanges").addEventListener("click", () => setOptimizationRangeMode("robust"));
    document.getElementById("optFullRanges").addEventListener("click", () => setOptimizationRangeMode("full"));
    document.getElementById("invDataset").addEventListener("change", event => renderInverseSetup(event.target.value));
    document.getElementById("invRobustRanges").addEventListener("click", () => setInverseRangeMode("robust"));
    document.getElementById("invFullRanges").addEventListener("click", () => setInverseRangeMode("full"));
    document.getElementById("invTargetMode").addEventListener("change", syncInverseTargetMode);
    document.getElementById("invExecutionMode").addEventListener("change", syncInverseExecutionMode);
    document.getElementById("invEolMode").addEventListener("change", syncLcaProductSystem);
    document.getElementById("invObjectiveMode").addEventListener("change", syncLcaLccControls);
    document.getElementById("invEnvironmentalMode").addEventListener("change", syncLcaEvaluatorMode);
    document.getElementById("invTestOpenLca").addEventListener("click", testOpenLca);
    document.getElementById("invCountry").addEventListener("change", () => {
      applyCountryCostsToConstants();
      updateCountryMapSelection(true);
    });
    document.getElementById("countrySearch").addEventListener("input", () => updateCountryMapSelection(false));
    document.getElementById("resetButton").addEventListener("click", () => { resetToDefaults(); setEmptyState(); });
    document.getElementById("predictButton").addEventListener("click", predict);
    document.getElementById("optRunButton").addEventListener("click", runOptimization);
    document.getElementById("invRunButton").addEventListener("click", runInverseDesign);
    fetch("/api/metadata").then(r => r.json()).then(data => {
      assets = data;
      renderLcaConstants();
      renderCountrySelector();
      const meta = lcaMeta();
      document.getElementById("invIpcPort").value = meta.ipc_default_port || 8080;
      document.getElementById("invImpactMethod").value = meta.impact_method_default || "ReCiPe Midpoint (H)";
      document.getElementById("invEnvironmentalMode").value = meta.default_environmental_mode || (meta.cloud_evaluator_configured ? "openlca_cloud" : "proxy");
      syncLcaProductSystem();
      renderDataset("dataset1");
      renderOptimizationSetup(document.getElementById("optDataset").value);
      renderInverseSetup(document.getElementById("invDataset").value);
      syncInverseTargetMode();
      syncInverseExecutionMode();
      syncLcaEvaluatorMode();
      setTimeout(() => {
        let seen = false;
        try { seen = localStorage.getItem("pfasDecisionEngineTourSeen") === "1"; } catch (error) {}
        if (!seen) startTour();
      }, 850);
    }).catch(err => {
      errorBox.textContent = String(err.message || err);
    });
  </script>
</body>
</html>
"""


def load_pickle(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def app_path(path_text: str) -> Path:
    return ROOT / str(path_text).replace("\\", "/")


def _label(key: str) -> str:
    labels = {
        "initial_pfas_concentration_ug_L": "Initial PFAS concentration",
        "resin_dosage_mg_L": "Resin dosage",
        "temperature_C": "Temperature",
        "contact_time_h": "Contact time",
        "stirring_rate_rpm": "Stirring rate",
        "CDOC_mg_L": "DOC",
        "Polymer_matrix ": "Polymer matrix",
        "Resin_type": "Resin type",
        "Resin profile": "Resin profile",
    }
    return labels.get(key, key)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class PredictorState:
    def __init__(self) -> None:
        self.assets = load_json(ASSETS_PATH)
        self.models = {
            "dataset1": load_pickle(app_path(self.assets["models"]["dataset1"])),
            "dataset2": load_pickle(app_path(self.assets["models"]["dataset2"])),
        }
        self.optimization_spaces = None
        self.latest_prediction: dict[str, dict] = {}
        self.latest_optimization: dict[tuple[str, str], dict] = {}
        self.optimization_jobs: dict[str, dict] = {}
        self.optimization_jobs_lock = threading.Lock()
        self.latest_inverse: dict[str, dict] = {}
        self.inverse_jobs: dict[str, dict] = {}
        self.inverse_jobs_lock = threading.Lock()
        self.lca_lcc = BiocharLcaLccEvaluator()

    def metadata(self) -> dict:
        data = dict(self.assets)
        data["lca_lcc"] = self.lca_lcc.metadata()
        return data

    def openlca_status(self, mode: str, port: int, product_system: str, impact_method: str) -> dict:
        mode = (mode or "proxy").strip().lower()
        if mode == "proxy":
            return {
                "available": True,
                "mode": "proxy",
                "message": "Offline proxy EBI/LCC evaluator is available.",
            }
        if mode in {"openlca_cloud", "cloud", "remote"}:
            return self.lca_lcc.cloud_status(
                {
                    "product_system": product_system,
                    "impact_method": impact_method,
                    "api_key": os.environ.get("OPENLCA_EVALUATOR_API_KEY", ""),
                }
            )
        if mode in {"openlca_rest", "gdt_server", "openlca_gdt"}:
            return self.lca_lcc.openlca_rest_status(
                {
                    "product_system": product_system,
                    "impact_method": impact_method,
                    "openlca_rest_url": os.environ.get("OPENLCA_REST_URL", ""),
                }
            )
        try:
            import olca_ipc as ipc
            import olca_schema as o
        except Exception as exc:
            return {
                "available": False,
                "mode": "openlca_ipc",
                "port": port,
                "message": f"olca-ipc Python package is not available: {exc}",
            }
        try:
            client = ipc.Client(int(port))
            product_ref = client.find(o.ProductSystem, product_system) if product_system else None
            method_ref = client.find(o.ImpactMethod, impact_method) if impact_method else None
            return {
                "available": True,
                "mode": "openlca_ipc",
                "port": int(port),
                "product_system": product_system,
                "impact_method": impact_method,
                "product_system_found": product_ref is not None,
                "impact_method_found": method_ref is not None,
                "message": "OpenLCA IPC responded.",
            }
        except Exception as exc:
            return {
                "available": False,
                "mode": "openlca_ipc",
                "port": int(port),
                "product_system": product_system,
                "impact_method": impact_method,
                "message": str(exc),
            }

    def _optimization_space_map(self) -> dict[str, object]:
        if self.optimization_spaces is None:
            self.optimization_spaces = {space.key: space for space in load_search_spaces()}
        return self.optimization_spaces

    def _row(self, dataset: str, values: dict):
        bundle = self.models[dataset]
        ds_assets = self.assets["datasets"][dataset]
        if dataset == "dataset1":
            values = dict(values)
            selected_pfas = values.get("PFAS") or ds_assets["defaults"].get("PFAS")
            smiles_map = ds_assets.get("pfas_smiles_map", {})
            if selected_pfas in smiles_map:
                values["SMILES"] = smiles_map[selected_pfas]
            return build_dataset1_row(values, ds_assets["defaults"], bundle["input_columns"])
        return build_dataset2_row(values, ds_assets["defaults"], bundle["input_columns"])

    def _predict_value(self, dataset: str, values: dict) -> float:
        bundle = self.models[dataset]
        X = self._row(dataset, values)
        return float(np.clip(bundle["model"].predict(X)[0], 0, 100))

    def _baseline_values(self, dataset: str) -> dict:
        ds_assets = self.assets["datasets"][dataset]
        if dataset == "dataset1":
            return {key: ds_assets["defaults"].get(key) for key in ds_assets.get("numeric_fields", [])} | {
                "PFAS": ds_assets["defaults"].get("PFAS", "")
            }
        defaults = ds_assets["defaults"]
        values = {key: defaults.get(key) for key in ds_assets.get("categorical_fields", [])}
        values.update(
            {
                "initial_pfas_concentration_ug_L": (defaults.get("initial_concentration_mg_L") or 0) * 1000,
                "resin_dosage_mg_L": (defaults.get("resin_dosage_g_L") or 0) * 1000,
                "pH": defaults.get("pH"),
                "temperature_C": defaults.get("temperature\n(℃)"),
                "contact_time_h": defaults.get("contact_time\n(h)"),
                "stirring_rate_rpm": defaults.get("Stirring_rate_numeric"),
                "CDOC_mg_L": defaults.get("CDOC\n(mg/L)"),
            }
        )
        return values

    def _sensitivity(self, dataset: str, values: dict) -> list[dict]:
        ds_assets = self.assets["datasets"][dataset]
        ranges = ds_assets.get("ranges", {})
        items = []
        for field in ds_assets.get("sensitivity_fields", []):
            r = ranges.get(field)
            if not r:
                continue
            low = r.get("p10")
            high = r.get("p90")
            if low is None or high is None or low == high:
                continue
            low_values = dict(values)
            high_values = dict(values)
            low_values[field] = low
            high_values[field] = high
            low_pred = self._predict_value(dataset, low_values)
            high_pred = self._predict_value(dataset, high_values)
            items.append(
                {
                    "field": field,
                    "label": _label(field),
                    "low": low,
                    "high": high,
                    "low_prediction": low_pred,
                    "high_prediction": high_pred,
                    "swing": abs(high_pred - low_pred),
                }
            )
        return sorted(items, key=lambda x: x["swing"], reverse=True)[:7]

    def _range_status(self, dataset: str, values: dict) -> list[dict]:
        ds_assets = self.assets["datasets"][dataset]
        statuses = []
        for field, r in ds_assets.get("ranges", {}).items():
            if field not in values or values.get(field) in ("", None):
                continue
            try:
                value = float(values[field])
                min_val = float(r["min"])
                max_val = float(r["max"])
                p05 = float(r["p05"])
                p95 = float(r["p95"])
            except Exception:
                continue
            if max_val == min_val:
                position = 50.0
            else:
                position = (value - min_val) / (max_val - min_val) * 100.0
            if value < min_val or value > max_val:
                status = "out_of_range"
            elif value < p05 or value > p95:
                status = "near_edge"
            else:
                status = "inside"
            statuses.append(
                {
                    "kind": "numeric",
                    "field": field,
                    "label": _label(field),
                    "value": value,
                    "min": min_val,
                    "max": max_val,
                    "p05": p05,
                    "p95": p95,
                    "position": position,
                    "status": status,
                }
            )
        for field in ds_assets.get("categorical_fields", []):
            if field not in values:
                continue
            options = {str(x) for x in ds_assets.get("options", {}).get(field, [])}
            status = "known" if str(values[field]) in options else "unknown_category"
            statuses.append({"kind": "categorical", "field": field, "label": _label(field), "status": status})
        if dataset == "dataset1" and "PFAS" in values:
            options = {str(x) for x in ds_assets.get("pfas_options", [])}
            status = "known" if str(values["PFAS"]) in options else "unknown_category"
            statuses.append({"kind": "categorical", "field": "PFAS", "label": "PFAS", "status": status})
        priority = {"out_of_range": 0, "unknown_category": 1, "near_edge": 2, "inside": 3, "known": 4}
        return sorted(statuses, key=lambda x: (priority.get(x["status"], 9), x["label"]))[:12]

    def predict(self, dataset: str, values: dict) -> dict:
        if dataset not in self.models:
            raise ValueError(f"Unknown dataset: {dataset}")
        bundle = self.models[dataset]
        baseline_values = self._baseline_values(dataset)
        prediction = self._predict_value(dataset, values)
        baseline_prediction = self._predict_value(dataset, baseline_values)
        result = {
            "dataset": dataset,
            "prediction": prediction,
            "baseline_prediction": baseline_prediction,
            "target": bundle["target"],
            "model_name": bundle.get("best_model_name"),
            "sensitivity": self._sensitivity(dataset, values),
            "range_status": self._range_status(dataset, values),
            "values": dict(values),
        }
        self.latest_prediction[dataset] = result
        return result

    def latest_prediction_xlsx(self, dataset: str) -> bytes:
        if dataset not in self.latest_prediction:
            raise ValueError("No prediction result is available yet for this dataset.")
        result = self.latest_prediction[dataset]
        sheets = {
            "Prediction": pd.DataFrame([{k: v for k, v in result.items() if k not in {"sensitivity", "range_status", "values"}}]),
            "Inputs": pd.DataFrame([result.get("values") or {}]),
            "Sensitivity": pd.DataFrame(result.get("sensitivity") or []),
            "Range check": pd.DataFrame(result.get("range_status") or []),
        }
        return self._xlsx_from_sheets(sheets)

    def latest_prediction_report(self, dataset: str) -> str:
        if dataset not in self.latest_prediction:
            raise ValueError("No prediction result is available yet for this dataset.")
        result = self.latest_prediction[dataset]
        summary = pd.DataFrame(
            [
                {"metric": "Prediction", "value": f"{result.get('prediction'):.3f}%"},
                {"metric": "Baseline prediction", "value": f"{result.get('baseline_prediction'):.3f}%"},
                {"metric": "Model", "value": result.get("model_name")},
            ]
        )
        inputs = pd.DataFrame([result.get("values") or {}])
        sensitivity = pd.DataFrame(result.get("sensitivity") or [])
        ranges = pd.DataFrame(result.get("range_status") or [])
        return "".join(
            [
                "<!doctype html><html><head><meta charset='utf-8'><title>Prediction report</title>",
                "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:28px;color:#142033}table{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0 24px}th,td{border:1px solid #d7dde7;padding:6px 8px;text-align:left}th{background:#f4f6f8}</style></head><body>",
                "<h1>Prediction report</h1>",
                summary.to_html(index=False, escape=True),
                "<h2>Inputs</h2>",
                inputs.to_html(index=False, escape=True),
                "<h2>Sensitivity</h2>",
                sensitivity.to_html(index=False, escape=True) if not sensitivity.empty else "<p>No sensitivity data.</p>",
                "<h2>Range check</h2>",
                ranges.to_html(index=False, escape=True) if not ranges.empty else "<p>No range-check data.</p>",
                "</body></html>",
            ]
        )

    def optimization_ranges_csv(self, dataset: str) -> str:
        spaces = self._optimization_space_map()
        if dataset not in spaces:
            raise ValueError(f"Unknown dataset: {dataset}")
        df = parameter_ranges(spaces[dataset])
        return df.to_csv(index=False)

    def _choice_label(self, category_name: str, choice) -> str:
        if isinstance(choice, (tuple, list)):
            if category_name == "Resin profile" and len(choice) >= 5:
                return f"{choice[0]} | {choice[1]} | {choice[2]} | {choice[3]} | {choice[4]}"
            return " | ".join(str(item) for item in choice)
        return str(choice)

    def optimization_space(self, dataset: str) -> dict:
        spaces = self._optimization_space_map()
        if dataset not in spaces:
            raise ValueError(f"Unknown dataset: {dataset}")
        space = spaces[dataset]
        return {
            "dataset": dataset,
            "title": space.title,
            "numeric": [
                {
                    "name": num.name,
                    "label": _label(num.name),
                    "optimization_lower": num.lower,
                    "optimization_upper": num.upper,
                    "training_min": num.source_lower,
                    "training_max": num.source_upper,
                }
                for num in space.numeric
            ],
            "categorical": [
                {
                    "name": cat.name,
                    "label": _label(cat.name),
                    "choices": [
                        {"id": str(idx), "label": self._choice_label(cat.name, choice)}
                        for idx, choice in enumerate(cat.choices)
                    ],
                }
                for cat in space.categorical
            ],
        }

    def latest_optimization_csv(self, dataset: str, direction: str) -> str:
        key = (dataset, direction)
        if key not in self.latest_optimization:
            raise ValueError("No optimization result is available yet for this dataset/objective.")
        df = pd.DataFrame(self.latest_optimization[key]["candidates"])
        return df.to_csv(index=False)

    def _flatten_records(self, rows: list[dict]) -> list[dict]:
        flat_rows = []
        for row in rows:
            flat = {}
            for key, value in row.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat[f"{key}.{sub_key}"] = sub_value
                elif isinstance(value, list):
                    flat[key] = json.dumps(value, ensure_ascii=False)
                else:
                    flat[key] = value
            flat_rows.append(flat)
        return flat_rows

    def _xlsx_from_sheets(self, sheets: dict[str, pd.DataFrame]) -> bytes:
        output = io.BytesIO()
        with pd.ExcelWriter(output) as writer:
            for name, df in sheets.items():
                safe_name = "".join(ch for ch in name if ch not in r"[]:*?/\\")[:31] or "Sheet"
                df.to_excel(writer, sheet_name=safe_name, index=False)
        return output.getvalue()

    def latest_optimization_xlsx(self, dataset: str, direction: str) -> bytes:
        key = (dataset, direction)
        if key not in self.latest_optimization:
            raise ValueError("No optimization result is available yet for this dataset/objective.")
        result = self.latest_optimization[key]
        rows = self._flatten_records(result.get("candidates") or [])
        details = result.get("details", {}) or {}
        sheets = {
            "Top candidates": pd.DataFrame(rows),
            "Convergence": pd.DataFrame(details.get("history") or []),
            "Numeric ranges": pd.DataFrame(details.get("numeric_ranges") or []),
            "Categorical choices": pd.DataFrame(details.get("categorical_choices") or []),
        }
        return self._xlsx_from_sheets(sheets)

    def latest_inverse_csv(self, dataset: str) -> str:
        if dataset not in self.latest_inverse:
            raise ValueError("No inverse-design result is available yet for this dataset.")
        result = self.latest_inverse[dataset]
        rows = result.get("details", {}).get("all_candidates") or result.get("candidates") or []
        df = pd.DataFrame(self._flatten_records(rows))
        return df.to_csv(index=False)

    def latest_inverse_xlsx(self, dataset: str) -> bytes:
        if dataset not in self.latest_inverse:
            raise ValueError("No inverse-design result is available yet for this dataset.")
        result = self.latest_inverse[dataset]
        details = result.get("details", {}) or {}
        all_rows = self._flatten_records(details.get("all_candidates") or result.get("candidates") or [])
        top_rows = self._flatten_records(result.get("candidates") or [])
        sheets = {
            "Top candidates": pd.DataFrame(top_rows),
            "All evaluated candidates": pd.DataFrame(all_rows),
            "Convergence": pd.DataFrame(details.get("history") or []),
            "Numeric ranges": pd.DataFrame(details.get("numeric_ranges") or []),
            "Categorical choices": pd.DataFrame(details.get("categorical_choices") or []),
        }
        if details.get("lca_lcc_enabled"):
            sheets["LCA constants"] = pd.DataFrame(
                [{"parameter": k, "value": v} for k, v in (details.get("lca_lcc_options", {}).get("constants") or {}).items()]
            )
        return self._xlsx_from_sheets(sheets)

    def _html_report(self, title: str, result: dict) -> str:
        details = result.get("details", {}) or {}
        candidates = pd.DataFrame(self._flatten_records(result.get("candidates") or [])).head(20)
        history = pd.DataFrame(details.get("history") or [])
        summary_rows = [{"metric": key, "value": value} for key, value in details.items() if key not in {"all_candidates", "history", "lca_lcc_metadata", "numeric_ranges", "categorical_choices"}]
        summary = pd.DataFrame(summary_rows)
        html_parts = [
            "<!doctype html><html><head><meta charset='utf-8'><title>{}</title>".format(title),
            "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:28px;color:#142033}table{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0 24px}th,td{border:1px solid #d7dde7;padding:6px 8px;text-align:left;vertical-align:top}th{background:#f4f6f8}h1,h2{margin-bottom:6px}.muted{color:#667085}</style>",
            "</head><body>",
            f"<h1>{title}</h1><div class='muted'>Generated from the local PFAS predictor app.</div>",
            "<h2>Highlights</h2>",
            summary.to_html(index=False, escape=True) if not summary.empty else "<p>No summary available.</p>",
            "<h2>Top candidates</h2>",
            candidates.to_html(index=False, escape=True) if not candidates.empty else "<p>No candidates available.</p>",
            "<h2>Convergence</h2>",
            history.to_html(index=False, escape=True) if not history.empty else "<p>No convergence history available.</p>",
            "</body></html>",
        ]
        return "".join(html_parts)

    def latest_optimization_report(self, dataset: str, direction: str) -> str:
        key = (dataset, direction)
        if key not in self.latest_optimization:
            raise ValueError("No optimization result is available yet for this dataset/objective.")
        result = self.latest_optimization[key]
        return self._html_report(f"Optimization report - {result.get('title', dataset)}", result)

    def latest_inverse_report(self, dataset: str) -> str:
        if dataset not in self.latest_inverse:
            raise ValueError("No inverse-design result is available yet for this dataset.")
        result = self.latest_inverse[dataset]
        return self._html_report(f"Inverse-design report - {result.get('title', dataset)}", result)

    def _space_with_user_ranges(self, space, ranges: dict | None):
        if not isinstance(ranges, dict) or not ranges:
            return space, [], False

        warnings_out: list[str] = []
        used = False
        numeric = []
        known_fields = {num.name for num in space.numeric}
        for num in space.numeric:
            item = ranges.get(num.name)
            if not isinstance(item, dict):
                numeric.append(num)
                continue
            lower = item.get("lower", num.lower)
            upper = item.get("upper", num.upper)
            try:
                lower = float(lower)
                upper = float(upper)
            except Exception as exc:
                raise ValueError(f"{_label(num.name)} range must be numeric.") from exc
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(f"{_label(num.name)} range must be finite.")
            if lower > upper:
                raise ValueError(f"{_label(num.name)} lower bound cannot be greater than upper bound.")
            if lower < num.source_lower or upper > num.source_upper:
                warnings_out.append(
                    f"{_label(num.name)} extends outside training range ({num.source_lower:.4g} to {num.source_upper:.4g})."
                )
            numeric.append(replace(num, lower=lower, upper=upper))
            used = True

        unknown = sorted(set(ranges) - known_fields)
        if unknown:
            warnings_out.append(f"Ignored unknown range fields: {', '.join(unknown[:5])}.")
        return replace(space, numeric=numeric), warnings_out, used

    def _space_with_user_categories(self, space, categories: dict | None):
        if not isinstance(categories, dict) or not categories:
            return space, [], False

        warnings_out: list[str] = []
        used = False
        categorical = []
        known_fields = {cat.name for cat in space.categorical}
        for cat in space.categorical:
            selected = categories.get(cat.name)
            if selected in (None, ""):
                categorical.append(cat)
                continue
            if not isinstance(selected, list):
                raise ValueError(f"{_label(cat.name)} category selection must be a list.")

            selected_indices: list[int] = []
            choice_text_to_index = {str(choice): idx for idx, choice in enumerate(cat.choices)}
            choice_label_to_index = {self._choice_label(cat.name, choice): idx for idx, choice in enumerate(cat.choices)}
            for raw in selected:
                try:
                    idx = int(raw)
                except Exception:
                    idx = choice_text_to_index.get(str(raw), choice_label_to_index.get(str(raw), -1))
                if 0 <= idx < len(cat.choices):
                    selected_indices.append(idx)

            selected_indices = sorted(set(selected_indices))
            if not selected_indices:
                raise ValueError(f"{_label(cat.name)} needs at least one valid selected value.")
            selected_choices = tuple(cat.choices[idx] for idx in selected_indices)
            categorical.append(replace(cat, choices=selected_choices))
            used = True

        unknown = sorted(set(categories) - known_fields)
        if unknown:
            warnings_out.append(f"Ignored unknown categorical fields: {', '.join(unknown[:5])}.")
        return replace(space, categorical=categorical), warnings_out, used

    def optimize(
        self,
        dataset: str,
        direction: str,
        particle_size: int,
        iterations: int,
        top_k: int = 5,
        ranges: dict | None = None,
        categories: dict | None = None,
        progress_callback=None,
    ) -> dict:
        spaces = self._optimization_space_map()
        if dataset not in spaces:
            raise ValueError(f"Unknown dataset: {dataset}")
        if direction not in {"max", "min"}:
            raise ValueError("Objective must be 'max' or 'min'.")
        particle_size = int(np.clip(particle_size, 4, 80))
        iterations = int(np.clip(iterations, 1, 120))
        top_k = int(np.clip(top_k, 1, 10))
        search_space, range_warnings, user_ranges_used = self._space_with_user_ranges(spaces[dataset], ranges)
        search_space, category_warnings, user_categories_used = self._space_with_user_categories(search_space, categories)
        top_df, details = run_pso(
            search_space,
            direction,
            particle_size=particle_size,
            iterations=iterations,
            top_k=top_k,
            progress_callback=progress_callback,
        )
        details["range_mode"] = "user_specified" if user_ranges_used else details.get("range_mode")
        details["category_mode"] = "user_specified" if user_categories_used else "all_available"
        details["range_warnings"] = range_warnings
        details["category_warnings"] = category_warnings
        details["numeric_ranges"] = [
            {
                "parameter": num.name,
                "optimization_lower": num.lower,
                "optimization_upper": num.upper,
                "training_min": num.source_lower,
                "training_max": num.source_upper,
            }
            for num in search_space.numeric
        ]
        details["categorical_choices"] = [
            {
                "parameter": cat.name,
                "choices_count": len(cat.choices),
                "choices": [self._choice_label(cat.name, choice) for choice in cat.choices[:30]],
            }
            for cat in search_space.categorical
        ]
        top_df = top_df.replace({np.nan: None})
        top_df.insert(0, "rank", range(1, len(top_df) + 1))
        candidates = top_df.to_dict(orient="records")
        result = {
            "dataset": dataset,
            "title": search_space.title,
            "direction": direction,
            "candidates": candidates,
            "details": details,
        }
        self.latest_optimization[(dataset, direction)] = result
        return result

    def _update_optimization_job(self, job_id: str, **updates) -> None:
        with self.optimization_jobs_lock:
            job = self.optimization_jobs.get(job_id)
            if job is None:
                return
            job.update(_json_safe(updates))
            job["updated_at"] = time.time()

    def start_optimization_job(self, payload: dict) -> dict:
        job_id = uuid.uuid4().hex
        now = time.time()
        with self.optimization_jobs_lock:
            self.optimization_jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "phase": "queued",
                "message": "Queued optimization run.",
                "progress": 0.0,
                "history": [],
                "all_candidates": [],
                "created_at": now,
                "updated_at": now,
            }

        def progress(event: dict) -> None:
            self._update_optimization_job(job_id, **event)

        def worker() -> None:
            self._update_optimization_job(
                job_id,
                status="running",
                phase="setup",
                message="Preparing PSO search space.",
                progress=0.01,
            )
            try:
                result = self.optimize(
                    payload.get("dataset", "dataset1"),
                    payload.get("direction", "max"),
                    int(payload.get("particle_size", 24)),
                    int(payload.get("iterations", 30)),
                    int(payload.get("top_k", 5)),
                    payload.get("ranges") or {},
                    payload.get("categories") or {},
                    progress_callback=progress,
                )
                self._update_optimization_job(
                    job_id,
                    status="complete",
                    phase="complete",
                    message="Optimization complete.",
                    progress=1.0,
                    result=result,
                    history=result.get("details", {}).get("history", []),
                    all_candidates=result.get("candidates", []),
                )
            except Exception as exc:
                self._update_optimization_job(
                    job_id,
                    status="error",
                    phase="error",
                    message=str(exc),
                    progress=1.0,
                    error=str(exc),
                    traceback=traceback.format_exc(limit=8),
                )

        threading.Thread(target=worker, daemon=True).start()
        return {"job_id": job_id}

    def optimization_job_status(self, job_id: str) -> dict:
        with self.optimization_jobs_lock:
            job = self.optimization_jobs.get(job_id)
            if job is None:
                raise ValueError("Unknown optimization job.")
            return dict(job)

    def inverse_design(self, payload: dict, progress_callback=None) -> dict:
        dataset = payload.get("dataset", "dataset1")
        spaces = self._optimization_space_map()
        if dataset not in spaces:
            raise ValueError(f"Unknown dataset: {dataset}")

        search_space, range_warnings, user_ranges_used = self._space_with_user_ranges(
            spaces[dataset],
            payload.get("ranges") or {},
        )
        search_space, category_warnings, user_categories_used = self._space_with_user_categories(
            search_space,
            payload.get("categories") or {},
        )

        include_lca_lcc = bool(payload.get("include_lca_lcc", False))
        if include_lca_lcc and dataset != "dataset1":
            raise ValueError("LCA/LCC Pareto inverse design is currently configured for the Biochar use case only.")

        lca_lcc_options = {
            "environmental_mode": payload.get("environmental_mode", "proxy"),
            "ipc_port": int(payload.get("ipc_port") or 8080),
            "product_system": payload.get("product_system") or "",
            "impact_method": payload.get("impact_method") or "",
            "eol_mode": payload.get("eol_mode", "incineration"),
            "fallback_to_proxy": bool(payload.get("fallback_to_proxy", True)),
            "constants": payload.get("lca_constants") or {},
            "country": payload.get("country") or payload.get("country_iso3") or "",
        }

        def sustainability_evaluator(values: dict, prediction: float) -> dict:
            evaluation = self.lca_lcc.evaluate(values, prediction, lca_lcc_options)
            costs = evaluation.get("cost_breakdown", {}) or {}
            lci = evaluation.get("lci", {}) or {}
            return {
                "lcc_total_usd_m3": evaluation.get("lcc_total_usd_m3"),
                "ebi": evaluation.get("ebi"),
                "environmental_source": evaluation.get("environmental_source"),
                "lca_warning": evaluation.get("lca_warning"),
                "eol_mode": evaluation.get("eol_mode"),
                "product_system": evaluation.get("product_system"),
                "country": evaluation.get("country"),
                "country_iso3": evaluation.get("country_iso3"),
                "cost_p01_USD": costs.get("cost_p01_USD"),
                "cost_p02_USD": costs.get("cost_p02_USD"),
                "cost_p03_USD": costs.get("cost_p03_USD"),
                "cost_p04_USD": costs.get("cost_p04_USD"),
                "cost_p05_USD": costs.get("cost_p05_USD"),
                "cost_p06_USD": costs.get("cost_p06_USD"),
                "m_biochar_kg": lci.get("m_biochar_kg"),
                "m_PFAS_removed_kg": lci.get("m_PFAS_removed_kg"),
                "m_spent_biochar_kg": lci.get("m_spent_biochar_kg"),
                "normalized_impacts": evaluation.get("normalized_impacts") or {},
                "impact_vector": evaluation.get("impact_vector") or {},
                "cost_breakdown": costs,
            }

        top_df, details = run_inverse_design(
            search_space,
            target_mode=payload.get("target_mode", "target_value"),
            target_value=payload.get("target_value"),
            target_min=payload.get("target_min"),
            target_max=payload.get("target_max"),
            tolerance=payload.get("tolerance", 0.0),
            tolerance_mode=payload.get("tolerance_mode", "absolute"),
            initialization_strategy=payload.get("initialization_strategy", "sobol"),
            initialization_trials=int(payload.get("initialization_trials", 16)),
            execution_mode=payload.get("execution_mode", "batch"),
            batch_iterations=int(payload.get("batch_iterations", 20)),
            batch_size=int(payload.get("batch_size", 5)),
            raw_samples=int(payload.get("raw_samples", payload.get("particle_size", 64))),
            top_k=int(payload.get("top_k", 10)),
            candidate_evaluator=sustainability_evaluator if include_lca_lcc else None,
            multi_objective=include_lca_lcc,
            progress_callback=progress_callback,
        )
        details["range_mode"] = "user_specified" if user_ranges_used else "p05_p95"
        details["category_mode"] = "user_specified" if user_categories_used else "all_available"
        details["range_warnings"] = range_warnings
        details["category_warnings"] = category_warnings
        details["lca_lcc_enabled"] = include_lca_lcc
        if include_lca_lcc:
            details["lca_lcc_options"] = lca_lcc_options
            details["lca_lcc_metadata"] = self.lca_lcc.metadata()
        details["numeric_ranges"] = [
            {
                "parameter": num.name,
                "optimization_lower": num.lower,
                "optimization_upper": num.upper,
                "training_min": num.source_lower,
                "training_max": num.source_upper,
            }
            for num in search_space.numeric
        ]
        details["categorical_choices"] = [
            {
                "parameter": cat.name,
                "choices_count": len(cat.choices),
                "choices": [self._choice_label(cat.name, choice) for choice in cat.choices[:30]],
            }
            for cat in search_space.categorical
        ]
        top_df = top_df.replace({np.nan: None})
        candidates = top_df.to_dict(orient="records")
        result = {
            "dataset": dataset,
            "title": search_space.title,
            "candidates": candidates,
            "details": details,
        }
        self.latest_inverse[dataset] = result
        return result

    def _update_inverse_job(self, job_id: str, **updates) -> None:
        with self.inverse_jobs_lock:
            job = self.inverse_jobs.get(job_id)
            if job is None:
                return
            job.update(_json_safe(updates))
            job["updated_at"] = time.time()

    def start_inverse_design_job(self, payload: dict) -> dict:
        job_id = uuid.uuid4().hex
        now = time.time()
        with self.inverse_jobs_lock:
            self.inverse_jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "phase": "queued",
                "message": "Queued inverse-design run.",
                "progress": 0.0,
                "history": [],
                "all_candidates": [],
                "created_at": now,
                "updated_at": now,
            }

        def progress(event: dict) -> None:
            self._update_inverse_job(job_id, **event)

        def worker() -> None:
            self._update_inverse_job(
                job_id,
                status="running",
                phase="setup",
                message="Preparing search space and target-distance evaluator.",
                progress=0.01,
            )
            try:
                result = self.inverse_design(payload, progress_callback=progress)
                self._update_inverse_job(
                    job_id,
                    status="complete",
                    phase="complete",
                    message="Inverse design complete.",
                    progress=1.0,
                    result=result,
                    history=result.get("details", {}).get("history", []),
                    all_candidates=result.get("details", {}).get("all_candidates", []),
                )
            except Exception as exc:
                self._update_inverse_job(
                    job_id,
                    status="error",
                    phase="error",
                    message=str(exc),
                    progress=1.0,
                    error=str(exc),
                    traceback=traceback.format_exc(limit=8),
                )

        threading.Thread(target=worker, daemon=True).start()
        return {"job_id": job_id}

    def inverse_job_status(self, job_id: str) -> dict:
        with self.inverse_jobs_lock:
            job = self.inverse_jobs.get(job_id)
            if job is None:
                raise ValueError("Unknown inverse-design job.")
            return dict(job)


STATE: PredictorState | None = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, status: int = 200, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200) -> None:
        self._send(json.dumps(_json_safe(payload), ensure_ascii=False).encode("utf-8"), status, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(HTML.encode("utf-8"), 200, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/metadata":
            self._json(STATE.metadata())
            return
        if parsed.path == "/api/optimization-space":
            try:
                query = parse_qs(parsed.query)
                dataset = query.get("dataset", ["dataset1"])[0]
                self._json(STATE.optimization_space(dataset))
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/prediction-xlsx":
            try:
                query = parse_qs(parsed.query)
                dataset = query.get("dataset", ["dataset1"])[0]
                data = STATE.latest_prediction_xlsx(dataset)
                self._send(data, 200, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/prediction-report":
            try:
                query = parse_qs(parsed.query)
                dataset = query.get("dataset", ["dataset1"])[0]
                html = STATE.latest_prediction_report(dataset)
                self._send(html.encode("utf-8"), 200, "text/html; charset=utf-8")
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/openlca-status":
            try:
                query = parse_qs(parsed.query)
                mode = query.get("mode", [os.environ.get("PFAS_ENVIRONMENTAL_MODE", "proxy")])[0]
                port = int(query.get("port", ["8080"])[0] or 8080)
                product_system = query.get("product_system", [""])[0]
                impact_method = query.get("impact_method", [""])[0]
                self._json(STATE.openlca_status(mode, port, product_system, impact_method))
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/inverse-status":
            try:
                query = parse_qs(parsed.query)
                job_id = query.get("job_id", [""])[0]
                self._json(STATE.inverse_job_status(job_id))
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/optimization-status":
            try:
                query = parse_qs(parsed.query)
                job_id = query.get("job_id", [""])[0]
                self._json(STATE.optimization_job_status(job_id))
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/optimization-ranges":
            try:
                query = parse_qs(parsed.query)
                dataset = query.get("dataset", ["dataset1"])[0]
                csv_text = STATE.optimization_ranges_csv(dataset)
                self._send(csv_text.encode("utf-8"), 200, "text/csv; charset=utf-8")
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/optimization-csv":
            try:
                query = parse_qs(parsed.query)
                dataset = query.get("dataset", ["dataset1"])[0]
                direction = query.get("direction", ["max"])[0]
                csv_text = STATE.latest_optimization_csv(dataset, direction)
                self._send(csv_text.encode("utf-8"), 200, "text/csv; charset=utf-8")
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/optimization-xlsx":
            try:
                query = parse_qs(parsed.query)
                dataset = query.get("dataset", ["dataset1"])[0]
                direction = query.get("direction", ["max"])[0]
                data = STATE.latest_optimization_xlsx(dataset, direction)
                self._send(data, 200, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/optimization-report":
            try:
                query = parse_qs(parsed.query)
                dataset = query.get("dataset", ["dataset1"])[0]
                direction = query.get("direction", ["max"])[0]
                html = STATE.latest_optimization_report(dataset, direction)
                self._send(html.encode("utf-8"), 200, "text/html; charset=utf-8")
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/inverse-csv":
            try:
                query = parse_qs(parsed.query)
                dataset = query.get("dataset", ["dataset1"])[0]
                csv_text = STATE.latest_inverse_csv(dataset)
                self._send(csv_text.encode("utf-8"), 200, "text/csv; charset=utf-8")
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/inverse-xlsx":
            try:
                query = parse_qs(parsed.query)
                dataset = query.get("dataset", ["dataset1"])[0]
                data = STATE.latest_inverse_xlsx(dataset)
                self._send(data, 200, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/inverse-report":
            try:
                query = parse_qs(parsed.query)
                dataset = query.get("dataset", ["dataset1"])[0]
                html = STATE.latest_inverse_report(dataset)
                self._send(html.encode("utf-8"), 200, "text/html; charset=utf-8")
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
            return
        if parsed.path.startswith("/files/"):
            rel = unquote(parsed.path[len("/files/") :])
            path = (ROOT / rel).resolve()
            if not str(path).startswith(str(ROOT.resolve())) or not path.exists() or not path.is_file():
                self._json({"error": "File not found"}, 404)
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send(path.read_bytes(), 200, content_type)
            return
        self._json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if parsed.path == "/api/predict":
                result = STATE.predict(payload.get("dataset"), payload.get("values") or {})
                self._json(result)
                return
            if parsed.path == "/api/optimize":
                result = STATE.optimize(
                    payload.get("dataset", "dataset1"),
                    payload.get("direction", "max"),
                    int(payload.get("particle_size", 24)),
                    int(payload.get("iterations", 30)),
                    int(payload.get("top_k", 5)),
                    payload.get("ranges") or {},
                    payload.get("categories") or {},
                )
                self._json(result)
                return
            if parsed.path == "/api/optimization-job":
                result = STATE.start_optimization_job(payload)
                self._json(result)
                return
            if parsed.path == "/api/inverse-design":
                result = STATE.inverse_design(payload)
                self._json(result)
                return
            if parsed.path == "/api/inverse-design-job":
                result = STATE.start_inverse_design_job(payload)
                self._json(result)
                return
            self._json({"error": "Not found"}, 404)
        except Exception as exc:
            self._json({"error": str(exc), "traceback": traceback.format_exc(limit=4)}, 500)

    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write("%s - %s\n" % (self.address_string(), fmt % args))


def run_self_test() -> None:
    state = PredictorState()
    for name, ds in state.assets["datasets"].items():
        values = ds["examples"][0]["values"]
        result = state.predict(name, values)
        print(name, json.dumps(result, indent=2, ensure_ascii=False)[:1200])


def main() -> None:
    global STATE
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8057")))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    STATE = PredictorState()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"PFAS predictor app running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
