from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from run_pso_optimization import DatasetSearchSpace, Predictor, bounds_for, candidate_key, decode_particle

try:
    import torch
    from botorch.acquisition import qExpectedImprovement
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
    from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import ModelListGP
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.mlls import SumMarginalLogLikelihood

    BOTORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    qExpectedImprovement = None
    qLogExpectedImprovement = None
    qLogExpectedHypervolumeImprovement = None
    qExpectedHypervolumeImprovement = None
    fit_gpytorch_mll = None
    ModelListGP = None
    SingleTaskGP = None
    Standardize = None
    optimize_acqf = None
    NondominatedPartitioning = None
    ExactMarginalLogLikelihood = None
    SumMarginalLogLikelihood = None
    BOTORCH_AVAILABLE = False


def _safe_scale(*values: float, floor: float = 1.0) -> float:
    finite_abs = [abs(float(v)) for v in values if v is not None and np.isfinite(v)]
    if not finite_abs:
        return float(floor)
    return float(max([floor] + finite_abs))


def _target_distance(
    prediction: float,
    *,
    target_mode: str,
    target_value: float | None = None,
    target_min: float | None = None,
    target_max: float | None = None,
    tolerance: float = 0.0,
    tolerance_mode: str = "absolute",
) -> tuple[float, float, bool]:
    pred = float(prediction)
    mode = str(target_mode or "target_value").strip().lower()

    if mode == "target_range":
        low = float(target_min)
        high = float(target_max)
        if low > high:
            low, high = high, low
        width = abs(high - low)
        if width < 1e-12:
            width = _safe_scale(low, high, pred)
        if low <= pred <= high:
            return 0.0, 0.0, True
        error = low - pred if pred < low else pred - high
        return float(error / width), float(error), False

    target = float(target_value)
    tol = max(0.0, float(tolerance or 0.0))
    if str(tolerance_mode or "absolute").lower() == "percent":
        tol = abs(target) * tol / 100.0 if abs(target) > 1e-12 else tol / 100.0
    deviation = abs(pred - target)
    if deviation <= tol:
        return 0.0, float(deviation), True
    scale = _safe_scale(target, pred, tol)
    return float((deviation - tol) / scale), float(deviation), False


def _objective_columns(multi_objective: bool) -> list[str]:
    return ["distance", "lcc_total_usd_m3", "ebi"] if multi_objective else ["distance"]


def _finite_objective_value(row: dict[str, Any], column: str) -> float:
    value = row.get(column)
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return float("inf")


def _objective_cost_vector(row: dict[str, Any], multi_objective: bool) -> np.ndarray:
    columns = _objective_columns(multi_objective)
    values = np.asarray([_finite_objective_value(row, col) for col in columns], dtype=float)
    values[~np.isfinite(values)] = np.finfo(float).max / 1e200
    return values


def _composite_score(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    pieces = []
    for column in columns:
        values = pd.to_numeric(frame.get(column), errors="coerce").astype(float)
        finite = values[np.isfinite(values)]
        if finite.empty:
            pieces.append(pd.Series(1.0, index=frame.index))
            continue
        low = float(finite.min())
        high = float(finite.max())
        span = high - low
        if abs(span) < 1e-12:
            pieces.append(pd.Series(0.0, index=frame.index))
        else:
            pieces.append(((values - low) / span).clip(lower=0, upper=1).fillna(1.0))
    return pd.concat(pieces, axis=1).mean(axis=1)


def _pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=bool)
    finite = np.asarray(values, dtype=float)
    finite[~np.isfinite(finite)] = np.finfo(float).max / 1e200
    n_rows = finite.shape[0]
    mask = np.ones(n_rows, dtype=bool)
    for i in range(n_rows):
        if not mask[i]:
            continue
        dominated_by_other = np.all(finite <= finite[i], axis=1) & np.any(finite < finite[i], axis=1)
        dominated_by_other[i] = False
        if np.any(dominated_by_other):
            mask[i] = False
    return mask


def _rank_archive_frame(archive_df: pd.DataFrame, multi_objective: bool, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if archive_df.empty:
        return archive_df, archive_df
    columns = _objective_columns(multi_objective)
    ranked = archive_df.copy()
    for column in columns:
        ranked[column] = pd.to_numeric(ranked.get(column), errors="coerce")
    ranked["composite_objective"] = _composite_score(ranked, columns)
    if multi_objective:
        values = ranked[columns].to_numpy(dtype=float)
        pareto = _pareto_mask_minimize(values)
        ranked["pareto_front"] = pareto
        ranked["pareto_rank"] = np.where(pareto, 1, 2)
        ranked = ranked.sort_values(
            ["pareto_rank", "composite_objective", "distance", "lcc_total_usd_m3", "ebi", "prediction"],
            ascending=[True, True, True, True, True, False],
        )
    else:
        ranked["pareto_front"] = False
        ranked["pareto_rank"] = 1
        ranked = ranked.sort_values(["distance", "target_error", "prediction"], ascending=[True, True, False])
    top_df = ranked.head(top_k).copy()
    return ranked, top_df


def _sample_positions(
    low: np.ndarray,
    high: np.ndarray,
    n: int,
    rng: np.random.Generator,
    strategy: str,
) -> np.ndarray:
    n = int(max(0, n))
    dim = int(len(low))
    if n <= 0:
        return np.empty((0, dim), dtype=float)

    strategy = str(strategy or "sobol").strip().lower()
    if strategy == "sobol" and dim > 0 and torch is not None:
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
        unit = engine.draw(n).double().cpu().numpy()
    elif strategy == "sobol" and dim > 0:
        unit = _low_discrepancy_unit(n, dim, rng)
    else:
        unit = rng.random((n, dim))
    return low + unit * (high - low)


def _to_unit(positions: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    span = np.where(np.abs(high - low) <= 1e-12, 1.0, high - low)
    return np.clip((positions - low) / span, 0.0, 1.0)


def _from_unit(unit_positions: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return low + np.clip(unit_positions, 0.0, 1.0) * (high - low)


def _first_primes(n: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        limit = int(np.sqrt(candidate)) + 1
        for prime in primes:
            if prime > limit:
                break
            if candidate % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    fraction = 1.0 / float(base)
    while index > 0:
        value += (index % base) * fraction
        index //= base
        fraction /= float(base)
    return value


def _low_discrepancy_unit(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    # CEID exposes "Sobol" as the default low-discrepancy initialization. This
    # local sampler avoids SciPy Sobol warnings seen in the deployed Windows env.
    primes = _first_primes(dim)
    offset = int(rng.integers(1, 10_000))
    shift = rng.random(dim)
    unit = np.zeros((n, dim), dtype=float)
    for row in range(n):
        index = row + offset
        for col, base in enumerate(primes):
            unit[row, col] = (_radical_inverse(index, base) + shift[col]) % 1.0
    return unit


def _evaluate_positions(
    *,
    space: DatasetSearchSpace,
    predictor: Predictor,
    positions: np.ndarray,
    archive: dict[tuple[Any, ...], dict[str, Any]],
    source: str,
    batch: int,
    iteration: int,
    target_config: dict[str, Any],
    candidate_evaluator: Callable[[dict[str, Any], float], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if positions.size == 0:
        return []

    candidates = [decode_particle(space, positions[i]) for i in range(len(positions))]
    predictions = predictor.predict_many(candidates)
    rows: list[dict[str, Any]] = []

    for values, pred in zip(candidates, predictions):
        distance, target_error, meets_target = _target_distance(float(pred), **target_config)
        extra: dict[str, Any] = {}
        if candidate_evaluator is not None:
            extra = candidate_evaluator(values, float(pred)) or {}
        row = {
            "prediction": float(pred),
            "distance": float(distance),
            "target_error": float(target_error),
            "meets_target": bool(meets_target),
            "source": source,
            "batch": int(batch),
            "iteration": int(iteration),
            **values,
            **extra,
        }
        key = candidate_key(space, values)
        existing = archive.get(key)
        if existing is None or tuple(_objective_cost_vector(row, "lcc_total_usd_m3" in row and "ebi" in row)) < tuple(
            _objective_cost_vector(existing, "lcc_total_usd_m3" in existing and "ebi" in existing)
        ):
            archive[key] = row
        rows.append(row)
    return rows


def _fit_botorch_gp(x_unit: np.ndarray, objective: np.ndarray):
    if not BOTORCH_AVAILABLE:
        raise RuntimeError("BoTorch is not installed in this Python environment.")
    if len(x_unit) < 2:
        return None
    train_x = torch.as_tensor(np.asarray(x_unit, dtype=np.float64), dtype=torch.double)
    train_y = torch.as_tensor(np.asarray(objective, dtype=np.float64).reshape(-1, 1), dtype=torch.double)
    if train_x.ndim != 2 or train_y.ndim != 2 or train_x.shape[0] != train_y.shape[0]:
        return None
    if train_x.shape[0] < 2 or float(torch.std(train_y)) < 1e-12:
        return None
    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll, options={"maxiter": 75, "disp": False})
    except TypeError:
        fit_gpytorch_mll(mll)
    return model


def _fit_botorch_multi_gp(x_unit: np.ndarray, utilities: np.ndarray):
    if not BOTORCH_AVAILABLE:
        raise RuntimeError("BoTorch is not installed in this Python environment.")
    if len(x_unit) < 4:
        return None
    train_x = torch.as_tensor(np.asarray(x_unit, dtype=np.float64), dtype=torch.double)
    train_y = torch.as_tensor(np.asarray(utilities, dtype=np.float64), dtype=torch.double)
    if train_x.ndim != 2 or train_y.ndim != 2 or train_x.shape[0] != train_y.shape[0]:
        return None
    if train_y.shape[1] < 2:
        return None
    if train_x.shape[0] < 4:
        return None
    if not torch.isfinite(train_y).all():
        return None
    models = []
    for col in range(train_y.shape[1]):
        y_col = train_y[:, col : col + 1]
        if float(torch.std(y_col)) < 1e-12:
            y_col = y_col + torch.randn_like(y_col) * 1e-9
        models.append(SingleTaskGP(train_x, y_col, outcome_transform=Standardize(m=1)))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll, options={"maxiter": 75, "disp": False})
    except TypeError:
        fit_gpytorch_mll(mll)
    return model


def _propose_botorch_candidates(
    *,
    x_unit: np.ndarray,
    objective: np.ndarray,
    q: int,
    dim: int,
    raw_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    if not BOTORCH_AVAILABLE:
        raise RuntimeError("BoTorch is not installed in this Python environment.")

    model = _fit_botorch_gp(x_unit, objective)
    if model is None:
        return _sample_unit(q, dim, rng, "sobol"), "BoTorch warmup fallback"

    bounds = torch.stack(
        [
            torch.zeros(dim, dtype=torch.double),
            torch.ones(dim, dtype=torch.double),
        ]
    )
    best_f = torch.as_tensor(float(np.max(objective)), dtype=torch.double)
    try:
        acq = qLogExpectedImprovement(model=model, best_f=best_f)
        acq_label = "BoTorch qLogEI"
    except Exception:
        acq = qExpectedImprovement(model=model, best_f=best_f)
        acq_label = "BoTorch qEI"

    raw_samples = int(np.clip(raw_samples, 16, 512))
    num_restarts = int(np.clip(raw_samples // 8, 2, 16))
    try:
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=int(q),
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 4, "maxiter": 80},
        )
        return candidates.detach().cpu().numpy(), acq_label
    except Exception:
        return _sample_unit(q, dim, rng, "sobol"), "BoTorch fallback Sobol"


def _propose_botorch_pareto_candidates(
    *,
    x_unit: np.ndarray,
    utilities: np.ndarray,
    q: int,
    dim: int,
    raw_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    if not BOTORCH_AVAILABLE:
        raise RuntimeError("BoTorch is not installed in this Python environment.")

    model = _fit_botorch_multi_gp(x_unit, utilities)
    if model is None:
        return _sample_unit(q, dim, rng, "sobol"), "BoTorch Pareto warmup fallback"

    train_y = torch.as_tensor(np.asarray(utilities, dtype=np.float64), dtype=torch.double)
    y_min = train_y.min(dim=0).values
    y_max = train_y.max(dim=0).values
    span = torch.clamp(y_max - y_min, min=1e-6)
    ref_point_tensor = y_min - 0.1 * span - 1e-6
    ref_point = ref_point_tensor.tolist()
    try:
        partitioning = NondominatedPartitioning(ref_point=ref_point_tensor, Y=train_y)
        try:
            acq = qLogExpectedHypervolumeImprovement(model=model, ref_point=ref_point, partitioning=partitioning)
            acq_label = "BoTorch qLogEHVI Pareto"
        except Exception:
            acq = qExpectedHypervolumeImprovement(model=model, ref_point=ref_point, partitioning=partitioning)
            acq_label = "BoTorch qEHVI Pareto"
    except Exception:
        scalar = np.asarray(utilities, dtype=float)
        scalar = scalar.mean(axis=1)
        return _propose_botorch_candidates(
            x_unit=x_unit,
            objective=scalar,
            q=q,
            dim=dim,
            raw_samples=raw_samples,
            rng=rng,
        )[0], "BoTorch scalarized Pareto fallback"

    bounds = torch.stack(
        [
            torch.zeros(dim, dtype=torch.double),
            torch.ones(dim, dtype=torch.double),
        ]
    )
    raw_samples = int(np.clip(raw_samples, 16, 512))
    num_restarts = int(np.clip(raw_samples // 8, 2, 16))
    try:
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=int(q),
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 4, "maxiter": 80},
        )
        return candidates.detach().cpu().numpy(), acq_label
    except Exception:
        return _sample_unit(q, dim, rng, "sobol"), "BoTorch qEHVI fallback Sobol"


def _sample_unit(n: int, dim: int, rng: np.random.Generator, strategy: str = "sobol") -> np.ndarray:
    low = np.zeros(dim, dtype=float)
    high = np.ones(dim, dtype=float)
    return _sample_positions(low, high, n, rng, strategy)


def _archive_frame(archive: dict[tuple[Any, ...], dict[str, Any]], limit: int = 2000) -> pd.DataFrame:
    df = pd.DataFrame(list(archive.values()))
    if df.empty:
        return df
    return df.sort_values(["distance", "target_error", "prediction"], ascending=[True, True, False]).head(limit).copy()


def run_inverse_design(
    space: DatasetSearchSpace,
    *,
    target_mode: str,
    target_value: float | None = None,
    target_min: float | None = None,
    target_max: float | None = None,
    tolerance: float = 0.0,
    tolerance_mode: str = "absolute",
    initialization_strategy: str = "sobol",
    initialization_trials: int = 16,
    execution_mode: str = "batch",
    batch_iterations: int = 20,
    batch_size: int = 5,
    raw_samples: int = 64,
    top_k: int = 10,
    inertia: float = 0.72,
    cognitive: float = 1.45,
    social: float = 1.45,
    candidate_evaluator: Callable[[dict[str, Any], float], dict[str, Any]] | None = None,
    multi_objective: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mode = str(target_mode or "target_value").strip().lower()
    if mode not in {"target_value", "target_range"}:
        raise ValueError("target_mode must be target_value or target_range")
    if mode == "target_value" and target_value is None:
        raise ValueError("target_value is required for target value inverse design")
    if mode == "target_range" and (target_min is None or target_max is None):
        raise ValueError("target_min and target_max are required for target range inverse design")

    execution_mode = str(execution_mode or "batch").strip().lower()
    if execution_mode not in {"batch", "sequential"}:
        raise ValueError("execution_mode must be batch or sequential")
    if execution_mode == "sequential":
        batch_size = 1

    initialization_strategy = str(initialization_strategy or "sobol").strip().lower()
    if initialization_strategy not in {"none", "sobol", "uniform"}:
        initialization_strategy = "sobol"

    initialization_trials = int(np.clip(initialization_trials, 0, 256))
    batch_iterations = int(np.clip(batch_iterations, 1, 300))
    batch_size = int(np.clip(batch_size, 1, 50))
    raw_samples = int(np.clip(max(raw_samples, batch_size * 8, 16), 16, 512))
    top_k = int(np.clip(top_k, 1, 100))

    target_config = {
        "target_mode": mode,
        "target_value": target_value,
        "target_min": target_min,
        "target_max": target_max,
        "tolerance": tolerance,
        "tolerance_mode": tolerance_mode,
    }

    predictor = Predictor(space)
    low, high = bounds_for(space)
    dim = len(low)
    rng = np.random.default_rng()
    archive: dict[tuple[Any, ...], dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    x_unit_history: list[np.ndarray] = []
    objective_history: list[np.ndarray] = []
    total_steps = batch_iterations + (1 if initialization_trials > 0 and initialization_strategy != "none" else 0)

    def emit_progress(
        *,
        status: str,
        phase: str,
        message: str,
        completed_steps: int,
        current_batch: int = 0,
        generation_strategy: str = "",
    ) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(
                {
                    "status": status,
                    "phase": phase,
                    "message": message,
                    "progress": float(np.clip(completed_steps / max(total_steps, 1), 0.0, 1.0)),
                    "current_batch": int(current_batch),
                    "total_batches": int(batch_iterations),
                    "generation_strategy": generation_strategy,
                    "history": list(history),
                    "all_candidates": _archive_frame(archive, limit=500).replace({np.nan: None}).to_dict(orient="records"),
                }
            )
        except Exception:
            pass

    def append_training_rows(rows: list[dict[str, Any]], positions: np.ndarray) -> None:
        if not rows:
            return
        unit_positions = _to_unit(positions[: len(rows)], low, high)
        for unit_row, row in zip(unit_positions, rows):
            x_unit_history.append(np.asarray(unit_row, dtype=float))
            # BoTorch acquisition maximizes. All platform objectives are costs,
            # so the modeled utilities are the negative objective values.
            objective_history.append(-_objective_cost_vector(row, multi_objective))

    init_positions = _sample_positions(
        low,
        high,
        initialization_trials if initialization_strategy != "none" else 0,
        rng,
        initialization_strategy,
    )
    emit_progress(
        status="running",
        phase="initialization",
        message=f"Generating and evaluating {len(init_positions)} {initialization_strategy} initialization candidates.",
        completed_steps=0,
        generation_strategy=initialization_strategy,
    )
    init_rows = _evaluate_positions(
        space=space,
        predictor=predictor,
        positions=init_positions,
        archive=archive,
        source=initialization_strategy,
        batch=0,
        iteration=0,
        target_config=target_config,
        candidate_evaluator=candidate_evaluator,
    )
    append_training_rows(init_rows, init_positions)

    if init_rows:
        init_distances = np.asarray([row["distance"] for row in init_rows], dtype=float)
        init_lcc = np.asarray([row.get("lcc_total_usd_m3", np.nan) for row in init_rows], dtype=float)
        init_ebi = np.asarray([row.get("ebi", np.nan) for row in init_rows], dtype=float)
        history.append(
            {
                "batch": 0,
                "phase": "initialization",
                "batch_min_distance": float(np.min(init_distances)),
                "batch_mean_distance": float(np.mean(init_distances)),
                "best_overall_distance": float(min(row["distance"] for row in archive.values())),
                "best_overall_prediction": float(min(archive.values(), key=lambda row: row["distance"]).get("prediction", np.nan)),
                "batch_min_lcc": float(np.nanmin(init_lcc)) if np.isfinite(init_lcc).any() else None,
                "batch_min_ebi": float(np.nanmin(init_ebi)) if np.isfinite(init_ebi).any() else None,
                "n_candidates_evaluated": int(len(archive)),
                "generation_strategy": initialization_strategy,
            }
        )
        emit_progress(
            status="running",
            phase="initialization",
            message="Sobol initialization evaluated. Fitting BoTorch GP next.",
            completed_steps=1,
            generation_strategy=initialization_strategy,
        )

    for iteration in range(1, batch_iterations + 1):
        completed_before_batch = (1 if init_rows else 0) + iteration - 1
        emit_progress(
            status="running",
            phase=execution_mode,
            message=f"Fitting BoTorch GP and optimizing acquisition for batch {iteration}/{batch_iterations}.",
            completed_steps=completed_before_batch,
            current_batch=iteration,
            generation_strategy="BoTorch",
        )
        x_train = np.asarray(x_unit_history, dtype=float) if x_unit_history else np.empty((0, dim), dtype=float)
        y_train = np.asarray(objective_history, dtype=float) if objective_history else np.empty((0, len(_objective_columns(multi_objective))), dtype=float)
        if multi_objective:
            candidate_unit, strategy_label = _propose_botorch_pareto_candidates(
                x_unit=x_train,
                utilities=y_train,
                q=batch_size,
                dim=dim,
                raw_samples=raw_samples,
                rng=rng,
            )
        else:
            candidate_unit, strategy_label = _propose_botorch_candidates(
                x_unit=x_train,
                objective=y_train.reshape(-1),
                q=batch_size,
                dim=dim,
                raw_samples=raw_samples,
                rng=rng,
            )
        positions = _from_unit(candidate_unit, low, high)
        rows = _evaluate_positions(
            space=space,
            predictor=predictor,
            positions=positions,
            archive=archive,
            source=strategy_label,
            batch=iteration,
            iteration=iteration,
            target_config=target_config,
            candidate_evaluator=candidate_evaluator,
        )
        append_training_rows(rows, positions)

        selected = sorted(rows, key=lambda row: tuple(_objective_cost_vector(row, multi_objective)))[:batch_size]
        selected_scores = np.asarray([row["distance"] for row in selected], dtype=float)
        selected_lcc = np.asarray([row.get("lcc_total_usd_m3", np.nan) for row in selected], dtype=float)
        selected_ebi = np.asarray([row.get("ebi", np.nan) for row in selected], dtype=float)
        archive_now = pd.DataFrame(list(archive.values()))
        pareto_count = 0
        if multi_objective and not archive_now.empty:
            objective_values = archive_now[_objective_columns(True)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            pareto_count = int(_pareto_mask_minimize(objective_values).sum())
        history.append(
            {
                "batch": iteration,
                "phase": execution_mode,
                "batch_min_distance": float(np.min(selected_scores)),
                "batch_mean_distance": float(np.mean(selected_scores)),
                "best_overall_distance": float(min(row["distance"] for row in archive.values())),
                "best_overall_prediction": float(min(archive.values(), key=lambda row: row["distance"]).get("prediction", np.nan)),
                "batch_min_lcc": float(np.nanmin(selected_lcc)) if np.isfinite(selected_lcc).any() else None,
                "batch_min_ebi": float(np.nanmin(selected_ebi)) if np.isfinite(selected_ebi).any() else None,
                "best_overall_lcc": float(pd.to_numeric(archive_now.get("lcc_total_usd_m3"), errors="coerce").min()) if multi_objective and "lcc_total_usd_m3" in archive_now else None,
                "best_overall_ebi": float(pd.to_numeric(archive_now.get("ebi"), errors="coerce").min()) if multi_objective and "ebi" in archive_now else None,
                "pareto_front_size": pareto_count,
                "n_candidates_evaluated": int(len(archive)),
                "generation_strategy": strategy_label,
            }
        )
        emit_progress(
            status="running",
            phase=execution_mode,
            message=f"Batch {iteration}/{batch_iterations} evaluated with {strategy_label}.",
            completed_steps=completed_before_batch + 1,
            current_batch=iteration,
            generation_strategy=strategy_label,
        )

    archive_df = pd.DataFrame(list(archive.values()))
    if archive_df.empty:
        top_df = archive_df
        archive_sorted = archive_df
    else:
        archive_sorted, top_df = _rank_archive_frame(archive_df, multi_objective, top_k)
        top_df.insert(0, "rank", range(1, len(top_df) + 1))

    best = top_df.iloc[0].to_dict() if not top_df.empty else {}
    details = {
        "target_mode": mode,
        "target_value": target_value,
        "target_min": target_min,
        "target_max": target_max,
        "tolerance": tolerance,
        "tolerance_mode": tolerance_mode,
        "initialization_strategy": initialization_strategy,
        "initialization_trials": initialization_trials,
        "execution_mode": execution_mode,
        "batch_iterations": batch_iterations,
        "batch_size": batch_size,
        "strategy": "BoTorch Bayesian optimization",
        "acquisition": "qLogExpectedImprovement",
        "raw_samples": raw_samples,
        "objective_mode": "target_lca_lcc_pareto" if multi_objective else "target_distance",
        "objectives": _objective_columns(multi_objective),
        "evaluated_unique_candidates": int(len(archive_df)),
        "history": history,
        "all_candidates": archive_sorted.head(2000).replace({np.nan: None}).to_dict(orient="records"),
        "best_prediction": float(best.get("prediction", np.nan)) if best else None,
        "best_distance": float(best.get("distance", np.nan)) if best else None,
        "best_lcc_total_usd_m3": float(best.get("lcc_total_usd_m3", np.nan)) if best and "lcc_total_usd_m3" in best else None,
        "best_ebi": float(best.get("ebi", np.nan)) if best and "ebi" in best else None,
        "pareto_front_size": int(pd.Series(archive_sorted.get("pareto_front", [])).fillna(False).sum()) if not archive_sorted.empty else 0,
        "best_meets_target": bool(best.get("meets_target", False)) if best else False,
    }
    emit_progress(
        status="complete",
        phase="complete",
        message="Inverse design complete.",
        completed_steps=total_steps,
        current_batch=batch_iterations,
        generation_strategy="BoTorch",
    )
    return top_df, details
