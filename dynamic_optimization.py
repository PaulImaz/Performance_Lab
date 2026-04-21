"""Hybrid global+local optimization for dynamic simulation operating points.

This module is Flask-independent and is designed to be reused by multiple tools.
"""

from __future__ import annotations

import csv
import io
import itertools
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import minimize

    _SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - SciPy is optional at runtime.
    minimize = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False


SUPPORTED_DECISION_VARIABLES: Dict[str, str] = {
    "front_heave": "hf",
    "rear_heave": "hr",
    "front_roll": "rf",
    "rear_roll": "rr",
}


SUPPORTED_OBJECTIVES: Tuple[str, ...] = (
    "total_aero_load_n",
    "front_aero_load_n",
    "rear_aero_load_n",
    "aero_balance_front_pct",
    "drag_force_n",
    "front_total_load_n",
    "rear_total_load_n",
    "total_load_fl_n",
    "total_load_fr_n",
    "total_load_rl_n",
    "total_load_rr_n",
    "front_axle_lateral_force_n",
    "rear_axle_lateral_force_n",
    "total_lateral_force_n",
    "front_axle_longitudinal_force_n",
    "rear_axle_longitudinal_force_n",
    "total_longitudinal_force_n",
)


SUPPORTED_CONSTRAINT_NAMES: Tuple[str, ...] = (
    "aero_balance_front_pct",
    "hRideF",
    "hRideR",
    "total_load_fl_n",
    "total_load_fr_n",
    "total_load_rl_n",
    "total_load_rr_n",
    "front_total_load_n",
    "rear_total_load_n",
    "drag_force_n",
    "total_aero_load_n",
)


@dataclass
class OptimizationVariable:
    name: str
    min_value: float
    max_value: float
    initial_guess: float
    enabled: bool = True


@dataclass
class OptimizationConstraint:
    name: str
    kind: str  # "eq", "ge", "le"
    target: float
    weight: float


@dataclass
class OptimizationProblem:
    objective_mode: str  # "maximize", "minimize", "target"
    objective_name: str
    objective_target: Optional[float]
    variables: List[OptimizationVariable]
    constraints: List[OptimizationConstraint]
    fixed_inputs: Dict[str, Any]
    search_method: str = "auto"  # "auto", "grid_refine", "sample_refine"
    max_global_points: Optional[int] = None
    n_best_candidates: int = 5
    local_maxiter: Optional[int] = None
    local_xtol: Optional[float] = None
    local_ftol: Optional[float] = None


@dataclass
class CandidateEvaluation:
    decision_vector: Dict[str, float]
    objective_value_raw: float
    penalty_value: float
    total_cost: float
    outputs: Dict[str, Any]
    constraint_errors: Dict[str, float]
    is_feasible_like: bool
    constraint_norm_errors: Dict[str, float] = field(default_factory=dict)
    stage: str = "global"


CONSTRAINT_FEAS_NORM_TOL = 1e-2


@dataclass
class OptimizationResult:
    best_candidate: CandidateEvaluation
    top_candidates: List[CandidateEvaluation]
    global_stage_count: int
    refinement_stage_count: int
    search_method_used: str
    diagnostics: Dict[str, Any]
    global_candidates: List[CandidateEvaluation]


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _finite_or_fallback(value: Any, fallback: float = 0.0) -> float:
    numeric = _to_float(value)
    return numeric if np.isfinite(numeric) else float(fallback)


def _clip(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


def active_variables(problem: OptimizationProblem) -> List[OptimizationVariable]:
    vars_enabled = [v for v in problem.variables if v.enabled]
    if not vars_enabled:
        raise ValueError("At least one enabled optimization variable is required.")
    for var in vars_enabled:
        if var.name not in SUPPORTED_DECISION_VARIABLES:
            raise ValueError(f"Unsupported variable '{var.name}'.")
        if var.max_value <= var.min_value:
            raise ValueError(f"Variable '{var.name}' has invalid bounds.")
    return vars_enabled


def merge_decision_vector_with_base_state(
    decision_vector: Dict[str, float],
    base_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    state = {
        "hf": 0.0,
        "hr": 0.0,
        "rf": 0.0,
        "rr": 0.0,
    }
    if isinstance(base_state, dict):
        for key in ("hf", "hr", "rf", "rr"):
            state[key] = _finite_or_fallback(base_state.get(key), state[key])

    for name, value in decision_vector.items():
        mapped = SUPPORTED_DECISION_VARIABLES.get(name)
        if mapped:
            state[mapped] = _finite_or_fallback(value, state[mapped])
    return state


def build_state_from_decision_vector(
    decision_vector: Dict[str, float],
    base_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    return merge_decision_vector_with_base_state(decision_vector, base_state)


def _extract_metric(outputs: Dict[str, Any], metric_name: str) -> float:
    if metric_name not in outputs:
        return float("nan")
    return _to_float(outputs.get(metric_name))


def _objective_cost(
    mode: str,
    raw_value: float,
    target: Optional[float],
) -> float:
    if not np.isfinite(raw_value):
        return 1e12
    mode_norm = (mode or "").strip().lower()
    if mode_norm == "maximize":
        return -float(raw_value)
    if mode_norm == "minimize":
        return float(raw_value)
    if mode_norm == "target":
        if target is None:
            raise ValueError("objective_target is required in 'target' mode.")
        return float(raw_value - float(target)) ** 2
    raise ValueError(f"Unsupported objective_mode '{mode}'.")


def _constraint_penalty(
    outputs: Dict[str, Any],
    constraints: Sequence[OptimizationConstraint],
) -> Tuple[float, Dict[str, float], Dict[str, float], bool]:
    total_penalty = 0.0
    errors: Dict[str, float] = {}
    norm_errors: Dict[str, float] = {}
    feasible_like = True
    for idx, constraint in enumerate(constraints):
        name = constraint.name
        kind = (constraint.kind or "").strip().lower()
        weight = max(0.0, float(constraint.weight))
        target = float(constraint.target)
        value = _extract_metric(outputs, name)
        if not np.isfinite(value):
            err = 1e6
        elif kind == "eq":
            err = float(value - target)
        elif kind == "ge":
            err = float(max(0.0, target - value))
        elif kind == "le":
            err = float(max(0.0, value - target))
        else:
            raise ValueError(f"Unsupported constraint kind '{constraint.kind}'.")

        key = f"{name}__{kind}__{idx}"
        errors[key] = err
        norm_err = float(err / max(1.0, abs(target)))
        norm_errors[key] = norm_err
        total_penalty += weight * (err ** 2)
        if abs(norm_err) > CONSTRAINT_FEAS_NORM_TOL:
            feasible_like = False
    return float(total_penalty), errors, norm_errors, feasible_like


def evaluate_candidate(
    problem: OptimizationProblem,
    decision_vector: Dict[str, float],
    evaluator: Callable[[Dict[str, Any]], Dict[str, Any]],
    stage: str = "global",
) -> CandidateEvaluation:
    base_state = problem.fixed_inputs.get("base_state") if isinstance(problem.fixed_inputs, dict) else {}
    state_4w = build_state_from_decision_vector(decision_vector, base_state)

    sim_inputs = dict(problem.fixed_inputs or {})
    sim_inputs.update(state_4w)
    sim_inputs["state_4w"] = dict(state_4w)
    outputs_raw = evaluator(sim_inputs) or {}
    outputs = dict(outputs_raw)
    outputs["state_4w"] = dict(state_4w)
    outputs["decision_vector"] = dict(decision_vector)

    if problem.objective_name not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"Unsupported objective_name '{problem.objective_name}'.")
    objective_raw = _extract_metric(outputs, problem.objective_name)
    objective_cost = _objective_cost(problem.objective_mode, objective_raw, problem.objective_target)

    for constraint in problem.constraints:
        if constraint.name not in SUPPORTED_CONSTRAINT_NAMES:
            raise ValueError(f"Unsupported constraint '{constraint.name}'.")

    penalty, errors, norm_errors, feasible_like = _constraint_penalty(outputs, problem.constraints)
    total_cost = float(objective_cost + penalty)
    return CandidateEvaluation(
        decision_vector={k: float(v) for k, v in decision_vector.items()},
        objective_value_raw=float(objective_raw) if np.isfinite(objective_raw) else float("nan"),
        penalty_value=float(penalty),
        total_cost=float(total_cost),
        outputs=outputs,
        constraint_errors=errors,
        constraint_norm_errors=norm_errors,
        is_feasible_like=bool(feasible_like),
        stage=stage,
    )


def _candidate_rank_key(candidate: CandidateEvaluation) -> Tuple[int, float, float]:
    if not candidate.constraint_norm_errors:
        return (0, 0.0, float(candidate.total_cost))
    norms = [abs(float(v)) for v in candidate.constraint_norm_errors.values()]
    max_norm = max(norms) if norms else 0.0
    l2 = float(math.sqrt(sum(v * v for v in norms)))
    infeasible = 1 if max_norm > CONSTRAINT_FEAS_NORM_TOL else 0
    return (infeasible, l2, float(candidate.total_cost))


def _method_for_problem(problem: OptimizationProblem, n_vars: int) -> str:
    requested = (problem.search_method or "auto").strip().lower()
    if requested == "grid_refine":
        return "grid_refine"
    if requested == "sample_refine":
        return "sample_refine"
    if requested != "auto":
        raise ValueError(f"Unsupported search_method '{problem.search_method}'.")
    return "grid_refine" if n_vars <= 2 else "sample_refine"


def _grid_candidates(
    variables: Sequence[OptimizationVariable],
    max_global_points: Optional[int],
) -> List[Dict[str, float]]:
    n = len(variables)
    target_total = int(max_global_points) if (max_global_points and max_global_points > 0) else (81 if n == 1 else 121)
    per_dim = max(3, int(round(target_total ** (1.0 / max(1, n)))))
    per_dim = min(per_dim, 51)
    axes = [np.linspace(v.min_value, v.max_value, per_dim) for v in variables]
    candidates: List[Dict[str, float]] = []
    for point in itertools.product(*axes):
        candidates.append({variables[i].name: float(point[i]) for i in range(n)})
    if max_global_points and len(candidates) > max_global_points:
        # Deterministic thinning.
        step = max(1, int(math.ceil(len(candidates) / max_global_points)))
        candidates = candidates[::step][: max_global_points]
    return candidates


def _lhs_candidates(
    variables: Sequence[OptimizationVariable],
    max_global_points: Optional[int],
    seed: int = 42,
) -> List[Dict[str, float]]:
    n = len(variables)
    count = int(max_global_points) if (max_global_points and max_global_points > 0) else max(120, 40 * n)
    rng = np.random.default_rng(seed)
    samples = np.empty((count, n), dtype=float)
    for i in range(n):
        perm = rng.permutation(count)
        bins = (perm + rng.random(count)) / count
        lo = variables[i].min_value
        hi = variables[i].max_value
        samples[:, i] = lo + bins * (hi - lo)
    candidates: List[Dict[str, float]] = []
    for row in samples:
        candidates.append({variables[i].name: float(row[i]) for i in range(n)})
    return candidates


def generate_global_candidates(problem: OptimizationProblem, variables: Sequence[OptimizationVariable]) -> Tuple[str, List[Dict[str, float]]]:
    method = _method_for_problem(problem, len(variables))
    if method == "grid_refine":
        return method, _grid_candidates(variables, problem.max_global_points)
    return method, _lhs_candidates(variables, problem.max_global_points)


def _vector_from_decision(decision: Dict[str, float], variables: Sequence[OptimizationVariable]) -> np.ndarray:
    return np.array([_finite_or_fallback(decision.get(v.name), v.initial_guess) for v in variables], dtype=float)


def _decision_from_vector(x: Sequence[float], variables: Sequence[OptimizationVariable]) -> Dict[str, float]:
    return {variables[i].name: float(x[i]) for i in range(len(variables))}


def _bound_penalty(x: Sequence[float], variables: Sequence[OptimizationVariable], weight: float = 1e9) -> float:
    penalty = 0.0
    for i, var in enumerate(variables):
        xi = float(x[i])
        if xi < var.min_value:
            penalty += weight * (var.min_value - xi) ** 2
        elif xi > var.max_value:
            penalty += weight * (xi - var.max_value) ** 2
    return float(penalty)


def _local_refine_with_powell(
    problem: OptimizationProblem,
    start: CandidateEvaluation,
    variables: Sequence[OptimizationVariable],
    evaluator: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Tuple[CandidateEvaluation, Dict[str, Any]]:
    x0 = _vector_from_decision(start.decision_vector, variables)
    bounds = [(v.min_value, v.max_value) for v in variables]

    def cost_fn(x: np.ndarray) -> float:
        clipped = np.array([_clip(float(x[i]), bounds[i][0], bounds[i][1]) for i in range(len(bounds))], dtype=float)
        decision = _decision_from_vector(clipped, variables)
        cand = evaluate_candidate(problem, decision, evaluator, stage="refinement")
        return float(cand.total_cost + _bound_penalty(x, variables))

    if _SCIPY_AVAILABLE and minimize is not None:
        local_maxiter = int(start.outputs.get("_opt_local_maxiter", problem.local_maxiter or 220))
        local_xtol = float(start.outputs.get("_opt_local_xtol", problem.local_xtol or 1e-4))
        local_ftol = float(start.outputs.get("_opt_local_ftol", problem.local_ftol or 1e-4))
        result = minimize(
            fun=cost_fn,
            x0=x0,
            method="Powell",
            bounds=bounds,
            options={"maxiter": max(8, local_maxiter), "xtol": max(1e-7, local_xtol), "ftol": max(1e-7, local_ftol), "disp": False},
        )
        x_best = np.array(result.x, dtype=float)
        clipped = np.array([_clip(float(x_best[i]), bounds[i][0], bounds[i][1]) for i in range(len(bounds))], dtype=float)
        best = evaluate_candidate(problem, _decision_from_vector(clipped, variables), evaluator, stage="refinement")
        info = {
            "method": "Powell",
            "scipy_available": True,
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(getattr(result, "nfev", 0)),
            "nit": int(getattr(result, "nit", 0)),
            "start_point": start.decision_vector,
            "refined_point": best.decision_vector,
            "start_cost": float(start.total_cost),
            "refined_cost": float(best.total_cost),
        }
        return best, info

    # Deterministic fallback when SciPy is not available: coordinate pattern search.
    x = np.array(x0, dtype=float)
    spans = np.array([max(v.max_value - v.min_value, 1e-9) for v in variables], dtype=float)
    step = 0.2 * spans
    best = evaluate_candidate(problem, _decision_from_vector(x, variables), evaluator, stage="refinement")
    evals = 1
    for _ in range(80):
        improved = False
        for i, var in enumerate(variables):
            for sign in (-1.0, 1.0):
                x_try = np.array(x, dtype=float)
                x_try[i] = _clip(x_try[i] + sign * step[i], var.min_value, var.max_value)
                cand = evaluate_candidate(problem, _decision_from_vector(x_try, variables), evaluator, stage="refinement")
                evals += 1
                if cand.total_cost + 1e-12 < best.total_cost:
                    best = cand
                    x = x_try
                    improved = True
        step *= 0.7
        if not improved and float(np.max(step)) < 1e-4:
            break
    info = {
        "method": "coordinate_pattern_fallback",
        "scipy_available": False,
        "success": True,
        "message": "SciPy not available. Used deterministic coordinate-pattern fallback.",
        "nfev": evals,
        "nit": None,
        "start_point": start.decision_vector,
        "refined_point": best.decision_vector,
        "start_cost": float(start.total_cost),
        "refined_cost": float(best.total_cost),
    }
    return best, info


def run_optimization(
    problem: OptimizationProblem,
    evaluator: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> OptimizationResult:
    vars_enabled = active_variables(problem)
    for var in vars_enabled:
        var.initial_guess = _clip(var.initial_guess, var.min_value, var.max_value)

    search_used, candidate_vectors = generate_global_candidates(problem, vars_enabled)
    if not candidate_vectors:
        raise ValueError("No global candidates were generated.")

    global_candidates = [evaluate_candidate(problem, dv, evaluator, stage="global") for dv in candidate_vectors]
    global_candidates_sorted = sorted(global_candidates, key=_candidate_rank_key)
    n_best = max(1, min(int(problem.n_best_candidates or 1), len(global_candidates_sorted)))
    seeds = global_candidates_sorted[:n_best]

    refinement_infos: List[Dict[str, Any]] = []
    refined_candidates: List[CandidateEvaluation] = []
    for seed in seeds:
        refined, info = _local_refine_with_powell(problem, seed, vars_enabled, evaluator)
        refined_candidates.append(refined)
        refinement_infos.append(info)

    all_ranked = sorted(global_candidates_sorted + refined_candidates, key=_candidate_rank_key)
    best = all_ranked[0]
    top = all_ranked[: max(3, n_best)]

    diagnostics = {
        "objective_mode": problem.objective_mode,
        "objective_name": problem.objective_name,
        "objective_target": problem.objective_target,
        "active_variables": [asdict(v) for v in vars_enabled],
        "constraints": [asdict(c) for c in problem.constraints],
        "global_best_cost": float(global_candidates_sorted[0].total_cost),
        "global_best_violation_norm_l2": float(_candidate_rank_key(global_candidates_sorted[0])[1]),
        "refinement_infos": refinement_infos,
        "global_top_costs": [float(c.total_cost) for c in global_candidates_sorted[: min(10, len(global_candidates_sorted))]],
        "best_constraint_errors": dict(best.constraint_errors),
        "best_constraint_norm_errors": dict(best.constraint_norm_errors),
        "best_is_feasible_like": bool(best.is_feasible_like),
        "scipy_available": _SCIPY_AVAILABLE,
    }

    return OptimizationResult(
        best_candidate=best,
        top_candidates=top,
        global_stage_count=len(global_candidates),
        refinement_stage_count=len(refined_candidates),
        search_method_used=search_used,
        diagnostics=diagnostics,
        global_candidates=global_candidates_sorted,
    )


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def candidates_to_csv(
    candidates: Sequence[CandidateEvaluation],
    decision_columns: Optional[Iterable[str]] = None,
    output_columns: Optional[Iterable[str]] = None,
) -> str:
    if not candidates:
        return ""
    decision_cols = list(decision_columns or sorted({k for c in candidates for k in c.decision_vector.keys()}))
    output_cols = list(output_columns or [])
    if not output_cols:
        default_keys = [
            "total_aero_load_n",
            "aero_balance_front_pct",
            "drag_force_n",
            "front_total_load_n",
            "rear_total_load_n",
            "total_lateral_force_n",
            "total_longitudinal_force_n",
            "hRideF",
            "hRideR",
        ]
        output_cols = [k for k in default_keys if any(k in c.outputs for c in candidates)]

    constraint_cols = sorted({k for c in candidates for k in c.constraint_errors.keys()})
    headers = [
        "rank",
        "stage",
        "objective_value_raw",
        "penalty_value",
        "total_cost",
        "is_feasible_like",
    ] + [f"decision__{c}" for c in decision_cols] + [f"output__{c}" for c in output_cols] + [f"constraint_error__{c}" for c in constraint_cols]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    for idx, cand in enumerate(candidates, start=1):
        row: List[Any] = [
            idx,
            cand.stage,
            cand.objective_value_raw,
            cand.penalty_value,
            cand.total_cost,
            cand.is_feasible_like,
        ]
        row.extend([cand.decision_vector.get(c) for c in decision_cols])
        row.extend([cand.outputs.get(c) for c in output_cols])
        row.extend([cand.constraint_errors.get(c) for c in constraint_cols])
        writer.writerow([json_safe(v) for v in row])
    return buf.getvalue()
