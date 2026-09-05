"""Nested leave-one-participant-out assessment with leakage-free recipe selection.

Protocol (predeclared before any outer-fold results are inspected):
- Outer: leave-one-participant-out over all subjects. The held-out
  participant is touched exactly once, by the final evaluation.
- Inner: within each outer fold, the remaining participants are split into
  3 deterministic participant-grouped folds. Inner out-of-fold predictions
  (models trained only on inner-train participants) select the recipe:
  combine weights + cue threshold.
- Operating-point rule: maximize pooled event sensitivity subject to
  false-cue starts <= FP_BUDGET_PER_HOUR on the inner out-of-fold streams.
  If no candidate threshold meets the budget, take the minimum false-cue
  rate (tie-break: higher sensitivity, then lower mean detection delay).
- The candidate is then refit on ALL outer-training participants and
  evaluated once on the held-out participant with the selected recipe.

Candidates:
- cnn_fi      current 1D-CNN + Freeze Index (weights from WEIGHT_GRID)
- fi_only     Freeze Index alone (no learned model)
- logistic_fi logistic regression on streaming-safe window features + FI

All scalers are fit inside the training split only. Evaluation always uses
the deployment hop (0.25 s decisions) through the shared cue FSM.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    from src.benchmark import DEFAULT_TOL_S, JITTER_TOL_S, aggregate, evaluate_participant, to_jsonable
    from src.data import (
        EVAL_HOP_S,
        TRAIN_HOP_S,
        WINDOW_S,
        RunWindows,
        build_cache,
        file_sha256,
        list_subjects,
        load_participant_cached,
    )
    from src.features import p_fog_combined, window_feature_matrix
except ImportError:  # imported as a top-level module
    from benchmark import DEFAULT_TOL_S, JITTER_TOL_S, aggregate, evaluate_participant, to_jsonable
    from data import (
        EVAL_HOP_S,
        TRAIN_HOP_S,
        WINDOW_S,
        RunWindows,
        build_cache,
        file_sha256,
        list_subjects,
        load_participant_cached,
    )
    from features import p_fog_combined, window_feature_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- predeclared protocol constants ---------------------------------------
HYSTERESIS = 0.15
WEIGHT_GRID = ((0.5, 0.5), (0.6, 0.4))
FI_ONLY_WEIGHTS = ((0.0, 1.0),)
THRESHOLD_GRID = tuple(round(float(v), 2) for v in np.arange(0.30, 0.951, 0.05))
FP_BUDGET_PER_HOUR = 1.5  # engineering budget for operating-point selection (NOT a clinical threshold)
N_INNER_GROUPS = 3
CANDIDATES = ("cnn_fi", "fi_only", "logistic_fi")


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------

class CnnFi:
    """The current 1D-CNN recipe: fixed epochs, class weights, seeded."""

    name = "cnn_fi"

    def __init__(self, epochs: int, batch: int):
        self.epochs, self.batch = epochs, batch
        self.model = None
        self.scaler = None

    def fit(self, bundles: list[RunWindows], seed: int) -> None:
        from src.model import StandardScaler, build_fog_cnn

        X = np.concatenate([b.X_shank for b in bundles if len(b.decision_ms)])
        y = np.concatenate([b.label_any for b in bundles if len(b.decision_ms)]).astype(np.float32)
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        if n_pos == 0:
            raise ValueError("training split has no freeze windows")
        self.scaler = StandardScaler().fit(X)
        self.model = build_fog_cnn(window_length=X.shape[1])
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        self.model.fit(
            self.scaler.transform(X), y,
            epochs=self.epochs, batch_size=self.batch, verbose=0,
            class_weight={0: 1.0, 1: max(1.0, n_neg / (n_pos + 1e-8) * 0.5)},
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.empty(0, np.float32)
        return np.asarray(
            self.model.predict(self.scaler.transform(X), batch_size=512, verbose=0)
        ).ravel().astype(np.float32)

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        self.model.save(d / "model.keras")
        self.scaler.save(d / "scaler.json")


class LogisticFi:
    """Logistic regression on streaming-safe window features (raw mg)."""

    name = "logistic_fi"

    def __init__(self):
        self.mu = self.sd = self.coef = self.intercept = None

    def fit(self, bundles: list[RunWindows], seed: int) -> None:
        from sklearn.linear_model import LogisticRegression

        X = np.concatenate([b.X_shank for b in bundles if len(b.decision_ms)])
        y = np.concatenate([b.label_any for b in bundles if len(b.decision_ms)])
        if y.sum() == 0:
            raise ValueError("training split has no freeze windows")
        F = window_feature_matrix(X)
        self.mu, self.sd = F.mean(axis=0), F.std(axis=0) + 1e-8
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        lr.fit((F - self.mu) / self.sd, y)
        self.coef, self.intercept = lr.coef_.ravel(), float(lr.intercept_[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.empty(0, np.float32)
        F = window_feature_matrix(X)
        z = ((F - self.mu) / self.sd) @ self.coef + self.intercept
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "logistic.npz", mu=self.mu, sd=self.sd, coef=self.coef, intercept=self.intercept)


class FiOnly:
    """Freeze Index alone; the learned-model term is identically zero."""

    name = "fi_only"

    def fit(self, bundles: list[RunWindows], seed: int) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.float32)

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)


def make_candidate(name: str, epochs: int, batch: int):
    if name == "cnn_fi":
        return CnnFi(epochs, batch)
    if name == "logistic_fi":
        return LogisticFi()
    if name == "fi_only":
        return FiOnly()
    raise ValueError(f"unknown candidate {name}")


# ---------------------------------------------------------------------------
# Splits and recipe selection
# ---------------------------------------------------------------------------

def grouped_inner_splits(train_subjects: list[int], n_groups: int = N_INNER_GROUPS):
    """Deterministic participant-grouped inner splits (round-robin grouping).

    Returns [(inner_train, inner_val), ...] covering every training
    participant exactly once as inner-val.
    """
    ss = sorted(train_subjects)
    if len(ss) < n_groups:
        n_groups = max(1, len(ss))
    groups = [ss[i::n_groups] for i in range(n_groups)]
    return [([s for s in ss if s not in g], g) for g in groups if g]


def pooled_oof_metrics(run_evals, threshold: float) -> dict:
    """Pooled counts over all inner-val runs at one threshold."""
    return evaluate_participant(run_evals, threshold, HYSTERESIS, tol_s=DEFAULT_TOL_S)


def select_recipe(oof_evals: list[tuple[RunWindows, np.ndarray]], weight_grid, threshold_grid, budget: float):
    """Select (w_cnn, w_fi, threshold) on inner out-of-fold streams."""
    sweep = []
    for wc, wf in weight_grid:
        scored = [(b, p_fog_combined(p, b.fi, wc, wf)) for b, p in oof_evals]
        for thr in threshold_grid:
            m = pooled_oof_metrics(scored, thr)
            sens = m["event_sensitivity"] if m["event_sensitivity"] is not None else -1.0
            fp = m["false_cue_starts_per_nonfog_hour"]
            fp = fp if fp is not None else float("inf")
            delay = m["detection_delay"]["mean_s"]
            delay = delay if delay is not None else float("inf")
            sweep.append({
                "w_cnn": wc, "w_fi": wf, "threshold": float(thr),
                "event_sensitivity": m["event_sensitivity"],
                "false_cue_starts_per_nonfog_hour": m["false_cue_starts_per_nonfog_hour"],
                "mean_detection_delay_s": delay,
                "n_events": m["n_events"],
                "feasible": bool(fp <= budget),
            })

    feasible = [s for s in sweep if s["feasible"]]
    if feasible:
        best = min(feasible, key=lambda s: (-s["event_sensitivity"], s["false_cue_starts_per_nonfog_hour"], s["mean_detection_delay_s"]))
    else:
        best = min(sweep, key=lambda s: (s["false_cue_starts_per_nonfog_hour"], -s["event_sensitivity"], s["mean_detection_delay_s"]))
    recipe = {
        "w_cnn": best["w_cnn"], "w_fi": best["w_fi"], "threshold": best["threshold"],
        "hysteresis": HYSTERESIS, "fp_budget_per_hour": budget,
        "budget_met": best["feasible"],
        "selected_by": "max pooled event sensitivity s.t. false-cue budget (inner OOF)",
    }
    return recipe, sweep


def slim(bundle: RunWindows) -> RunWindows:
    """Drop the sample array (kept on disk) so selection holds only metadata."""
    return replace(bundle, X_shank=np.empty((0, 0, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# One outer fold
# ---------------------------------------------------------------------------

def run_fold(ctx: dict, candidate_name: str, subject: int) -> dict:
    t0 = time.time()
    fold_dir = Path(ctx["outdir"]) / ctx["tag"] / candidate_name / f"fold_s{subject:02d}"
    result_path = fold_dir / "result.json"
    if result_path.exists() and not ctx["force"]:
        print(f"[fold S{subject:02d}] exists, skipping ({result_path})", flush=True)
        return json.loads(result_path.read_text())

    dataset_dir, cache_dir = ctx["dataset"], ctx["cache_dir"]

    def train_bundles(subjects):
        out = []
        for s in subjects:
            for b in load_participant_cached(dataset_dir, cache_dir, s, TRAIN_HOP_S):
                out.append(b.truncate_for_eval(ctx["max_windows"]) if ctx["max_windows"] else b)
        return out

    def eval_bundles(s):
        out = []
        for b in load_participant_cached(dataset_dir, cache_dir, s, EVAL_HOP_S):
            out.append(b.truncate_for_eval(ctx["max_windows"]) if ctx["max_windows"] else b)
        return out

    train_subjects = [s for s in ctx["subjects"] if s != subject]
    inner_splits = grouped_inner_splits(train_subjects)
    for itr, iva in inner_splits:
        assert subject not in itr and subject not in iva, "leakage guard: outer subject in inner split"

    # 1) inner out-of-fold predictions (models never see their val participants)
    oof_evals: list[tuple[RunWindows, np.ndarray]] = []
    for k, (itr, iva) in enumerate(inner_splits):
        model = make_candidate(candidate_name, ctx["epochs"], ctx["batch"])
        if candidate_name == "fi_only":
            for vs in iva:
                for b in eval_bundles(vs):
                    oof_evals.append((slim(b), model.predict(b.X_shank)))
            continue
        try:
            model.fit(train_bundles(itr), seed=ctx["seed"] + 1000 * subject + 10 * k)
        except ValueError as e:
            print(f"[fold S{subject:02d}] inner split {k} skipped: {e}", flush=True)
            continue
        for vs in iva:
            for b in eval_bundles(vs):
                oof_evals.append((slim(b), model.predict(b.X_shank)))

    weight_grid = FI_ONLY_WEIGHTS if candidate_name == "fi_only" else WEIGHT_GRID
    recipe, sweep = select_recipe(oof_evals, weight_grid, THRESHOLD_GRID, ctx["fp_budget"])
    print(
        f"[fold S{subject:02d}] recipe: w=({recipe['w_cnn']:.1f},{recipe['w_fi']:.1f}) "
        f"thr={recipe['threshold']:.2f} budget_met={recipe['budget_met']}",
        flush=True,
    )

    # 2) refit on all outer-training participants, evaluate once on held-out
    final_model = make_candidate(candidate_name, ctx["epochs"], ctx["batch"])
    if candidate_name == "fi_only":
        final_model.save(fold_dir)
    else:
        final_model.fit(train_bundles(train_subjects), seed=ctx["seed"] + 1000 * subject)
        final_model.save(fold_dir)

    test_evals = []
    pred_save = {}
    for b in eval_bundles(subject):
        p = final_model.predict(b.X_shank)
        score = p_fog_combined(p, b.fi, recipe["w_cnn"], recipe["w_fi"])
        test_evals.append((slim(b), score))
        pred_save[f"S{b.subject:02d}R{b.run:02d}"] = np.stack(
            [b.decision_ms.astype(np.float64), b.fi, p, score,
             b.label_endpoint.astype(np.float64), b.label_any.astype(np.float64)]
        )
    np.savez(fold_dir / "predictions.npz", **pred_save)

    res = evaluate_participant(test_evals, recipe["threshold"], HYSTERESIS, tol_s=DEFAULT_TOL_S)
    jitter = evaluate_participant(test_evals, recipe["threshold"], HYSTERESIS, tol_s=JITTER_TOL_S)
    res["subject"] = subject
    res["candidate"] = candidate_name
    res["recipe"] = recipe
    res["jitter_tol_0.3s"] = {
        k: jitter[k] for k in
        ("event_sensitivity", "n_detected", "n_events", "false_cue_starts_per_nonfog_hour",
         "n_false_cue_starts", "detection_delay")
    }
    res["protocol"] = {
        "window_s": WINDOW_S, "train_hop_s": TRAIN_HOP_S, "eval_hop_s": EVAL_HOP_S,
        "hysteresis": HYSTERESIS, "threshold_grid": list(THRESHOLD_GRID),
        "weight_grid": [list(w) for w in weight_grid],
        "fp_budget_per_hour": ctx["fp_budget"], "epochs": ctx["epochs"], "batch": ctx["batch"],
        "seed": ctx["seed"], "label": "any-freeze-in-window (current recipe)",
    }
    res["elapsed_s"] = round(time.time() - t0, 1)

    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "recipe.json").write_text(json.dumps(to_jsonable(recipe), indent=2))
    (fold_dir / "sweep.json").write_text(json.dumps(to_jsonable(sweep), indent=2))
    result_path.write_text(json.dumps(to_jsonable(res), indent=2))
    print(
        f"[fold S{subject:02d}] events={res['n_events']} detected={res['n_detected']} "
        f"false_starts={res['n_false_cue_starts']} ({res['elapsed_s']}s)",
        flush=True,
    )
    return res


# ---------------------------------------------------------------------------
# Assessment run and summary
# ---------------------------------------------------------------------------

def dataset_fingerprint(dataset_dir: Path) -> dict:
    files = sorted(Path(dataset_dir).glob("S*R*.txt"))
    h = file_sha256(files[0]) if files else ""
    for f in files[1:]:
        h = file_sha256_str(h + file_sha256(f))
    return {"n_files": len(files), "combined_sha256": h}


def file_sha256_str(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()


def summarize(candidate_name: str, fold_results: list[dict], ctx: dict) -> dict:
    tol0 = aggregate(fold_results, n_boot=ctx["n_boot"], seed=ctx["seed"])
    jitter = aggregate(
        [
            {
                "n_events": r["n_events"],
                "n_detected": r["jitter_tol_0.3s"]["n_detected"],
                "n_false_cue_starts": r["jitter_tol_0.3s"]["n_false_cue_starts"],
                "nonfog_hours": r["nonfog_hours"],
                "event_sensitivity": r["jitter_tol_0.3s"]["event_sensitivity"],
                "false_cue_starts_per_nonfog_hour": r["jitter_tol_0.3s"]["false_cue_starts_per_nonfog_hour"],
            }
            for r in fold_results
        ],
        n_boot=ctx["n_boot"], seed=ctx["seed"],
    )

    # pooled window PR-AUC from saved per-fold predictions
    scores, labels = [], []
    for r in fold_results:
        pz = Path(ctx["outdir"]) / ctx["tag"] / candidate_name / f"fold_s{r['subject']:02d}" / "predictions.npz"
        if not pz.exists():
            continue
        z = np.load(pz)
        for key in z.files:
            arr = z[key]  # rows: t_ms, fi, p_model, score, label_endpoint, label_any
            scores.append(arr[3])
            labels.append(arr[4])
    pr_auc = None
    if scores:
        y = np.concatenate(labels) > 0.5
        if len(np.unique(y)) > 1:
            try:
                from sklearn.metrics import average_precision_score

                pr_auc = float(average_precision_score(y, np.concatenate(scores)))
            except Exception:
                pr_auc = None

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tensorflow": None,
        "sklearn": None,
    }
    try:
        import sklearn
        import tensorflow as tf

        versions.update({"tensorflow": tf.__version__, "sklearn": sklearn.__version__})
    except Exception:
        pass

    return {
        "candidate": candidate_name,
        "tol_0s": tol0,
        "jitter_tol_0.3s": jitter,
        "pooled_window_pr_auc_endpoint": pr_auc,
        "folds": [
            {
                "subject": r["subject"],
                "n_events": r["n_events"],
                "n_detected": r["n_detected"],
                "event_sensitivity": r["event_sensitivity"],
                "n_false_cue_starts": r["n_false_cue_starts"],
                "false_cue_starts_per_nonfog_hour": r["false_cue_starts_per_nonfog_hour"],
                "nonfog_hours": r["nonfog_hours"],
                "recipe": r["recipe"],
                "window_f1_endpoint": (r["window_endpoint"] or {}).get("f1"),
                "delay_mean_s": (r["detection_delay"] or {}).get("mean_s"),
            }
            for r in fold_results
        ],
        "protocol": fold_results[0]["protocol"] if fold_results else {},
        "versions": versions,
        "dataset": dataset_fingerprint(Path(ctx["dataset"])),
    }


def main():
    parser = argparse.ArgumentParser(description="Nested LOPO FOG assessment")
    parser.add_argument("--dataset", default="dataset_fog_release/dataset", type=Path)
    parser.add_argument("--cache-dir", default=".cache/windows", type=Path)
    parser.add_argument("--outdir", default="models/nested", type=Path)
    parser.add_argument("--tag", default="baseline_v1")
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--subjects", default="", help="Comma-separated subset (default: all)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp-budget", type=float, default=FP_BUDGET_PER_HOUR)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--max-windows", type=int, default=0, help="Cap windows per run (smoke only)")
    parser.add_argument("--force", action="store_true", help="Recompute existing folds")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)  # relative model paths assume the repo root

    subjects = [int(s) for s in args.subjects.split(",") if s.strip()] or list_subjects(args.dataset)
    print(f"Subjects: {subjects}", flush=True)
    print(f"Building/verifying window cache at {args.cache_dir} ...", flush=True)
    build_cache(args.dataset, args.cache_dir, hops=(EVAL_HOP_S, TRAIN_HOP_S))

    ctx = {
        "dataset": str(args.dataset), "cache_dir": str(args.cache_dir),
        "outdir": str(args.outdir), "tag": args.tag, "subjects": subjects,
        "epochs": args.epochs, "batch": args.batch, "seed": args.seed,
        "fp_budget": args.fp_budget, "n_boot": args.n_boot,
        "max_windows": args.max_windows or None, "force": args.force,
    }

    summaries = {}
    for cand in [c.strip() for c in args.candidates.split(",") if c.strip()]:
        print(f"\n=== Candidate: {cand} ===", flush=True)
        fold_results = []
        for subject in subjects:
            fold_results.append(run_fold(ctx, cand, subject))
        summaries[cand] = summarize(cand, fold_results, ctx)

    out_dir = Path(args.outdir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(to_jsonable(summaries), indent=2))

    print("\n=== Pooled summary (tol 0.0 s) ===", flush=True)
    for cand, s in summaries.items():
        p, b = s["tol_0s"]["pooled"], s["tol_0s"].get("bootstrap") or {}
        sens_ci = b.get("event_sensitivity_ci95")
        fp_ci = b.get("false_cues_per_hour_ci95")
        print(
            f"{cand:>12s}: sensitivity={fmt(p['event_sensitivity'])} "
            f"CI95={fmt_range(sens_ci)} | false-cues/h={fmt(p['false_cue_starts_per_nonfog_hour'])} "
            f"CI95={fmt_range(fp_ci)} | PR-AUC={fmt(s['pooled_window_pr_auc_endpoint'])}",
            flush=True,
        )
    print(f"\nSummary written to {out_dir / 'summary.json'}", flush=True)


def fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def fmt_range(ci):
    return "n/a" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


if __name__ == "__main__":
    main()
