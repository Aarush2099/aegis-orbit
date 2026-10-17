"""Reproducible Step 2C baselines, SAC training, and evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from aegis_orbit.agents import ReplayBuffer, SACAgent
from aegis_orbit.envs import ContinuousConjunctionEnv
from .baselines import passive_policy, random_policy
from .oracle_reference import (
    evaluate_encounter,
    get,
    load_config,
    summarize_oracle_avoidable_subset,
)

_RUN_STARTED_AT = time.monotonic()
SUMMARY_FIELDS = (
    "method",
    "collision_rate",
    "mean_min_distance_m",
    "mean_fuel_mps",
    "bootstrap_95_ci",
)
ORACLE_SUMMARY_FIELDS = (
    "method",
    "label",
    "passive_collision_count",
    "avoidable_count",
    "avoidable_fraction",
    "mean_best_min_distance_m",
)


def progress(message: str) -> None:
    """Print one flushed run-progress line with elapsed seconds."""
    print(f"[{time.monotonic() - _RUN_STARTED_AT:.1f}s] {message}", flush=True)


def config_hash(config: dict) -> str:
    """Hash the parsed config dictionary rather than the raw YAML text."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def make_environment(config: dict[str, float], seed: int) -> ContinuousConjunctionEnv:
    """Create the unshielded CW environment with the calibrated band."""
    low = float(get(config, "environment", "initial_offset_m", "low"))
    high = float(get(config, "environment", "initial_offset_m", "high"))
    midpoint = 0.5 * (low + high)
    return ContinuousConjunctionEnv(
        n=float(get(config, "environment", "n")),
        dt=float(get(config, "environment", "dt")),
        horizon_steps=int(get(config, "environment", "horizon_steps")),
        u_max=float(get(config, "environment", "u_max")),
        d_safe=float(get(config, "safety_geometry", "d_safe_legacy_unused_m")),
        initial_offset_m=midpoint,
        closing_speed_mps=float(get(config, "environment", "closing_speed_mps")),
        initial_offset_range_m=(low, high),
    )


def make_fixed_environment(
    config: dict[str, float],
    seed: int,
    offset_m: float,
) -> ContinuousConjunctionEnv:
    """Create an unshielded CW environment with no randomized offset band."""
    return ContinuousConjunctionEnv(
        n=float(get(config, "environment", "n")),
        dt=float(get(config, "environment", "dt")),
        horizon_steps=int(get(config, "environment", "horizon_steps")),
        u_max=float(get(config, "environment", "u_max")),
        d_safe=float(get(config, "safety_geometry", "d_safe_legacy_unused_m")),
        initial_offset_m=offset_m,
        closing_speed_mps=float(get(config, "environment", "closing_speed_mps")),
        initial_offset_range_m=None,
    )


def oracle_preflight(config: dict) -> dict[str, float | int]:
    """Validate the calibrated passive/oracle boundary before any benchmark run."""
    checked_config = deepcopy(config)
    offset = float(get(checked_config, "oracle_preflight", "offset_m"))
    checked_config["environment"]["initial_offset_m"]["low"] = offset
    checked_config["environment"]["initial_offset_m"]["high"] = offset

    hard_body_radius = float(get(checked_config, "safety_geometry", "hard_body_radius_m"))
    expected_collisions = int(get(checked_config, "oracle_preflight", "passive_collision_count"))
    min_allowed_margin = float(get(checked_config, "oracle_preflight", "min_margin_m"))
    seed_start = int(get(checked_config, "oracle_preflight", "evaluation_seeds", "start"))
    seed_count = int(get(checked_config, "oracle_preflight", "evaluation_seeds", "count"))
    passive_collision_seeds = []
    for seed in range(seed_start, seed_start + seed_count):
        environment = make_fixed_environment(checked_config, seed, offset)
        environment.reset(np.random.default_rng(seed))
        collided = False
        for _ in range(environment.horizon_steps):
            _, info = environment.step(np.zeros(3, dtype=np.float64))
            if float(info["min_distance_in_step"]) < hard_body_radius:
                collided = True
                break
        if collided:
            passive_collision_seeds.append(seed)

    oracle_avoidable_count = 0
    minimum_margin = float("inf")
    for seed in passive_collision_seeds:
        avoidable, best_distance = evaluate_encounter(seed, checked_config, hard_body_radius)
        oracle_avoidable_count += int(avoidable)
        minimum_margin = min(minimum_margin, best_distance - hard_body_radius)

    observed = {
        "passive_collision_count": len(passive_collision_seeds),
        "oracle_avoidable_count": oracle_avoidable_count,
        "minimum_margin_m": minimum_margin,
    }
    if len(passive_collision_seeds) != expected_collisions or minimum_margin < min_allowed_margin:
        raise RuntimeError(
            "Oracle pre-flight failed: "
            f"observed_passive_collisions={len(passive_collision_seeds)} "
            f"expected_passive_collisions={expected_collisions}; "
            f"oracle_avoidable_count={oracle_avoidable_count}; "
            f"minimum_margin_m={minimum_margin:.6g} "
            f"required_minimum_margin_m={min_allowed_margin:.6g}"
        )
    return observed


def reward_components(
    action: np.ndarray,
    info: dict[str, float],
    config: dict,
    *,
    prior_collided: bool,
    at_horizon: bool,
    proximity_weight: float,
    clipped_action: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute each reward component separately and enforce one collision penalty."""
    hard = float(get(config, "safety_geometry", "hard_body_radius_m"))
    reference = float(get(config, "safety_geometry", "proximity_reference_m"))
    fuel_weight = float(get(config, "reward", "fuel_weight"))
    collision_penalty = float(get(config, "reward", "collision_penalty"))
    survival_bonus = float(get(config, "reward", "survival_bonus"))
    clipped = np.asarray(action if clipped_action is None else clipped_action, dtype=np.float64)
    fuel_term = -fuel_weight * float(np.linalg.norm(clipped))
    collision_this_step = bool(float(info["min_distance_in_step"]) < hard)
    collision_term = -collision_penalty if collision_this_step and not prior_collided else 0.0
    proximity_term = -proximity_weight * max(0.0, reference - float(info["distance"]))
    survival_term = survival_bonus * float((at_horizon and not prior_collided and not collision_this_step))
    reward = fuel_term + collision_term + proximity_term + survival_term
    return {
        "fuel_term": fuel_term,
        "collision_term": collision_term,
        "proximity_term": proximity_term,
        "survival_term": survival_term,
        "reward": reward,
    }


def step_reward_and_termination(
    action: np.ndarray,
    info: dict[str, float],
    config: dict,
    *,
    prior_collided: bool,
    at_horizon: bool,
    proximity_weight: float,
    clipped_action: np.ndarray,
) -> tuple[dict[str, float], bool, bool]:
    """Return reward components, terminal status, and cumulative collision status."""
    components = reward_components(
        action,
        info,
        config,
        prior_collided=prior_collided,
        at_horizon=at_horizon,
        proximity_weight=proximity_weight,
        clipped_action=clipped_action,
    )
    collision_this_step = float(info["min_distance_in_step"]) < float(
        get(config, "safety_geometry", "hard_body_radius_m")
    )
    collided = prior_collided or collision_this_step
    return components, collided or at_horizon, collided


def run_episode(
    environment,
    policy,
    rng: np.random.Generator,
    hard_body_radius_m: float,
    config: dict,
    proximity_weight: float = 0.0,
):
    """Run one episode and return physical metrics plus reward components."""
    state = environment.reset(rng)
    minimum = np.inf
    fuel = 0.0
    clipped = 0
    collided = False
    final_distance = float("inf")
    components = {"fuel_term": 0.0, "collision_term": 0.0, "proximity_term": 0.0, "survival_term": 0.0}
    for step in range(1, environment.horizon_steps + 1):
        action = np.asarray(policy(state, rng), dtype=np.float64)
        clipped_action = np.clip(action, -environment.u_max, environment.u_max)
        state, info = environment.step(action)
        minimum = min(minimum, float(info["min_distance_in_step"]))
        fuel += float(np.linalg.norm(clipped_action) * environment.dt)
        clipped += int(bool(info["clipped"]))
        current, done, collided = step_reward_and_termination(
            action,
            info,
            config,
            prior_collided=collided,
            at_horizon=step == environment.horizon_steps,
            proximity_weight=proximity_weight,
            clipped_action=clipped_action,
        )
        for key in components:
            components[key] += current[key]
        final_distance = float(info["distance"])
        if done:
            return {
                "collided": collided,
                "min_distance_m": float(minimum),
                "final_distance_m": final_distance,
                "fuel_mps": fuel,
                "clipped_fraction": clipped / environment.horizon_steps,
                "episode_length": step,
                **components,
                "reward": sum(components.values()),
            }
    raise RuntimeError("episode loop ended without a terminal condition")


def train(
    seed: int,
    config: dict[str, float],
    proximity_weight: float,
    logs: list[dict] | None = None,
    stability_logs: list[dict] | None = None,
    reward_logs: list[dict] | None = None,
    selection_logs: list[dict] | None = None,
    selection_path: Path | None = None,
    policy_output_dir: Path | None = None,
    method_name: str | None = None,
    episode_diagnostic_logs: list[dict] | None = None,
):
    """Train one seed, restore the best validation checkpoint, and return agent + selection metadata."""
    if logs is None:
        logs = []
    if stability_logs is None:
        stability_logs = []
    if reward_logs is None:
        reward_logs = []
    if selection_logs is None:
        selection_logs = []
    if method_name is None:
        method_name = _method_name(config, proximity_weight)

    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = SACAgent(
        state_dim=12,
        action_dim=3,
        u_max=float(get(config, "environment", "u_max")),
        learning_rate=float(get(config, "training", "actor_lr")),
        gamma=float(get(config, "training", "gamma")),
        tau=float(get(config, "training", "tau")),
        target_entropy=float(get(config, "training", "target_entropy")),
    )
    buffer = ReplayBuffer(
        capacity=int(get(config, "training", "replay_capacity")),
        rng=np.random.default_rng(seed + 1000),
    )
    env = make_environment(config, seed)
    rng = np.random.default_rng(seed)
    state = env.reset(rng)
    episode_initial_offset_m = float(state[0])
    total_steps = int(get(config, "training", "total_env_steps"))
    warmup = int(get(config, "training", "warmup_steps"))
    batch_size = int(get(config, "training", "batch_size"))
    interval = int(get(config, "training", "validation_interval_steps"))
    hard = float(get(config, "safety_geometry", "hard_body_radius_m"))
    started_at = time.monotonic()

    best_score = (np.inf, np.inf)
    best_step = 0
    best_actor_state = None
    selection_rows: list[dict] = []
    episode_components = {"fuel_term": 0.0, "collision_term": 0.0, "proximity_term": 0.0, "survival_term": 0.0}
    episode_steps = 0
    episode_collided = False
    episode_collision_control_step = None
    episode_collision_substep_index = None
    episode_collision_substep_time_s = None
    episode_started_at = time.monotonic()

    for step in range(1, total_steps + 1):
        if step <= warmup:
            action = rng.uniform(-env.u_max, env.u_max, 3)
        else:
            action = agent.action(state)
        clipped_action = np.clip(action, -env.u_max, env.u_max)
        if episode_diagnostic_logs is None:
            next_state, info = env.step(action)
        else:
            next_state, info = env.step(action, collision_radius_m=hard)
        prior_episode_collided = episode_collided
        current, done, episode_collided = step_reward_and_termination(
            action,
            info,
            config,
            prior_collided=episode_collided,
            at_horizon=env.done,
            proximity_weight=proximity_weight,
            clipped_action=clipped_action,
        )
        for key in episode_components:
            episode_components[key] += current[key]
        episode_steps += 1
        if episode_diagnostic_logs is not None and not prior_episode_collided and episode_collided:
            substep_index = info["first_collision_substep_index"]
            substep_time_s = info["first_collision_substep_time_s"]
            if substep_index is None or substep_time_s is None:
                raise RuntimeError(
                    "Collision was reported without a sampled sub-step crossing"
                )
            episode_collision_control_step = episode_steps
            episode_collision_substep_index = substep_index
            episode_collision_substep_time_s = substep_time_s
        buffer.add(state, action, current["reward"], next_state, float(done))
        if done:
            reward_logs.append({
                "seed": seed,
                "step": step,
                "method": "sac_primary" if proximity_weight == 0.0 else f"sac_ablation_proximity_{proximity_weight}",
                "episode_length": episode_steps,
                "collided": int(episode_collided),
                **episode_components,
                "reward": sum(episode_components.values()),
            })
            if episode_diagnostic_logs is not None:
                episode_elapsed_s = time.monotonic() - episode_started_at
                episode_diagnostic_logs.append({
                    "seed": seed,
                    "step": step,
                    "episode_length": episode_steps,
                    "collided": int(episode_collided),
                    "initial_offset_m": episode_initial_offset_m,
                    "collision_control_step": episode_collision_control_step,
                    "collision_substep_index_zero_based": episode_collision_substep_index,
                    "collision_substep_time_s": episode_collision_substep_time_s,
                    "episode_to_terminal_s": episode_elapsed_s,
                })
                progress(
                    f"seed={seed} episode_end step={step} "
                    f"episode_length={episode_steps} collided={int(episode_collided)} "
                    f"initial_offset_m={episode_initial_offset_m:.9f} "
                    f"episode_to_terminal_s={episode_elapsed_s:.3f}"
                )
            episode_components = {"fuel_term": 0.0, "collision_term": 0.0, "proximity_term": 0.0, "survival_term": 0.0}
            episode_steps = 0
            episode_collided = False
            episode_collision_control_step = None
            episode_collision_substep_index = None
            episode_collision_substep_time_s = None
        if done:
            state = env.reset(rng)
            episode_initial_offset_m = float(state[0])
            episode_started_at = time.monotonic()
        else:
            state = next_state

        if len(buffer) >= batch_size:
            metrics = agent.update(buffer.sample(batch_size))
            reported_nan = int(metrics.get("nan", 0))
            metrics.update({
                "seed": seed,
                "step": step,
                "proximity_weight": proximity_weight,
                "nan": reported_nan,
            })
            if not np.isfinite([float(value) for value in metrics.values() if isinstance(value, (int, float, np.floating))]).all():
                metrics["nan"] = 1
                raise ValueError(f"NaN detected in training update for seed {seed} step {step}")
            if reported_nan:
                raise ValueError(f"NaN detected in training update for seed {seed} step {step}")
            stability_logs.append(
                {"seed": seed, "step": step, "actor_loss": metrics["actor_loss"], "critic_loss": metrics["critic_loss"], "q_min": metrics["q_min"], "q_max": metrics["q_max"], "alpha": metrics["alpha"], "entropy": metrics["entropy"], "nan": metrics["nan"]}
            )
            logs.append(metrics)

        if step % interval == 0:
            validation_rows = []
            for validation_seed in range(int(get(config, "validation_seeds", "start")), int(get(config, "validation_seeds", "start")) + int(get(config, "validation_seeds", "count"))):
                validation_rows.append(run_episode(make_environment(config, validation_seed), lambda s, r, a=agent: a.action(s, deterministic=True), np.random.default_rng(validation_seed), hard, config, proximity_weight))
            collision_rate = float(np.mean([row["collided"] for row in validation_rows]))
            mean_fuel = float(np.mean([row["fuel_mps"] for row in validation_rows]))
            selected = (collision_rate, mean_fuel) < best_score
            best_score = (collision_rate, mean_fuel) if selected else best_score
            if selected:
                best_step = step
                best_actor_state = deepcopy(agent.actor.state_dict())
                if policy_output_dir is not None:
                    _save_selected_policy(
                        actor=agent.actor,
                        policy_dir=policy_output_dir,
                        method=method_name,
                        train_seed=seed,
                        step=step,
                        configuration_hash=config_hash(config),
                        validation_collision_rate=collision_rate,
                        validation_mean_fuel=mean_fuel,
                    )
            selection_row = {
                "seed": seed,
                "proximity_weight": proximity_weight,
                "step": step,
                "collision_rate": collision_rate,
                "mean_fuel": mean_fuel,
                "selected": bool(selected),
            }
            selection_rows.append(selection_row)
            selection_logs.append(selection_row)
            if selection_path is not None:
                selection_path.parent.mkdir(parents=True, exist_ok=True)
                with selection_path.open("w", newline="", encoding="utf-8") as handle:
                    fields = ["seed", "proximity_weight", "step", "collision_rate", "mean_fuel", "selected"]
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(selection_logs)
            progress(
                f"validation seed={seed} step={step}/{total_steps} "
                f"elapsed_s={time.monotonic() - started_at:.1f} "
                f"collision_rate={collision_rate:.4f} mean_fuel={mean_fuel:.6g} "
                f"selected={int(selected)} best_step={best_step} "
                f"best_collision_rate={best_score[0]:.4f} best_mean_fuel={best_score[1]:.6g}"
            )

    if best_actor_state is not None:
        agent.actor.load_state_dict(best_actor_state)
    return agent, best_step, best_score


def bootstrap(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Return a percentile bootstrap interval over encounters."""
    if len(values) == 0:
        return 0.0, 0.0
    means = [float(np.mean(rng.choice(values, len(values), replace=True))) for _ in range(1000)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _csv_content(fields: tuple[str, ...], rows: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_summary_files(
    output_dir: Path,
    benchmark_rows: list[dict],
    oracle_summary: dict[str, float | int],
    oracle_label: str,
) -> tuple[str, str]:
    """Write the five-column metrics table and a separate oracle summary CSV."""
    methods = sorted({
        row["method"]
        for row in benchmark_rows
        if row["method"] != "oracle_avoidable_subset"
    })
    summary_rows = []
    for method in methods:
        method_rows = [row for row in benchmark_rows if row["method"] == method]
        values = np.asarray([float(row["collided"]) for row in method_rows], dtype=np.float64)
        minima = np.asarray([float(row["min_distance_m"]) for row in method_rows], dtype=np.float64)
        fuel = np.asarray([float(row["fuel_mps"]) for row in method_rows], dtype=np.float64)
        confidence_interval = bootstrap(values, np.random.default_rng(1))
        summary_rows.append({
            "method": method,
            "collision_rate": f"{values.mean():.4f}",
            "mean_min_distance_m": f"{minima.mean():.4f}",
            "mean_fuel_mps": f"{fuel.mean():.6g}",
            "bootstrap_95_ci": str(confidence_interval),
        })
    oracle_rows = [{
        "method": "oracle_avoidable_subset",
        "label": oracle_label,
        "passive_collision_count": oracle_summary["passive_collision_count"],
        "avoidable_count": oracle_summary["oracle_avoidable_count"],
        "avoidable_fraction": f"{oracle_summary['avoidable_fraction']:.4f}",
        "mean_best_min_distance_m": f"{oracle_summary['mean_best_min_distance_m']:.6g}",
    }]
    summary_content = _csv_content(SUMMARY_FIELDS, summary_rows)
    oracle_content = _csv_content(ORACLE_SUMMARY_FIELDS, oracle_rows)
    (output_dir / "summary.csv").write_text(summary_content, encoding="utf-8", newline="")
    (output_dir / "oracle_summary.csv").write_text(oracle_content, encoding="utf-8", newline="")
    return summary_content, oracle_content


def _write_manifest(
    config: dict,
    output_dir: Path,
    command: str,
    methods: list[str],
) -> None:
    """Write the run manifest with version and git metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_commit = "no git"
    manifest = {
        "config_hash": config_hash(config),
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "library_versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pyyaml": __import__("yaml").__version__,
        },
        "command": command,
    }
    policy_index = output_dir / "policies" / "index.csv"
    if policy_index.is_file():
        with policy_index.open(newline="", encoding="utf-8") as handle:
            manifest["saved_policy_count"] = sum(1 for _ in csv.DictReader(handle))
    else:
        manifest["saved_policy_count"] = 0
    manifest["methods"] = methods
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _save_selected_policy(
    actor: torch.nn.Module,
    policy_dir: Path,
    method: str,
    train_seed: int,
    step: int,
    configuration_hash: str,
    validation_collision_rate: float,
    validation_mean_fuel: float,
) -> Path:
    """Persist one selected actor and its integrity metadata."""
    policy_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{method}_seed{train_seed}_step{step}.pt"
    actor_path = policy_dir / file_name
    torch.save(actor.state_dict(), actor_path)
    digest = hashlib.sha256(actor_path.read_bytes()).hexdigest()
    metadata = {
        "method": method,
        "train_seed": train_seed,
        "step": step,
        "config_hash": configuration_hash,
        "validation_collision_rate": validation_collision_rate,
        "validation_mean_fuel": validation_mean_fuel,
        "sha256": digest,
    }
    with actor_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    index_path = policy_dir / "index.csv"
    fields = [
        "method",
        "train_seed",
        "step",
        "file",
        "sha256",
        "validation_collision_rate",
        "validation_mean_fuel",
    ]
    write_header = not index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "method": method,
            "train_seed": train_seed,
            "step": step,
            "file": file_name,
            "sha256": digest,
            "validation_collision_rate": validation_collision_rate,
            "validation_mean_fuel": validation_mean_fuel,
        })
    return actor_path


def _method_name(config: dict, proximity_weight: float) -> str:
    primary_weight = float(get(config, "reward", "proximity_weight"))
    if proximity_weight == primary_weight:
        return "sac_primary"
    return f"sac_ablation_proximity_{proximity_weight:.2f}".replace(".00", "")


def _configured_methods(config: dict) -> dict[str, float]:
    return {
        _method_name(config, float(weight)): float(weight)
        for weight in get(config, "ablations", "proximity_weight_values")
    }


def _prepare_output_dir(output_dir: Path) -> None:
    """Refuse to reuse output content, allowing only a fresh launch log reservation."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    entries = list(output_dir.iterdir())
    if not entries:
        return
    reservation = output_dir / ".runner-output-reservation"
    token = os.environ.get("AEGIS_ORBIT_OUTPUT_RESERVATION")
    allowed_names = {
        ".runner-output-reservation",
        "run.log",
        "run.log.err",
        "pid.txt",
    }
    if (
        token
        and reservation.is_file()
        and reservation.read_text(encoding="utf-8") == token
        and {entry.name for entry in entries}
        == allowed_names | {"launch.json"}
        and all((output_dir / name).is_file() for name in allowed_names)
        and (output_dir / "launch.json").is_file()
        and all((output_dir / name).stat().st_size == 0 for name in ("run.log", "run.log.err"))
    ):
        reservation.unlink()
        return
    raise FileExistsError(f"Refusing to write into non-empty output directory: {output_dir}")


def run_baseline_episodes(
    method: str,
    policy,
    seeds: range,
    config: dict,
    hard_body_radius_m: float,
) -> list[dict]:
    """Evaluate one baseline and report each completed episode."""
    rows = []
    total = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        row = run_episode(
            make_environment(config, seed),
            policy,
            np.random.default_rng(seed),
            hard_body_radius_m,
            config,
        )
        rows.append({
            "method": method,
            "train_seed": -1,
            "eval_seed": seed,
            "config_hash": config_hash(config),
            **row,
        })
        progress(f"{method} episode {index}/{total} seed={seed} completed")
    return rows


def deterministic_evaluation_policy(agent: SACAgent):
    """Return the deterministic action path used for policy evaluation."""
    return lambda state, _rng: agent.action(state, deterministic=True)


def main(
    config_path: Path = Path("configs/sac_unshielded.yaml"),
    output_dir: Path = Path("results"),
    methods: list[str] | None = None,
) -> None:
    """Run primary SAC, proximity ablations, and all unshielded references."""
    global _RUN_STARTED_AT
    _RUN_STARTED_AT = time.monotonic()
    _prepare_output_dir(output_dir)
    config = load_config(config_path)
    configured_methods = _configured_methods(config)
    selected_methods = list(configured_methods) if methods is None else methods
    if not selected_methods:
        raise ValueError("At least one SAC method must be selected")
    unknown_methods = sorted(set(selected_methods) - set(configured_methods))
    if unknown_methods:
        raise ValueError(
            f"Unknown methods {unknown_methods}; configured methods are {sorted(configured_methods)}"
        )
    selected_weights = {
        method: configured_methods[method]
        for method in selected_methods
    }
    hard = float(get(config, "safety_geometry", "hard_body_radius_m"))
    start = int(get(config, "evaluation_seeds", "start"))
    count = int(get(config, "evaluation_seeds", "count"))
    rows = []
    stability_logs = []
    reward_logs = []
    selection_logs = []
    preflight_started = time.monotonic()
    preflight = oracle_preflight(config)
    progress(
        "oracle pre-flight passed "
        f"passive_collision_count={preflight['passive_collision_count']} "
        f"oracle_avoidable_count={preflight['oracle_avoidable_count']} "
        f"minimum_margin_m={preflight['minimum_margin_m']:.6g} "
        f"preflight_elapsed_s={time.monotonic() - preflight_started:.1f}"
    )
    passive_rows = run_baseline_episodes(
        "passive",
        passive_policy,
        range(start, start + count),
        config,
        hard,
    )
    rows.extend(passive_rows)
    oracle_results = []
    for row in passive_rows:
        if row["collided"]:
            seed = int(row["eval_seed"])
            initial_state = make_environment(config, seed).reset(np.random.default_rng(seed))
            oracle_results.append(
                evaluate_encounter(seed, config, hard, initial_state=initial_state)
            )
        else:
            oracle_results.append((False, 0.0))
    oracle_summary = summarize_oracle_avoidable_subset(
        [bool(row["collided"]) for row in passive_rows],
        oracle_results,
    )
    oracle_row = {
        "method": "oracle_avoidable_subset",
        "method_label": "oracle finite-search ceiling, not a policy result",
        "train_seed": -1,
        "eval_seed": -1,
        "config_hash": config_hash(config),
        "collided": "",
        "min_distance_m": "",
        "final_distance_m": "",
        "fuel_mps": "",
        "clipped_fraction": "",
        "episode_length": "",
        "fuel_term": "",
        "collision_term": "",
        "proximity_term": "",
        "survival_term": "",
        "reward": "",
        "oracle_passive_collision_count": oracle_summary["passive_collision_count"],
        "oracle_avoidable_count": oracle_summary["oracle_avoidable_count"],
        "oracle_avoidable_fraction": oracle_summary["avoidable_fraction"],
        "oracle_mean_best_min_distance_m": oracle_summary["mean_best_min_distance_m"],
    }
    rows.append(oracle_row)
    random = random_policy(float(get(config, "environment", "u_max")))
    rows.extend(run_baseline_episodes("random", random, range(start, start + count), config, hard))

    for method_name, proximity_weight in selected_weights.items():
        for train_seed in get(config, "seeds"):
            progress(f"training start method={method_name} train_seed={train_seed}")
            agent, _, _ = train(
                train_seed,
                config,
                proximity_weight,
                [],
                stability_logs,
                reward_logs,
                selection_logs,
                output_dir / "checkpoint_selection.csv",
                output_dir / "policies",
                method_name,
            )
            progress(f"training end method={method_name} train_seed={train_seed}")
            for index, seed in enumerate(range(start, start + count), start=1):
                rows.append({
                    "method": method_name,
                    "train_seed": train_seed,
                    "eval_seed": seed,
                    "config_hash": config_hash(config),
                    **run_episode(
                        make_environment(config, seed),
                        deterministic_evaluation_policy(agent),
                        np.random.default_rng(seed),
                        hard,
                        config,
                        proximity_weight,
                    ),
                })
                progress(
                    f"{method_name} evaluation train_seed={train_seed} "
                    f"episode {index}/{count} eval_seed={seed} completed"
                )

    with (output_dir / "unshielded_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in reward_logs for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reward_logs)
    with (output_dir / "stability_log.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "step", "actor_loss", "critic_loss", "q_min", "q_max", "alpha", "entropy", "nan"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(stability_logs)
    with (output_dir / "checkpoint_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "proximity_weight", "step", "collision_rate", "mean_fuel", "selected"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selection_logs)
    summary_csv, oracle_summary_csv = write_summary_files(
        output_dir,
        rows,
        oracle_summary,
        oracle_row["method_label"],
    )
    _write_manifest(
        config,
        output_dir,
        f"python -m benchmarks.run_unshielded_benchmark --config {config_path} "
        f"--output-dir {output_dir} --methods {' '.join(selected_methods)}",
        selected_methods,
    )
    from .check_gates import check_gates

    checks = check_gates(config_path, output_dir)
    print("Artifact-integrity checks (not the original Step 2C gate definitions)", flush=True)
    for name, passed, evidence in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}: {evidence}", flush=True)
    if not all(passed for _, passed, _ in checks):
        raise RuntimeError("One or more benchmark artifact-integrity checks failed")

    print("The oracle is a finite constant-thrust search. avoidable=False means not found, not impossible.")
    print(summary_csv, end="", flush=True)
    print("Oracle summary CSV:", flush=True)
    print(oracle_summary_csv, end="", flush=True)


def cli(arguments: list[str] | None = None) -> None:
    """Parse the command-line config/output paths and run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sac_unshielded.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--methods", nargs="+", default=None)
    parsed = parser.parse_args(arguments)
    main(parsed.config, parsed.output_dir, parsed.methods)


if __name__ == "__main__":
    cli()
