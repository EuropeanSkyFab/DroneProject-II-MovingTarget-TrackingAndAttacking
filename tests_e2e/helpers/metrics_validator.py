"""Quantitative telemetry, trace geometric recomputation, and ROS JSON schema validators."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .arena_parser import point_min_clearance

REQUIRED_SUMMARY_KEYS = (
    "trace_id",
    "drone_count",
    "target_count",
    "obstacle_count",
    "vision_samples",
    "stereo_3d_samples",
    "vision_3d_rmse_m",
    "imm_pred_1s_rmse_m",
    "min_inter_drone_dist_m",
    "min_obstacle_clearance_m",
    "inter_uav_collisions",
    "obstacle_collisions",
    "targets_neutralized",
    "target_kill_details",
    "re_tasking_events",
    "return_helipad_error_m",
    "process_crashes",
    "gpu3_violation",
    "completed",
)

VALID_PLANNING_ACTIONS = {
    "SEARCH",
    "TRACK",
    "INTERCEPT",
    "ATTACK",
    "RETURN",
    "PATROL",
}


@dataclass
class MetricsValidationResult:
    """Validation result for mission_summary.json."""

    ok: bool
    errors: List[str]
    data: Dict[str, Any]


@dataclass
class TraceValidationResult:
    """Validation result for mission_trace.jsonl with recomputed geometry metrics."""

    ok: bool
    frame_count: int
    errors: List[str]
    computed_min_inter_drone_m: Optional[float]
    computed_min_obstacle_m: Optional[float]
    phases_observed: List[str] = field(default_factory=list)


@dataclass
class MsgValidationResult:
    """Validation result for ROS 2 JSON payload contracts."""

    ok: bool
    errors: List[str]
    data: Dict[str, Any]


def _is_finite_num(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def _is_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _parse_json_payload(payload: Union[str, Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    if isinstance(payload, dict):
        return payload, None
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            return None, f"Invalid JSON payload: {exc}"
        if not isinstance(parsed, dict):
            return None, f"Expected JSON object, got {type(parsed).__name__}"
        return parsed, None
    return None, f"Expected str or dict payload, got {type(payload).__name__}"


def validate_mission_summary(
    summary_path: Union[str, Path],
    min_drones: int = 6,
    min_targets: int = 3,
    min_obstacles: int = 6,
    min_inter_drone_m: float = 1.0,
    min_obstacle_clearance_m: float = 0.3,
    max_helipad_err_m: float = 2.0,
) -> MetricsValidationResult:
    """Validate Contract #7 mission summary JSON against quantitative thresholds."""
    path = Path(summary_path)
    if not path.is_file():
        return MetricsValidationResult(
            ok=False,
            errors=[f"Summary file does not exist: {path}"],
            data={},
        )

    try:
        raw_text = path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except (OSError, json.JSONDecodeError) as exc:
        return MetricsValidationResult(
            ok=False,
            errors=[f"Failed to parse summary JSON {path}: {exc}"],
            data={},
        )

    if not isinstance(data, dict):
        return MetricsValidationResult(
            ok=False,
            errors=[f"Summary root must be JSON object, got {type(data).__name__}"],
            data={},
        )

    errors: List[str] = []
    for key in REQUIRED_SUMMARY_KEYS:
        if key not in data:
            errors.append(f"Missing required Contract #7 key '{key}'")

    if errors:
        return MetricsValidationResult(ok=False, errors=errors, data=data)

    if not isinstance(data["trace_id"], str) or not data["trace_id"].strip():
        errors.append(f"trace_id must be non-empty string, got {data['trace_id']!r}")

    if not _is_int(data["drone_count"]) or data["drone_count"] < min_drones:
        errors.append(f"drone_count={data['drone_count']!r} < min_drones={min_drones}")

    if not _is_int(data["target_count"]) or data["target_count"] < min_targets:
        errors.append(f"target_count={data['target_count']!r} < min_targets={min_targets}")

    if not _is_int(data["obstacle_count"]) or data["obstacle_count"] < min_obstacles:
        errors.append(
            f"obstacle_count={data['obstacle_count']!r} < min_obstacles={min_obstacles}"
        )

    if not _is_int(data["vision_samples"]) or data["vision_samples"] <= 0:
        errors.append(f"vision_samples must be int > 0, got {data['vision_samples']!r}")

    if not _is_int(data["stereo_3d_samples"]) or data["stereo_3d_samples"] <= 0:
        errors.append(
            f"stereo_3d_samples must be int > 0, got {data['stereo_3d_samples']!r}"
        )
    elif _is_int(data["vision_samples"]) and data["stereo_3d_samples"] > data["vision_samples"]:
        errors.append(
            f"stereo_3d_samples ({data['stereo_3d_samples']}) > vision_samples ({data['vision_samples']})"
        )

    for metric_key in ("vision_3d_rmse_m", "imm_pred_1s_rmse_m"):
        val = data[metric_key]
        if not _is_finite_num(val) or float(val) < 0.0:
            errors.append(f"{metric_key} must be finite >= 0, got {val!r}")

    min_inter = data["min_inter_drone_dist_m"]
    if not _is_finite_num(min_inter) or float(min_inter) < min_inter_drone_m:
        errors.append(
            f"min_inter_drone_dist_m={min_inter!r} < min_inter_drone_m={min_inter_drone_m}"
        )

    min_obs = data["min_obstacle_clearance_m"]
    if not _is_finite_num(min_obs) or float(min_obs) < min_obstacle_clearance_m:
        errors.append(
            f"min_obstacle_clearance_m={min_obs!r} < min_obstacle_clearance_m={min_obstacle_clearance_m}"
        )

    if not _is_int(data["inter_uav_collisions"]) or data["inter_uav_collisions"] != 0:
        errors.append(
            f"inter_uav_collisions must be 0, got {data['inter_uav_collisions']!r}"
        )

    if not _is_int(data["obstacle_collisions"]) or data["obstacle_collisions"] != 0:
        errors.append(
            f"obstacle_collisions must be 0, got {data['obstacle_collisions']!r}"
        )

    if not _is_int(data["targets_neutralized"]):
        errors.append(
            f"targets_neutralized must be int, got {data['targets_neutralized']!r}"
        )
    else:
        if data["targets_neutralized"] < min_targets:
            errors.append(
                f"targets_neutralized={data['targets_neutralized']} < min_targets={min_targets}"
            )
        if _is_int(data["target_count"]) and data["targets_neutralized"] != data["target_count"]:
            errors.append(
                f"targets_neutralized ({data['targets_neutralized']}) != target_count ({data['target_count']})"
            )

    kill_details = data["target_kill_details"]
    if not isinstance(kill_details, list):
        errors.append(
            f"target_kill_details must be a list, got {type(kill_details).__name__}"
        )
    else:
        if _is_int(data["targets_neutralized"]) and len(kill_details) < data["targets_neutralized"]:
            errors.append(
                f"len(target_kill_details)={len(kill_details)} < targets_neutralized={data['targets_neutralized']}"
            )
        for idx, item in enumerate(kill_details):
            if not isinstance(item, dict):
                errors.append(f"target_kill_details[{idx}] must be object")
                continue
            if not isinstance(item.get("target_id"), str) or not item["target_id"].strip():
                errors.append(f"target_kill_details[{idx}] missing valid target_id")
            if not isinstance(item.get("striker_id"), str) or not item["striker_id"].strip():
                errors.append(f"target_kill_details[{idx}] missing valid striker_id")
            md = item.get("miss_dist_m")
            if not _is_finite_num(md) or float(md) < 0.0:
                errors.append(
                    f"target_kill_details[{idx}].miss_dist_m must be finite >= 0, got {md!r}"
                )

    if not _is_int(data["re_tasking_events"]) or data["re_tasking_events"] < 0:
        errors.append(
            f"re_tasking_events must be int >= 0, got {data['re_tasking_events']!r}"
        )

    helipad_err = data["return_helipad_error_m"]
    if isinstance(helipad_err, dict):
        if not helipad_err:
            errors.append("return_helipad_error_m dict must not be empty")
        for did, err_val in helipad_err.items():
            if (
                not _is_finite_num(err_val)
                or float(err_val) < 0.0
                or float(err_val) > max_helipad_err_m
            ):
                errors.append(
                    f"return_helipad_error_m[{did!r}]={err_val!r} out of [0, {max_helipad_err_m}]"
                )
    elif _is_finite_num(helipad_err):
        if float(helipad_err) < 0.0 or float(helipad_err) > max_helipad_err_m:
            errors.append(
                f"return_helipad_error_m={helipad_err!r} out of [0, {max_helipad_err_m}]"
            )
    else:
        errors.append(
            f"return_helipad_error_m must be dict or finite number, got {helipad_err!r}"
        )

    if not _is_int(data["process_crashes"]) or data["process_crashes"] != 0:
        errors.append(f"process_crashes must be 0, got {data['process_crashes']!r}")

    if data["gpu3_violation"] is not False:
        errors.append(f"gpu3_violation must be False, got {data['gpu3_violation']!r}")

    if data["completed"] is not True:
        errors.append(f"completed must be True, got {data['completed']!r}")

    return MetricsValidationResult(ok=(len(errors) == 0), errors=errors, data=data)


def validate_mission_trace(
    trace_path: Union[str, Path],
    obstacles_cfg: Optional[Dict[str, Any]] = None,
    summary_data: Optional[Dict[str, Any]] = None,
    min_airborne_z: float = 1.2,
    dist_tolerance_m: float = 0.25,
) -> TraceValidationResult:
    """Validate JSONL mission trace and independently recompute geometric clearances."""
    path = Path(trace_path)
    if not path.is_file():
        return TraceValidationResult(
            ok=False,
            frame_count=0,
            errors=[f"Trace file does not exist: {path}"],
            computed_min_inter_drone_m=None,
            computed_min_obstacle_m=None,
        )

    errors: List[str] = []
    phases_observed: List[str] = []
    seen_phases = set()
    trace_ids = set()
    prev_ts: Optional[float] = None
    frame_count = 0
    airborne_multi_drone_frames = 0
    min_inter_drone: Optional[float] = None
    min_obstacle_clr: Optional[float] = None

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return TraceValidationResult(
            ok=False,
            frame_count=0,
            errors=[f"Failed to read trace file {path}: {exc}"],
            computed_min_inter_drone_m=None,
            computed_min_obstacle_m=None,
        )

    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        frame_count += 1
        try:
            frame = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(f"Line {line_no}: invalid JSON ({exc})")
            continue

        if not isinstance(frame, dict):
            errors.append(f"Line {line_no}: expected JSON object")
            continue

        ts = frame.get("timestamp")
        if not _is_finite_num(ts):
            errors.append(f"Line {line_no}: non-finite or missing timestamp {ts!r}")
        else:
            ts_f = float(ts)
            if prev_ts is not None and ts_f < prev_ts:
                errors.append(
                    f"Line {line_no}: non-monotonic timestamp {ts_f} < {prev_ts}"
                )
            prev_ts = ts_f

        tid = frame.get("trace_id")
        if not isinstance(tid, str) or not tid.strip():
            errors.append(f"Line {line_no}: missing/empty trace_id")
        else:
            trace_ids.add(tid)

        sid = frame.get("span_id")
        if not isinstance(sid, str) or not sid.strip():
            errors.append(f"Line {line_no}: missing/empty span_id")

        phase = frame.get("phase")
        if not isinstance(phase, str) or not phase.strip():
            errors.append(f"Line {line_no}: missing/empty phase")
        elif phase not in seen_phases:
            seen_phases.add(phase)
            phases_observed.append(phase)

        drones = frame.get("drones")
        if not isinstance(drones, dict):
            errors.append(f"Line {line_no}: 'drones' must be a dict")
            continue

        airborne_pts: List[tuple[str, float, float, float]] = []
        for did, dstate in drones.items():
            if not isinstance(dstate, dict):
                errors.append(f"Line {line_no} drone '{did}': state must be dict")
                continue
            pos = dstate.get("pos")
            vel = dstate.get("vel")
            if (
                not isinstance(pos, list)
                or len(pos) != 3
                or not all(_is_finite_num(v) for v in pos)
            ):
                errors.append(f"Line {line_no} drone '{did}': invalid pos {pos!r}")
                continue
            if (
                not isinstance(vel, list)
                or len(vel) != 3
                or not all(_is_finite_num(v) for v in vel)
            ):
                errors.append(f"Line {line_no} drone '{did}': invalid vel {vel!r}")
                continue

            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            if pz > min_airborne_z:
                airborne_pts.append((str(did), px, py, pz))

            if obstacles_cfg is not None and pz > 0.5:
                clr = point_min_clearance((px, py, pz), obstacles_cfg)
                if min_obstacle_clr is None or clr < min_obstacle_clr:
                    min_obstacle_clr = clr
                if clr < 0.0:
                    errors.append(
                        f"Line {line_no} drone '{did}': obstacle penetration (clearance={clr:.4f}m)"
                    )

        if len(airborne_pts) >= 2:
            airborne_multi_drone_frames += 1
            for i in range(len(airborne_pts)):
                _, x1, y1, z1 = airborne_pts[i]
                for j in range(i + 1, len(airborne_pts)):
                    _, x2, y2, z2 = airborne_pts[j]
                    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
                    if min_inter_drone is None or dist < min_inter_drone:
                        min_inter_drone = dist

    if frame_count == 0:
        errors.append(f"Trace file {path} contains 0 frames")
    elif airborne_multi_drone_frames == 0:
        errors.append("No airborne multi-drone frames found in trace")

    if summary_data is not None:
        exp_tid = summary_data.get("trace_id")
        if isinstance(exp_tid, str) and trace_ids and exp_tid not in trace_ids:
            errors.append(
                f"Summary trace_id '{exp_tid}' does not match trace trace_ids {sorted(trace_ids)}"
            )
        if min_inter_drone is not None and _is_finite_num(summary_data.get("min_inter_drone_dist_m")):
            sum_inter = float(summary_data["min_inter_drone_dist_m"])
            if abs(min_inter_drone - sum_inter) > dist_tolerance_m:
                errors.append(
                    f"Recomputed min_inter_drone_dist_m ({min_inter_drone:.4f}m) "
                    f"mismatches summary ({sum_inter:.4f}m) beyond tolerance {dist_tolerance_m}m"
                )
        if (
            min_obstacle_clr is not None
            and _is_finite_num(summary_data.get("min_obstacle_clearance_m"))
        ):
            sum_obs = float(summary_data["min_obstacle_clearance_m"])
            if abs(min_obstacle_clr - sum_obs) > dist_tolerance_m:
                errors.append(
                    f"Recomputed min_obstacle_clearance_m ({min_obstacle_clr:.4f}m) "
                    f"mismatches summary ({sum_obs:.4f}m) beyond tolerance {dist_tolerance_m}m"
                )

    return TraceValidationResult(
        ok=(len(errors) == 0),
        frame_count=frame_count,
        errors=errors,
        computed_min_inter_drone_m=min_inter_drone,
        computed_min_obstacle_m=min_obstacle_clr,
        phases_observed=phases_observed,
    )


def validate_vision_targets_msg(
    payload: Union[str, Dict[str, Any]],
) -> MsgValidationResult:
    """Validate Contract #3 /vision/targets JSON message schema and finiteness."""
    data, parse_err = _parse_json_payload(payload)
    if parse_err is not None or data is None:
        return MsgValidationResult(ok=False, errors=[parse_err or "Invalid payload"], data={})

    errors: List[str] = []
    if not _is_finite_num(data.get("stamp")) or float(data["stamp"]) <= 0.0:
        errors.append(f"stamp must be finite > 0, got {data.get('stamp')!r}")
    if data.get("frame_id") != "world":
        errors.append(f"frame_id must be 'world', got {data.get('frame_id')!r}")

    targets = data.get("targets")
    if not isinstance(targets, list):
        errors.append(f"targets must be a list, got {type(targets).__name__}")
    else:
        for i, tgt in enumerate(targets):
            if not isinstance(tgt, dict):
                errors.append(f"targets[{i}] must be dict")
                continue
            if not _is_int(tgt.get("det_id")) or tgt["det_id"] < 0:
                errors.append(f"targets[{i}].det_id must be int >= 0")
            if not isinstance(tgt.get("class_id"), str) or not tgt["class_id"].strip():
                errors.append(f"targets[{i}].class_id must be non-empty str")
            for coord in ("x", "y", "z"):
                if not _is_finite_num(tgt.get(coord)):
                    errors.append(f"targets[{i}].{coord} must be finite")
            conf = tgt.get("conf")
            if not _is_finite_num(conf) or not (0.0 <= float(conf) <= 1.0):
                errors.append(f"targets[{i}].conf must be finite in [0, 1], got {conf!r}")
            stereo = tgt.get("stereo")
            if not isinstance(stereo, bool):
                errors.append(f"targets[{i}].stereo must be bool")
            rays = tgt.get("rays_used")
            if not _is_int(rays) or rays < (2 if stereo is True else 1):
                errors.append(f"targets[{i}].rays_used invalid for stereo={stereo!r}: {rays!r}")
            for nonneg in ("parallax_deg", "skew_rms_m"):
                val = tgt.get(nonneg)
                if not _is_finite_num(val) or float(val) < 0.0:
                    errors.append(f"targets[{i}].{nonneg} must be finite >= 0, got {val!r}")

    return MsgValidationResult(ok=(len(errors) == 0), errors=errors, data=data)


def validate_planning_target_msg(
    payload: Union[str, Dict[str, Any]],
) -> MsgValidationResult:
    """Validate Contract #5 /planning/target_pose JSON message schema and finiteness."""
    data, parse_err = _parse_json_payload(payload)
    if parse_err is not None or data is None:
        return MsgValidationResult(ok=False, errors=[parse_err or "Invalid payload"], data={})

    errors: List[str] = []
    if not isinstance(data.get("drone_id"), str) or not data["drone_id"].strip():
        errors.append(f"drone_id must be non-empty str, got {data.get('drone_id')!r}")
    if not _is_finite_num(data.get("stamp")) or float(data["stamp"]) <= 0.0:
        errors.append(f"stamp must be finite > 0, got {data.get('stamp')!r}")
    for coord in ("x", "y", "z"):
        if not _is_finite_num(data.get(coord)):
            errors.append(f"{coord} must be finite, got {data.get(coord)!r}")

    action = data.get("action")
    if not isinstance(action, str) or action not in VALID_PLANNING_ACTIONS:
        errors.append(
            f"action must be in {sorted(VALID_PLANNING_ACTIONS)}, got {action!r}"
        )

    t_go = data.get("t_go")
    if not _is_finite_num(t_go) or float(t_go) < 0.0:
        errors.append(f"t_go must be finite >= 0, got {t_go!r}")

    for opt_num in ("vx", "vy", "vz", "ax", "ay", "az", "focus_x", "focus_y", "focus_z"):
        if opt_num in data and data[opt_num] is not None and not _is_finite_num(data[opt_num]):
            errors.append(f"optional field {opt_num} must be finite, got {data[opt_num]!r}")

    if "imm_probs" in data and data["imm_probs"] is not None:
        probs = data["imm_probs"]
        if not isinstance(probs, dict) or not probs:
            errors.append("imm_probs must be non-empty dict when present")
        else:
            prob_sum = 0.0
            for k, v in probs.items():
                if not _is_finite_num(v) or not (0.0 <= float(v) <= 1.0):
                    errors.append(f"imm_probs[{k!r}] must be in [0, 1], got {v!r}")
                else:
                    prob_sum += float(v)
            if abs(prob_sum - 1.0) > 0.05:
                errors.append(f"imm_probs sum={prob_sum:.4f} deviates from 1.0")

    if "pred_path" in data and data["pred_path"] is not None:
        ppath = data["pred_path"]
        if not isinstance(ppath, list):
            errors.append("pred_path must be a list")
        else:
            for idx, pt in enumerate(ppath):
                if (
                    not isinstance(pt, list)
                    or len(pt) != 3
                    or not all(_is_finite_num(c) for c in pt)
                ):
                    errors.append(f"pred_path[{idx}] must be 3 finite floats, got {pt!r}")

    return MsgValidationResult(ok=(len(errors) == 0), errors=errors, data=data)


def validate_kill_event_msg(
    payload: Union[str, Dict[str, Any]],
) -> MsgValidationResult:
    """Validate Contract #4 /engagement/kill_event JSON message schema and finiteness."""
    data, parse_err = _parse_json_payload(payload)
    if parse_err is not None or data is None:
        return MsgValidationResult(ok=False, errors=[parse_err or "Invalid payload"], data={})

    errors: List[str] = []
    if not _is_finite_num(data.get("stamp")) or float(data["stamp"]) <= 0.0:
        errors.append(f"stamp must be finite > 0, got {data.get('stamp')!r}")
    for sfield in ("target_id", "striker_id"):
        if not isinstance(data.get(sfield), str) or not data[sfield].strip():
            errors.append(f"{sfield} must be non-empty str, got {data.get(sfield)!r}")
    for nonneg in ("miss_dist_m", "closing_speed_ms"):
        val = data.get(nonneg)
        if not _is_finite_num(val) or float(val) < 0.0:
            errors.append(f"{nonneg} must be finite >= 0, got {val!r}")
    for coord in ("x", "y", "z"):
        if not _is_finite_num(data.get(coord)):
            errors.append(f"{coord} must be finite, got {data.get(coord)!r}")

    return MsgValidationResult(ok=(len(errors) == 0), errors=errors, data=data)
