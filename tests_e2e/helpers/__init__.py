"""Shared automated verification helpers for tests_e2e."""

from .arena_parser import (
    SdfCollision,
    SdfWorldAudit,
    check_line_of_sight,
    load_obstacles_yaml,
    parse_sdf_collisions,
    point_min_clearance,
)
from .hardware_git_guard import (
    GitAnchorResult,
    Gpu3Status,
    OgreLogAudit,
    check_cuda_env_excludes_gpu3,
    check_git_anchors,
    check_gpu3_zero_occupancy,
    check_ogre2_log_gpu3_excluded,
)
from .media_validator import (
    VideoProbeResult,
    VideoQualityResult,
    check_video_quality,
    probe_video,
)
from .metrics_validator import (
    MetricsValidationResult,
    MsgValidationResult,
    TraceValidationResult,
    validate_kill_event_msg,
    validate_mission_summary,
    validate_mission_trace,
    validate_planning_target_msg,
    validate_vision_targets_msg,
)
from .ros_probe import (
    ColconTestResult,
    CommandResult,
    echo_topic_once,
    probe_ros_topics,
    run_colcon_test,
    run_command_with_gpu_guard,
)

__all__ = [
    "ColconTestResult",
    "CommandResult",
    "GitAnchorResult",
    "Gpu3Status",
    "MetricsValidationResult",
    "MsgValidationResult",
    "OgreLogAudit",
    "SdfCollision",
    "SdfWorldAudit",
    "TraceValidationResult",
    "VideoProbeResult",
    "VideoQualityResult",
    "check_cuda_env_excludes_gpu3",
    "check_git_anchors",
    "check_gpu3_zero_occupancy",
    "check_line_of_sight",
    "check_ogre2_log_gpu3_excluded",
    "check_video_quality",
    "echo_topic_once",
    "load_obstacles_yaml",
    "parse_sdf_collisions",
    "point_min_clearance",
    "probe_ros_topics",
    "probe_video",
    "run_colcon_test",
    "run_command_with_gpu_guard",
    "validate_kill_event_msg",
    "validate_mission_summary",
    "validate_mission_trace",
    "validate_planning_target_msg",
    "validate_vision_targets_msg",
]
