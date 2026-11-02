"""The policy manifest (schema 2) and the checks a published policy has to pass.

One vocabulary for two shapes — a single-policy repo (fields at the top level) and the official
set (the same fields per entry under ``policies``). This module writes the first; the daemon
(`pollen-robotics/microduck`, ``updater/src/policy.rs`` and ``robotd-params``) reads both. The
contract is `docs/policy-manifest.md` over there; the numbers below are what the daemon publishes
in ``duck_ipc_proto`` and refuses a policy for disagreeing with.

Deliberately free of mjlab / torch imports so the tests run on a laptop in milliseconds and the
CLI can validate an ONNX file without a GPU.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 2
# `duck_ipc_proto`: the daemon refuses a policy whose manifest disagrees with these, and refuses
# at load a network whose graph does. 61 = 48 proprioception + 13 command; 14 = the servos.
MODEL_API = 1
OBS_LEN = 61
ACTION_LEN = 14
# Perception (manifest axis `perception`, docs/policy-manifest.md in the `microduck` repo): what the
# network reads besides the robot's own state. `tof8x8` = the 61 above, bit-identical, then the
# head ToF's 64 zones (tasks/tof.py is the sim side of that contract). The daemon reads which
# kind a network is off its `obs` input, so the ONNX width is the ground truth here too.
PERCEPTIONS: dict[str, int] = {"none": OBS_LEN, "tof8x8": OBS_LEN + 64}
MODEL_API_PERCEPTIVE = 3   # a daemon that cannot feed the frame refuses at fetch, not at load
ROBOT: dict[str, Any] = {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50}

# The one `.onnx` a repo carries. The daemon takes the sole `.onnx` in a repo and refuses several.
POLICY_FILE = "policy.onnx"

# A rollout video, if the repo carries one. The Hub's replay widget picks up a file of exactly
# this name, so it is shown there and neither the manifest nor the README points at it.
REPLAY_FILE = "replay.mp4"

Kind = Literal["episodic", "perpetual"]
KINDS: tuple[str, ...] = ("episodic", "perpetual")

ZERO_TWIST: tuple[float, float, float] = (0.0, 0.0, 0.0)


def perception_of_width(obs_len: int) -> str:
    """The perception an observation width implies (what the daemon reads off the graph)."""
    for name, width in PERCEPTIONS.items():
        if width == obs_len:
            return name
    raise ManifestError(
        f"observation width {obs_len}: the robot builds {OBS_LEN} (blind) or {PERCEPTIONS['tof8x8']} "
        "(tof8x8); a 51-D policy is the legacy 3-value-command family, which the daemon refuses"
    )


def perception_of_task(task_id: str) -> str:
    """The perception a registered task trains: `tof8x8` when its actor observation carries the
    head ToF term, else `none`. Read off the env cfg, so nothing has to be remembered at publish."""
    import mjlab.tasks  # noqa: F401  (populates the registry)
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    terms = load_env_cfg(task_id, play=True).observations["actor"].terms
    return "tof8x8" if "tof" in terms else "none"

# The daemon's policy slots, for a gait's `slot` hint (display-only: `robotctl policy load <slot>`).
SLOTS: tuple[str, ...] = ("walk", "stand", "sitstand", "ground_pick", "kick_left", "kick_right", "roulade")


class ManifestError(ValueError):
    """A manifest that the daemon would refuse, or that would load and run wrongly."""


@dataclass(frozen=True)
class Provenance:
    """Where the weights came from. Display-only for the daemon; the part people skip by hand."""

    task_id: str | None = None
    repo: str = "pollen-robotics/microduck_rl"
    commit: str | None = None
    branch: str | None = None
    dirty: bool | None = None
    run: str | None = None
    checkpoint: int | None = None
    source_file: str | None = None
    exported: str = field(default_factory=lambda: _now_utc())

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    """`commit`, `branch`, `dirty` of the checkout the export ran from, or `{}` outside git."""
    root = str(repo_root or Path(__file__).resolve().parents[3])

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", root, *args], capture_output=True, text=True, check=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = git("rev-parse", "--short=9", "HEAD")
    if commit is None:
        return {}
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "branch": branch, "dirty": bool(status)}


def build_manifest(
    *,
    name: str,
    kind: str,
    description: str,
    duration_s: float | None = None,
    chain: bool = False,
    unwind_s: float | None = None,
    idle: tuple[float, float, float] = ZERO_TWIST,
    action_scale: float | None = None,
    entry_pose: str = "standing",
    slot: str | None = None,
    command_help: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    eval: dict[str, Any] | None = None,
    perception: str = "none",
) -> dict[str, Any]:
    """A single-policy manifest the daemon loads without surprises.

    Only the constant-command family is publishable from here — a skill's network is fed a fixed
    twist. Phase and posture-flag encodings are the official set's own arms and are not something
    a community policy can be.
    """
    if kind not in KINDS:
        raise ManifestError(f"kind must be one of {KINDS}, not {kind!r}")
    if perception not in PERCEPTIONS:
        raise ManifestError(f"perception must be one of {tuple(PERCEPTIONS)}, not {perception!r}")
    if not name or "/" in name or name != name.strip():
        raise ManifestError(f"name must be a bare word a client can ask for, not {name!r}")
    if kind == "episodic":
        if duration_s is None or duration_s <= 0:
            raise ManifestError(
                "an episodic policy ends itself: say how long it runs with duration_s > 0"
            )
        if unwind_s:
            raise ManifestError(
                "an episodic policy is already back when duration_s is up; unwind_s is for perpetual"
            )
    else:
        # Two things are perpetual: a gait, which lives in a slot (`policy load walk <repo>`) and
        # needs nothing here, and a held pose like the flamingo, which the owner runs as a
        # one-shot with `policy add --hold` and which then needs `unwind_s` so the robot is not
        # let go of on one foot. `unwind_s` is what says which.
        if duration_s is not None:
            raise ManifestError(
                "a perpetual policy has no length of its own; leave duration_s unset "
                "(a gait runs until told otherwise; a held pose gets --hold when added as a skill)"
            )
        if unwind_s is not None and unwind_s <= 0:
            raise ManifestError("unwind_s must be > 0 when given")
        if chain:
            raise ManifestError("chain is for episodic one-shots a held button repeats")
    if slot is not None and slot not in SLOTS:
        raise ManifestError(f"slot must be one of {SLOTS}, not {slot!r}")
    if action_scale is not None and not 0 < action_scale <= 2.0:
        raise ManifestError(f"action_scale {action_scale} is outside (0, 2]")
    if len(idle) != 3:
        raise ManifestError("idle is a 3-vector twist")

    command: dict[str, Any] = {
        "encoding": "constant",
        "idle": [float(v) for v in idle],
        "twist": "unused (zeros)",
        "head": "unused (zeros)",
        "body": "unused (zeros)",
    }
    if command_help:
        command.update(command_help)

    perceptive = perception != "none"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_api": MODEL_API_PERCEPTIVE if perceptive else MODEL_API,
        "obs_len": PERCEPTIONS[perception],
        "action_len": ACTION_LEN,
        "robot": dict(ROBOT),
        "name": name,
        "kind": kind,
        "entry_pose": entry_pose,
        "description": description,
        "command": command,
    }
    if perceptive:
        manifest["perception"] = perception
    if kind == "episodic":
        manifest["duration_s"] = float(duration_s)  # type: ignore[arg-type]
        manifest["chain"] = bool(chain)
    else:
        manifest["duration_s"] = None
        if unwind_s is not None:
            manifest["unwind_s"] = float(unwind_s)
    if slot is not None:
        manifest["slot"] = slot
    if action_scale is not None:
        manifest["action_scale"] = float(action_scale)
    if training:
        manifest["training"] = training
    if eval:
        manifest["eval"] = eval
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Refuse what the daemon would refuse, plus the mistakes it would load and run wrongly.

    Accepts both shapes and any schema version, because absence is not evidence — a repo is under
    no obligation to carry any field. Only a claim that is present and wrong fails.
    """
    if "policies" in manifest:
        for entry in manifest["policies"]:
            if "file" not in entry:
                raise ManifestError("every set entry needs a `file`")
            validate_manifest({k: v for k, v in entry.items() if k != "file"})
        return
    perception = manifest.get("perception") or "none"
    if perception not in PERCEPTIONS:
        raise ManifestError(f"perception {perception!r}: this robot feeds {tuple(PERCEPTIONS)}")
    expected_obs = PERCEPTIONS[perception]
    if (obs := manifest.get("obs_len")) is not None and obs != expected_obs:
        if perception == "none" and obs in PERCEPTIONS.values():
            raise ManifestError(f"obs_len {obs} without `perception`: say which frame the policy reads")
        raise ManifestError(f"obs_len {obs}: this robot builds {expected_obs} for perception {perception!r}")
    if (act := manifest.get("action_len")) is not None and act != ACTION_LEN:
        raise ManifestError(f"action_len {act}: this robot has {ACTION_LEN}")
    api = manifest.get("model_api")
    allowed = MODEL_API_PERCEPTIVE if perception != "none" else MODEL_API
    if api is not None and api > allowed:
        raise ManifestError(f"model_api {api}: this repo targets {allowed} for perception {perception!r}")
    if perception != "none" and api is not None and api < MODEL_API_PERCEPTIVE:
        raise ManifestError(f"model_api {api}: a {perception} policy needs {MODEL_API_PERCEPTIVE}")
    model = (manifest.get("robot") or {}).get("model")
    if model is not None and model.lower() != ROBOT["model"]:
        raise ManifestError(f"robot.model {model!r}: this is a {ROBOT['model']} policy repo")
    kind = manifest.get("kind")
    if kind is not None and kind not in (*KINDS, "scripted"):
        raise ManifestError(f"kind {kind!r} is not one of episodic, perpetual, scripted")
    encoding = (manifest.get("command") or {}).get("encoding")
    if encoding is not None and encoding not in ("constant", "phase", "posture_flag"):
        raise ManifestError(f"command.encoding {encoding!r} is not one the daemon drives")
    if kind == "episodic" and encoding in (None, "constant"):
        duration = manifest.get("duration_s")
        if duration is None or duration <= 0:
            raise ManifestError("an episodic constant-command policy needs duration_s > 0")
    idle = (manifest.get("command") or {}).get("idle")
    if idle is not None and len(idle) != 3:
        raise ManifestError("command.idle is a 3-vector twist")


# ---------------------------------------------------------------------------------------------
# The ONNX file: the shape gate the daemon applies at load, applied before the upload.


@dataclass(frozen=True)
class OnnxShape:
    input_name: str
    output_name: str
    obs_len: int
    action_len: int


def inspect_onnx(path: Path) -> OnnxShape:
    """The graph's single input and output widths, as the daemon checks them at load."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    initializers = {i.name for i in graph.initializer}
    inputs = [i for i in graph.input if i.name not in initializers]
    if len(inputs) != 1 or len(graph.output) != 1:
        raise ManifestError(
            f"{path.name}: expected one input and one output, found "
            f"{[i.name for i in inputs]} -> {[o.name for o in graph.output]}"
        )

    def last_dim(value) -> int:
        dims = value.type.tensor_type.shape.dim
        if not dims:
            raise ManifestError(f"{path.name}: {value.name} has no shape")
        last = dims[-1]
        if not last.HasField("dim_value"):
            raise ManifestError(f"{path.name}: {value.name}'s last dimension is symbolic")
        return int(last.dim_value)

    return OnnxShape(
        input_name=inputs[0].name,
        output_name=graph.output[0].name,
        obs_len=last_dim(inputs[0]),
        action_len=last_dim(graph.output[0]),
    )


def check_onnx(path: Path) -> OnnxShape:
    """Refuse a file the daemon would refuse at load: wrong widths — 61 -> 14 (blind) or
    125 -> 14 (tof8x8) are the two the robot builds."""
    if not path.exists():
        raise ManifestError(f"{path}: no such file")
    shape = inspect_onnx(path)
    try:
        perception_of_width(shape.obs_len)
    except ManifestError as e:
        raise ManifestError(f"{path.name}: {e}") from None
    if shape.action_len != ACTION_LEN:
        raise ManifestError(f"{path.name}: {shape.action_len} actions, the robot has {ACTION_LEN}")
    return shape


def smoke_run_onnx(path: Path, steps: int = 50, seed: int = 0) -> None:
    """Run the network on plausible inputs and refuse a NaN/inf or a saturated output.

    Not a physics rehearsal — `scripts/infer_policy.py` is that — but it catches a broken export
    (an un-baked normalizer producing NaNs on raw observations, a graph that will not execute)
    before anything is uploaded.
    """
    import numpy as np
    import onnxruntime as ort

    shape = inspect_onnx(path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    obs = np.zeros((1, shape.obs_len), dtype=np.float32)
    outputs = []
    for _ in range(steps):
        (out,) = session.run([shape.output_name], {shape.input_name: obs})
        if not np.all(np.isfinite(out)):
            raise ManifestError(f"{path.name}: the network produced a non-finite action")
        outputs.append(out)
        # Feed the action back into the last-action slots and jitter the rest, the way an
        # observation evolves on the robot; enough to leave the zero point.
        obs = rng.normal(0.0, 0.05, size=obs.shape).astype(np.float32)
        obs[0, OBS_LEN - ACTION_LEN - 13 : OBS_LEN - 13] = np.clip(out[0], -1, 1)
        if shape.obs_len > OBS_LEN:      # a ToF frame: floor ahead in the lower rows, nothing above
            frame = np.ones(64, dtype=np.float32)
            frame[32:] = rng.uniform(0.2, 0.8, size=32)
            obs[0, OBS_LEN:] = frame
    spread = float(np.std(np.stack(outputs)))
    if spread == 0.0:
        raise ManifestError(f"{path.name}: the network's output never changes; is it a real policy?")


# ---------------------------------------------------------------------------------------------
# What else goes in the repo.


def install_commands(manifest: dict[str, Any], repo_id: str) -> str:
    """The `robotctl` lines that put this policy on a robot — one story per shape.

    Episodic: a skill, length from the manifest. Perpetual with `unwind_s`: a held pose the owner
    runs as a skill with `--hold`. Perpetual without: a gait, loaded into a slot.
    """
    name = manifest["name"]
    if manifest["kind"] == "episodic":
        return f"sudo robotctl policy add {name} {repo_id}\nrobotctl robot do {name}"
    if manifest.get("unwind_s") is not None:
        return f"sudo robotctl policy add {name} {repo_id} --hold <seconds>\nrobotctl robot do {name}"
    slot = manifest.get("slot", "<slot>")
    return f"sudo robotctl policy load {slot} {repo_id}"


def render_readme(manifest: dict[str, Any], repo_id: str, base_model: str | None = None) -> str:
    """A model card that says how to run the policy on a robot, generated so it cannot go stale.

    ``base_model`` is the Hub repo a remix was trained from; the card declares it so the Hub links
    the two (the parent lists this repo among its fine-tunes).
    """
    if base_model is not None:
        _check_repo_id(base_model, "base_model")
        if base_model == repo_id:
            raise ManifestError(f"base_model {base_model!r} is the repo being published")
    kind = manifest["kind"]
    name = manifest["name"]
    description = manifest.get("description", "")
    training = manifest.get("training", {})
    perception = manifest.get("perception") or "none"
    run = install_commands(manifest, repo_id)
    if kind == "episodic":
        timing = f"Runs {manifest['duration_s']} s and returns itself to a standing pose."
        if manifest.get("chain"):
            timing += " Holding the button chains another run."
    elif manifest.get("unwind_s") is not None:
        timing = (
            f"Holds until told otherwise; the daemon drives `command.idle` for "
            f"{manifest['unwind_s']} s before handing back to the gait."
        )
    else:
        slot = manifest.get("slot")
        timing = "Runs until told otherwise" + (
            f" — a gait for the `{slot}` slot." if slot else " — a gait, loaded into a policy slot."
        )
    lines = [
        "---",
        "tags:",
        "- microduck",
        "- robotics",
        "- reinforcement-learning",
        "- onnx",
        # The Hub's `robotctl policy load <slot>` snippet reads the slot from this tag.
        *([f"- microduck-slot:{manifest['slot']}"] if manifest.get("slot") else []),
        "library_name: microduck",
        "pipeline_tag: robotics",
        *([f"base_model: {base_model}", "base_model_relation: finetune"] if base_model else []),
        "---",
        "",
        f"# {name}",
        "",
        description,
        "",
        f"A **{kind}** policy for the [microduck](https://github.com/pollen-robotics/microduck) "
        f"({PERCEPTIONS[perception]}-D observation"
        + (f": the {OBS_LEN}-D proprioception + command, then the head ToF's 8×8 frame" if perception == "tof8x8" else "")
        + f", {ACTION_LEN} actions, {ROBOT['control_hz']} Hz). {timing}",
        *(["", "Reads the head's VL53L8CX time-of-flight sensor (`perception: tof8x8`): the daemon "
            "appends the 64 ranges to the observation, so it needs `tofd` running and a daemon with "
            f"`model_api` ≥ {MODEL_API_PERCEPTIVE}."] if perception == "tof8x8" else []),
        "",
        "## Run it on a robot",
        "",
        "```bash",
        run,
        "```",
        "",
        "The observation normalizer is baked into `policy.onnx`; feed raw observations.",
        "`manifest.json` follows schema 2 of the microduck policy manifest "
        "(`docs/policy-manifest.md` in the daemon repo).",
    ]
    if base_model:
        lines += ["", f"A remix of [{base_model}](https://huggingface.co/{base_model})."]
    if training:
        lines += ["", "## Training", ""]
        for key in ("task_id", "repo", "branch", "commit", "run", "checkpoint", "exported"):
            if key in training:
                lines.append(f"- **{key}**: `{training[key]}`")
        if training.get("dirty"):
            lines.append("- exported from a checkout with uncommitted changes")
    return "\n".join(lines) + "\n"


def _check_repo_id(repo_id: str, what: str) -> None:
    owner, _, name = repo_id.partition("/")
    if not owner or not name or "/" in name or repo_id != repo_id.strip():
        raise ManifestError(f"{what} must be a Hub repo id `<user-or-org>/<name>`, not {repo_id!r}")


def dump_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2) + "\n"
