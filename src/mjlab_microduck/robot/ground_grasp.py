"""Ground-grasp robot: the all-collisions Microduck with an opening beak, plus the graspable objects.

The CAD export merges the jaw into the head body (``jaw_soft``) and MuJoCo collides meshes by their CONVEX HULLS:
the jaw hull (plate + side arms) and the bottom-shell hull fill the whole mouth cavity, so nothing could ever enter
the mouth. ``add_mouth`` fixes both (measured on the 2026-09 export, head-local frame x = up, y = left, z = back):

- body ``jaw`` on hinge ``passive_mouth`` at the real pivot (the mouth servo on one side and a bearing on the other sit
  on the same axis at the back of the head: direct drive, jaw angle = servo angle; robotd range −5° closed/squeeze ..
  +30° open). Named ``passive_*`` so the BAM actuator regex, the 14-servo obs/reward selectors and the ONNX export all
  skip it — it gets its own position actuator (``MOUTH_ACTUATOR``), driven by a scripted action term, never by the
  policy (the 61-obs / 14-action contract is unchanged).
- the four underside hulls (bottom shell, upper pad, jaw, lower pad) replaced by boxes: ``head_rear_collision``,
  ``upper_beak_collision`` (head) and ``jaw_plate_collision`` (jaw). Named ``*_collision`` so FULL_COLLISION keeps them.
- sites ``lip_up``, ``pad_up`` (head) and ``lip_low``, ``jaw_tip_bottom``, ``pad_low`` (jaw). The pad sites sit
  0.9 cm inside the lips; the grasp point is their midpoint.
"""

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from mjlab_microduck.robot.microduck_constants import (
    HOME_FRAME,
    MICRODUCK_ALLCOLLISIONS_XML,
    actuators,
    name_servo_collision_geoms,
)

MOUTH_JOINT = "passive_mouth"
MOUTH_PIVOT = np.array([0.0037, 0.0, -0.0182])   # head-local: up, left, back
MOUTH_CLOSED_RAD = -0.0873                         # robotd MOUTH_CLOSED (-5°): squeezes whatever is between the pads
MOUTH_OPEN_MAX_RAD = 0.5236                        # robotd MOUTH_OPEN (+30°)
MOUTH_RANGE = (MOUTH_CLOSED_RAD, 0.55)
JAW_MASS = 0.010
_UNDERSIDE_HULLS = ("bottom_head_shell", "soft_mouth_top", "jaw", "jaw_soft")
_JAW_MESHES = ("jaw", "jaw_soft")                  # soft_mouth_top is the UPPER pad: stays on the head


def _uf(u: float, f: float) -> np.ndarray:
    """Head-local position from (up, forward)."""
    return np.array([u, 0.0, -f])


def add_mouth(spec: mujoco.MjSpec, prefix: str = "") -> None:
    """Give the head an opening jaw and box collisions for the beak (see module docstring). Edits ``spec`` in place."""
    head = spec.body(prefix + "jaw_soft")
    jaw = head.add_body(name=prefix + "jaw", pos=MOUTH_PIVOT)
    jaw.mass = JAW_MASS
    jaw.inertia = [4e-6, 4e-6, 4e-6]
    jaw.ipos = _uf(-0.0134, 0.054) - MOUTH_PIVOT
    jaw.explicitinertial = True
    head.mass = head.mass - JAW_MASS
    jaw.add_joint(name=prefix + MOUTH_JOINT, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0],
                  range=list(MOUTH_RANGE), damping=0.02, armature=0.0018, frictionloss=0.005)
    for g in list(head.geoms):
        mesh = g.meshname[len(prefix):] if g.meshname.startswith(prefix) else g.meshname
        if mesh not in _UNDERSIDE_HULLS:
            continue
        if g.contype == 0 and g.conaffinity == 0:          # visual
            if mesh in _JAW_MESHES:
                jaw.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=g.meshname, material=g.material,
                             pos=np.array(g.pos) - MOUTH_PIVOT, quat=g.quat, contype=0, conaffinity=0, group=2)
                spec.delete(g)
        else:                                               # collision hull: replaced by the boxes below
            spec.delete(g)
    # rear underside: forward -0.0385..0.042, up -0.0169..0.0032
    head.add_geom(name=prefix + "head_rear_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=_uf(-0.00685, 0.00175), size=[0.01005, 0.0459, 0.04025], group=3)
    # upper beak (upper pad + front rim of the shell): forward 0.042..0.078, up -0.0092..-0.0013
    head.add_geom(name=prefix + "upper_beak_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=_uf(-0.00525, 0.060), size=[0.00395, 0.0459, 0.018], group=3)
    # jaw plate + lower pad: forward 0.042..0.0777 (the plate centre is cut out behind 0.042 for the neck),
    # up -0.0174..-0.0094 (0.2 mm below the upper box when closed)
    jaw.add_geom(name=prefix + "jaw_plate_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                 pos=_uf(-0.0134, 0.05985) - MOUTH_PIVOT, size=[0.0040, 0.0440, 0.01785], group=3)
    # the jaw sits right under the head-yaw bearing and the neck bracket (parent / grandparent of the head)
    for other in ("yaw_roll_motion", "neck_pitch"):
        ex = spec.add_exclude()
        ex.bodyname1 = prefix + "jaw"
        ex.bodyname2 = prefix + other
    head.add_site(name=prefix + "lip_up", pos=_uf(-0.0092, 0.0770), group=4)
    head.add_site(name=prefix + "pad_up", pos=_uf(-0.0092, 0.0680), group=4)
    jaw.add_site(name=prefix + "lip_low", pos=_uf(-0.0134, 0.0777) - MOUTH_PIVOT, group=4)
    jaw.add_site(name=prefix + "jaw_tip_bottom", pos=_uf(-0.0174, 0.0777) - MOUTH_PIVOT, group=4)
    jaw.add_site(name=prefix + "pad_low", pos=_uf(-0.0094, 0.0680) - MOUTH_PIVOT, group=4)


def get_ground_grasp_spec() -> mujoco.MjSpec:
    """All-collisions robot + the flamingo-v2 hull-phantom excludes + the opening beak."""
    spec = name_servo_collision_geoms(mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML)))
    # MuJoCo collides the hip bracket's convex hull, which fills its U: 0.3-1.8 mm phantom contact with the trunk
    # over the whole hip-roll range (measured 2026-09-01, see get_allcollisions_flamingo_spec on flamingo-allcol).
    for body in ("hip_l", "hip_l_2"):
        ex = spec.add_exclude()
        ex.name = f"trunk_base_{body}_hull_phantom"
        ex.bodyname1 = "trunk_base"
        ex.bodyname2 = body
    # Hip-roll range: the export's +-0.384 rad is conservative. The physical robot was posed at 0.463 rad
    # (flamingo, 2026-09-01) and its untorqued rest pose reads -0.566 rad on the right (duck-control REST_POSITION).
    # A narrower sim stop let run 3 brace against it with targets 1.9 rad past it (robotd does not clamp there).
    for jn in ("left_hip_roll", "right_hip_roll"):
        spec.joint(jn).range = [-0.52, 0.52]
    add_mouth(spec)
    return spec


SERVO_JOINTS = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
                "neck_pitch", "head_pitch", "head_yaw", "head_roll",
                "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")


def ground_grasp_joint_limits() -> dict:
    """{servo joint: (lo, hi)} on the ground-grasp model, in the 14-servo order."""
    m = get_ground_grasp_spec().compile()
    return {jn: tuple(float(v) for v in m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)])
            for jn in SERVO_JOINTS}


# The jaw's own servo, driven by the scripted MouthScheduleAction (action_dim 0), never by the policy. An XL330 at
# firmware kP 200 is ~0.55 N.m/rad (BAM); Rémi's brief says to assume enough beak torque, so it squeezes at 1.0 N.m/rad
# (0.3 N.m on a 1.6 cm block), effort capped at 0.5 N.m (XL330 class).
MOUTH_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(MOUTH_JOINT,),
    stiffness=1.0,
    damping=0.01,
    effort_limit=0.5,
)

# FULL_COLLISION + friction on the beak boxes (condim 4: torsional friction keeps a pinched object from spinning).
GROUND_GRASP_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    condim={
        r"^(left|right)_foot_collision$": 3,
        r"^(upper_beak|jaw_plate)_collision$": 4,
        ".*_collision": 1,
    },
    priority={r"^(left|right)_foot_collision$": 1},
    friction={
        r"^(left|right)_foot_collision$": (1.0,),
        r"^(upper_beak|jaw_plate)_collision$": (1.5, 0.02, 0.0001),
    },
)

MICRODUCK_GROUND_GRASP_ROBOT_CFG = EntityCfg(
    spec_fn=get_ground_grasp_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=HOME_FRAME.pos,
        rot=HOME_FRAME.rot,
        joint_pos={**HOME_FRAME.joint_pos, MOUTH_JOINT: MOUTH_CLOSED_RAD},
        joint_vel={".*": 0.0},
    ),
    collisions=(GROUND_GRASP_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators, MOUTH_ACTUATOR),
        soft_joint_pos_limit_factor=0.9,
    ),
)


# ── Graspable objects ──────────────────────────────────────────────────────────
# One free box per env. Its size, mass and friction are randomised per reset (dr.geom_size / dr.body_mass /
# dr.geom_friction); the nominal geom is a 2.0 x 2.0 x 1.6 cm, 12 g block (a 2x2 Lego brick is 1.6 x 1.6 x 1.1 cm,
# a die 1.6 cm, an AA battery lying 1.45 cm high).
OBJECT_HALF = (0.010, 0.010, 0.008)
OBJECT_MASS = 0.012


def get_grasp_object_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="object", pos=[0.15, 0.0, OBJECT_HALF[2]])
    body.add_freejoint(name="object_free")
    body.add_geom(name="object_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(OBJECT_HALF),
                  mass=OBJECT_MASS, rgba=[0.15, 0.45, 0.95, 1.0], condim=4, friction=[1.4, 0.02, 0.0001])
    return spec


GRASP_OBJECT_CFG = EntityCfg(
    spec_fn=get_grasp_object_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.15, 0.0, OBJECT_HALF[2])),
)
