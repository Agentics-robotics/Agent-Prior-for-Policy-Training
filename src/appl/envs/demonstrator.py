"""Fixed motion-planning demonstrator for data acquisition, NOT an APPL policy.

Uses contact and native arm/gripper actions throughout execution. Object and
drawer poses are never overwritten after reset; failures remain failed attempts.
"""

import numpy as np
import sapien
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from . import scene as cad


def collect_episode(env, recorder):
    native = env.unwrapped
    planner = PandaArmMotionPlanningSolver(
        recorder,
        vis=False,
        debug=False,
        base_pose=native.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )

    def move(point, closing=(0, 1, 0), approaching=(0, 0, -1), free=False):
        pose = native.agent.build_grasp_pose(
            np.array(approaching, dtype=float),
            np.array(closing, dtype=float),
            np.asarray(point),
        )
        if free:
            result = planner.planner.plan_qpos_to_pose(
                np.r_[pose.p, pose.q],
                native.agent.robot.get_qpos()[0].cpu().numpy(),
                time_step=native.control_timestep,
                wrt_world=True,
            )
        else:
            result = planner.planner.plan_screw(
                np.r_[pose.p, pose.q],
                native.agent.robot.get_qpos()[0].cpu().numpy(),
                time_step=native.control_timestep,
            )
        if result["status"] != "Success":
            raise RuntimeError(
                "Fixed demonstrator motion planning failed: " + str(result["status"])
            )
        return planner.follow_path(result, refine_steps=8)

    def current(actor):
        return actor.pose.p[0].cpu().numpy()

    def phase(name):
        recorder.mark(name)

    try:
        phase("open_drawer")
        handle = np.array(cad.DRAWER_ORIGIN) + cad.HANDLE_LOCAL
        move(handle + [0, 0, 0.23], (1, 0, 0), free=True)
        move(handle, (1, 0, 0))
        planner.close_gripper(t=12)
        move(handle + [-cad.TRAVEL, 0, 0], (1, 0, 0))
        if float(native.drawer.get_qpos()[0, 0]) < 0.26:
            raise RuntimeError("Drawer was not opened by physical contact")
        planner.open_gripper(t=10)
        move(handle + [-cad.TRAVEL, 0, 0.24], (1, 0, 0))
        if float(native.drawer.get_qpos()[0, 0]) < 0.26:
            raise RuntimeError(
                "Drawer lost open state during collision-free withdrawal"
            )
        phase("take_red_out")
        red = current(native.red)
        move(red + [0, 0, 0.25], free=True)
        move(red)
        planner.close_gripper(t=12)
        move(red + [0, 0, 0.26])
        if current(native.red)[2] < red[2] + 0.18:
            raise RuntimeError("Red pickup did not lift the object through contact")
        goal = np.array(cad.OUTSIDE_GOAL)
        move(goal + [0, 0, 0.28])
        move(goal + [0, 0, 0.003])
        planner.open_gripper(t=12)
        move(goal + [0, 0, 0.28])
        phase("put_blue_in")
        blue = current(native.blue)
        move(blue + [0, 0, 0.26])
        move(blue)
        planner.close_gripper(t=12)
        move(blue + [0, 0, 0.26])
        if current(native.blue)[2] < blue[2] + 0.18:
            raise RuntimeError("Blue pickup did not lift the object through contact")
        drawer = float(native.drawer.get_qpos()[0, 0])
        goal = (
            np.array(cad.DRAWER_ORIGIN)
            + [-drawer, 0, 0]
            + np.array(cad.INSIDE_GOAL_LOCAL)
        )
        move(goal + [0, 0, 0.26])
        move(goal + [0, 0, 0.003])
        planner.open_gripper(t=12)
        move(goal + [0, 0, 0.26])
        phase("settle")
        planner.open_gripper(t=20)
        result = {k: bool(v[0]) for k, v in native.evaluate().items()}
        if not result["success"]:
            raise RuntimeError(
                "Fixed demonstrator failed task predicates: " + str(result)
            )
        return result
    finally:
        planner.close()
