# Round 2 geometry audit

Runtime model inventory: `artifacts/round2/model_audit.json`. The implementation is `src/round2/geometry.py`. Installed pinned source and XML, not remembered benchmark conventions, determine the following mappings.

| Task | fixed root | moving body | native joint | finite range | observation handle |
|---|---|---|---|---|---|
| drawer | drawer | drawer_link | goal_slidey | [-.16,0] m | body xpos + body xmat @ [0,-.16,0] |
| door | door | door_link | doorjoint | [-2,0] rad | handle geom xpos |

Native drawer body quaternion is WXYZ, whereas native door geom quaternion is XYZW. Round 2 consistently uses XYZW in all current and historical observations, converted from actual matrices, with nearest previous sign. Rotation only changes the cabinet stationary root. Internal joint axis, stationary case, moving panel and all handle visual/collision geoms remain in the same parent tree.

Drawer handle contact maps the three collision capsules on drawer_link at local positions [-.05,-.12,0], [0,-.15,0], [.05,-.12,0]. The observation point is .01 m in front of the main capsule center, matching the original interaction-point convention. Door handle contact maps the four cylinder collision geoms on door_link with local x>.3; hinge cylinders and panel are excluded. Gripper geoms are collidable descendants of leftclaw/rightclaw/leftpad/rightpad. Numeric IDs are emitted with each completed evaluation and source inventory; initial robot self-contacts are not hand-handle contact. This maps the colliders rather than relying on visual mesh names whose collision masks can be zero.

Fresh MjSpec compilation at every reset avoids relying on unsafe mutations of a static body's compiled collision structures. MuJoCo documents that static body position/orientation changes can invalidate the collision BVH ([official model-change documentation](https://mujoco.readthedocs.io/en/latest/programming/simulation.html)); the pinned version's state and forward APIs are documented in [MuJoCo 3.3 simulation documentation](https://mujoco.readthedocs.io/en/3.3.0/programming/simulation.html). Recompilation alone did not fix the initial drawer failure: contact diagnostics identified real penetration of RetainingWall geom5 into drawer back/side/bottom colliders, reaching about .024 m at50 degrees. At35 degrees zero action could produce progress >.9. These records are invalid calibration evidence, not useful demonstrations.

Only the rear accessory retaining rail is moved: local y=.39 to .55 under body RetainingWall at y=.6, hence world y=.99 to1.15. The same change is compiled for both tasks at all angles, including zero. Robot/tabletop/gravity remain native. Other rails remain in place. Repaired final calibration includes 100 zero-action steps per each of57 states/task, requiring max absolute progress <.02 and no initial mechanism penetration deeper than .002 m. Drawer and door passed tier A.

Progress is based on mechanism q, never a world-axis proxy. Goal FK uses a separate MjData with copied current qpos and q_goal=.8*q_open; mj_forward on that data gives the handle goal and quaternion without changing real integration state. Native reset history and target are replaced only after the rotated state has been constructed. The wrapper maintains current/previous native blocks explicitly and appends constant episode yaw metadata.

Calibration revisions, all retained in `calibration_rollouts.json`: v1 naive rotation; v2 recompiled static models, still rail collision; v3 relocated rail with original canonical experts; v4 door tangent expert (approach [.01,.02,0], gain25); v5 slowed drawer after progress .35 and tried door front approach [0,-.01,0]; v6 retained drawer v5 and restored door [.01,.02,0] with approach gain10. V5 drawer and v6 door pass the highest tier, without policy scores. Later calibration versions deliberately reuse calibration base seeds to diagnose physical controller changes; none of those seeds is a train/dev/test seed.
