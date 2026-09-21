"""Fixed geometry, release and sustained stability; never candidate-controlled."""
import numpy as np
from . import scene


def extent_wxyz(q, half):
    w,x,y,z = np.asarray(q)/np.linalg.norm(q)
    rotation = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    return np.abs(rotation) @ np.full(3,half)


def measure(env, state, rules, speeds=None):
    red=np.asarray(state['red_pose'][:3]); blue=np.asarray(state['blue_pose'][:3])
    drawer=float(state['drawer_position'][0]); tcp=np.asarray(state['tcp_pose'][:3])
    red_extent=extent_wxyz(state['red_pose'][3:],rules['object_half_m'])
    blue_extent=extent_wxyz(state['blue_pose'][3:],rules['object_half_m'])
    center=np.asarray(scene.DRAWER_ORIGIN)+[-drawer,0,0]
    if speeds is None:
        u=env.unwrapped;speeds={}
        for name,actor in [('red',u.red),('blue',u.blue)]:
            speeds[name+'_linear_speed']=float(actor.linear_velocity[0].norm())
            speeds[name+'_angular_speed']=float(actor.angular_velocity[0].norm())
    red_inside=bool(np.all(np.abs(red[:2]-scene.OUTSIDE_GOAL[:2])+red_extent[:2] <= rules['red_pad_half_m']) and .014 < red[2] < .031)
    blue_inside=bool(np.all(np.abs((blue-center)[:2])+blue_extent[:2] <= [.172,.182]) and .053 < blue[2] < .074)
    open_hand=bool(min(state['qpos'][7:]) >= rules['finger_open_m'])
    red_clear=bool(np.linalg.norm(tcp-red) >= rules['tcp_clearance_m'])
    blue_clear=bool(np.linalg.norm(tcp-blue) >= rules['tcp_clearance_m'])
    stable={name: speeds[name+'_linear_speed'] < rules['linear_speed_m_s'] and speeds[name+'_angular_speed'] < rules['angular_speed_rad_s'] for name in ('red','blue')}
    return dict(drawer_open=drawer>rules['drawer_open_m'],red_on_pad=red_inside,blue_inside=blue_inside,
        gripper_released=open_hand,red_clear=red_clear,blue_clear=blue_clear,
        handoff_clear=bool(tcp[2]>.28),
        red_stable=stable['red'],blue_stable=stable['blue'],**speeds)


class SuccessTracker:
    def __init__(self,rules):
        self.rules=rules; self.streaks=dict(open_drawer=0,move_red=0,move_blue=0,task=0)
        self.last={}; self.step=0

    def update(self,measurement):
        m=measurement
        released=m['gripper_released']
        red=m['red_on_pad'] and m['red_stable'] and released and m['red_clear']
        blue=m['blue_inside'] and m['blue_stable'] and released and m['blue_clear']
        active=dict(open_drawer=m['drawer_open'] and released and m['red_clear'] and m['blue_clear'] and m['handoff_clear'],
                    move_red=red and m['handoff_clear'],move_blue=blue and m['handoff_clear'],task=m['drawer_open'] and red and blue)
        for k,v in active.items():self.streaks[k]=self.streaks[k]+1 if v else 0
        self.step+=1
        self.last=dict(**m,streaks=dict(self.streaks),success=self.streaks['task']>=self.rules['sustained_steps'])
        return self.last

    def skill_succeeded(self,skill):
        return self.streaks[skill]>=self.rules['sustained_steps']
