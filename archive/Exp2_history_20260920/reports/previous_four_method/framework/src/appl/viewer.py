"""Read-only saved-rollout viewer. FK visualization never steps a simulator."""
from pathlib import Path
import json
import time
import numpy as np
from .io import atomic


def serve(cfg,trace=None,host='127.0.0.1',port=8086,duration=None):
    import viser
    import mujoco
    from PIL import Image
    from .surrogate import Simulator
    from .verification import model_path
    trace=Path(trace) if trace else cfg['output']/'diagnostics/D2/rollouts/1000/trace.jsonl'
    rows=[json.loads(line) for line in trace.read_text().splitlines()]
    if not rows:raise ValueError('No saved physical observations in trace')
    output=cfg['output']/'viewer';output.mkdir(exist_ok=True)
    sim=Simulator(model_path(cfg,output));m=sim.model
    server=viser.ViserServer(host=host,port=port,label='APPL saved rollout — read only')
    server.scene.set_up_direction('+z')
    handles={}
    for index in range(m.ngeom):
        name='/scene/geom_'+str(index);kind=int(m.geom_type[index]);color=tuple((np.clip(m.geom_rgba[index,:3],0,1)*255).astype(np.uint8))
        if kind==int(mujoco.mjtGeom.mjGEOM_BOX):
            handle=server.scene.add_box(name,color=color,dimensions=tuple(2*m.geom_size[index]))
        elif kind==int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh=int(m.geom_dataid[index]);start=m.mesh_vertadr[mesh];fstart=m.mesh_faceadr[mesh]
            handle=server.scene.add_mesh_simple(name,m.mesh_vert[start:start+m.mesh_vertnum[mesh]].copy(),
                m.mesh_face[fstart:fstart+m.mesh_facenum[mesh]].copy(),color=color)
        else:continue
        handles[index]=handle
    path=np.array([r['state']['tcp_pose'][:3] for r in rows])
    if len(path)>1:server.scene.add_line_segments('/trajectory/tcp',np.stack([path[:-1],path[1:]],axis=1),colors=(255,190,30),line_width=2)
    tcp=server.scene.add_frame('/tcp',axes_length=.06,axes_radius=.003)
    server.gui.add_markdown('Saved observations with calibrated robot/scene geometry. **No physics, policy or API execution.** RGB is sampled at 1 Hz; pose trace at 20 Hz.')
    slider=server.gui.add_slider('Recorded frame',min=0,max=len(rows)-1,step=1,initial_value=0)
    play=server.gui.add_checkbox('Play',False)
    details=server.gui.add_markdown('')
    first=next(iter(sorted(trace.parent.glob('frame_*.png'))),None)
    rgb=server.gui.add_image(np.asarray(Image.open(first))) if first else None
    def draw(index):
        row=rows[index];sim.reset(row['state'])
        with server.atomic():
            for geom,handle in handles.items():
                quat=np.zeros(4);mujoco.mju_mat2Quat(quat,sim.data.geom_xmat[geom])
                handle.position=sim.data.geom_xpos[geom].copy();handle.wxyz=quat
            tcp.position=tuple(row['state']['tcp_pose'][:3]);tcp.wxyz=tuple(row['state']['tcp_pose'][3:])
            details.content=f"Step {row.get('step',index)} · skill {row.get('skill','full policy')}\n\n```json\n{json.dumps(row.get('metrics',{}),indent=2)}\n```"
            image_index=((int(row.get('step',index+1))-1)//20)*20+1
            image_path=trace.parent/f'frame_{image_index:04d}.png'
            if rgb is not None and image_path.exists():rgb.image=np.asarray(Image.open(image_path))
    slider.on_update(lambda event:draw(slider.value));draw(0)
    atomic(output/'startup.json',dict(host=host,port=port,trace=str(trace),frames=len(rows),geometry_handles=len(handles),read_only=True,physics_steps=0))
    print(f'Viser listening at http://{host}:{port}; saved trace {trace}',flush=True)
    started=time.monotonic()
    try:
        while duration is None or time.monotonic()-started<duration:
            if play.value:slider.value=(slider.value+1)%len(rows)
            time.sleep(.05)
    finally:server.stop()
