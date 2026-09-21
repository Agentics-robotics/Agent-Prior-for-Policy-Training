"""Finite initial-layout coverage; the empirical hull is not the ID definition."""
import numpy as np
from scipy.spatial import Delaunay
from appl.io import ROOT, read, atomic
from .run import BASE


def xy(state):
    return np.array(state['red_pose'][:2] + state['blue_pose'][:2])


def main():
    states=read(ROOT/'experiments/exp2/paper/initial_states.json')['states'];out={}
    for task,entry in read(BASE/'plan.json')['tasks'].items():
        train=np.array([xy(read(v['path'])['observations'][0]['state']) for v in entry['demonstrations'].values()])
        center=train.mean(0);_,singular,basis=np.linalg.svd(train-center,full_matrices=False)
        rank=int((singular>1e-8).sum());basis=basis[:rank].T
        hull=Delaunay((train-center)@basis)
        values={}
        for condition in ['ID','OOD']:
            test=np.array([xy(r['state']) for r in states if r['task']==task and r['condition']==condition])
            projected=(test-center)@basis
            residual=np.linalg.norm((test-center)-projected@basis.T,axis=1)
            inside=(hull.find_simplex(projected)>=0)&(residual<1e-7)
            nearest=np.linalg.norm(test[:,None]-train[None],axis=-1).min(1)
            values[condition]=dict(count=len(test),inside_training_coordinate_box=int(((test>=train.min(0))&(test<=train.max(0))).all(1).sum()),
                inside_empirical_convex_hull=int(inside.sum()),nearest_training_layout_L2_m=dict(min=float(nearest.min()),median=float(np.median(nearest)),max=float(nearest.max())))
        out[task]=dict(affine_rank=rank,training_red_blue_xy_min=train.min(0).tolist(),training_red_blue_xy_max=train.max(0).tolist(),conditions=values)
    atomic(BASE/'initial_support.json',dict(tasks=out,note='ID is the predeclared reset distribution, not the finite sample convex hull. Unstack uses shared XY and is rank 2; hulls use an SVD projection without artificial jitter. L2 combines both objects.',runtime_API_requests=0))
    print(out)


if __name__=='__main__':main()
