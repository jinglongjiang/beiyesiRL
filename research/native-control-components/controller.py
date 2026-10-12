"""Finite native-control components as proposals, not new physical actions."""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch
from pytorch_mppi.mppi import SpecificActionSampler

path = Path(__file__).resolve().parents[1] / 'progress-contract-v2/controller.py'
spec = importlib.util.spec_from_file_location('component_fixed8', path)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class SafeSampler(SpecificActionSampler):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def sample_trajectories(self, state, info):
        o = self.owner
        common = base.base.Library(o).sample_trajectories(state, info)
        current = state[None].repeat(96,1)
        plans = []
        done = torch.zeros(96,dtype=torch.bool)
        draws = torch.rand((16,96),dtype=torch.float64).numpy()
        for step in range(16):
            expanded = current[:,None,:].expand(-1,80,-1).reshape(-1,5)
            future = o.dynamics(expanded,o.raw.repeat(96,1))
            safe = (o.clearance(future).reshape(96,80)>=.2).numpy()
            safe[done.numpy()] = True
            chosen = []
            for i, mask in enumerate(safe):
                ids, labels, count = o.components(mask)
                if not done[i]:
                    o.queries += 1
                    o.split_queries += int(count>1)
                    o.empty_queries += int(not mask.any())
                if o.proposal=='uniform':
                    index = ids[min(int(draws[step,i]*len(ids)),len(ids)-1)]
                else:
                    index = o.balanced(ids,labels,count,draws[step,i])
                chosen.append(index)
            action = o.raw[chosen]
            plans.append(action)
            current = o.dynamics(current,action)
            done |= torch.linalg.vector_norm(current[:,:2]-o.goal,dim=-1)<o.radius
        return torch.cat([common,torch.stack(plans,1)])


class Controller(base.Controller):
    def __init__(self, arm, policy):
        super().__init__('hold8',policy)
        self.proposal = arm
        self.queries = self.split_queries = self.empty_queries = 0
        raw = self.raw.numpy()
        speeds = np.round(np.linalg.norm(raw,axis=1),10)
        headings = np.round(np.mod(np.arctan2(raw[:,1],raw[:,0]),2*np.pi),10)
        ss, hh = sorted(set(speeds)), sorted(set(headings))
        assert len(ss)==5 and len(hh)==16
        lookup = {(ss.index(s),hh.index(h)): i for i,(s,h) in enumerate(zip(speeds,headings))}
        assert len(lookup)==80
        adjacency = np.zeros((80,80),bool)
        for (speed,heading),i in lookup.items():
            for key in [(speed-1,heading),(speed+1,heading),(speed,(heading-1)%16),(speed,(heading+1)%16)]:
                if key in lookup:
                    adjacency[i,lookup[key]] = True
        assert np.array_equal(adjacency,adjacency.T)
        self.graph = csr_matrix(adjacency)
        self.cache = {}
        self.optimizer.specific_action_sampler = SafeSampler(self)

    def components(self, mask):
        key = np.packbits(mask).tobytes()
        if key not in self.cache:
            ids = np.flatnonzero(mask)
            if not len(ids):
                ids = np.arange(80)
            count, labels = connected_components(self.graph[ids][:,ids],directed=False)
            self.cache[key] = (ids,labels,count)
        return self.cache[key]

    @staticmethod
    def balanced(ids,labels,count,draw):
        component = min(int(draw*count),count-1)
        members = ids[labels==component]
        fraction = draw*count-component
        return int(members[min(int(fraction*len(members)),len(members)-1)])

    def select(self, *args, **kwargs):
        before = self.queries,self.split_queries,self.empty_queries
        index, diag = super().select(*args,**kwargs)
        diag.update(sample_queries=self.queries-before[0],split_queries=self.split_queries-before[1],
                    empty_queries=self.empty_queries-before[2])
        return index,diag
