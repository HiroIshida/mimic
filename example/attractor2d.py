import copy
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import attr
from typing import List, Tuple

@dataclass
class State(object):
    x: np.ndarray
    v: np.ndarray

@dataclass
class Attractor(object):
    center: np.ndarray
    k: float = 0.4 
    c: float = 0.2
    eps: float = 0.02

    def propagate(self, state, dt):
        direcion = state.x - self.center
        r = np.linalg.norm(direcion)
        gravitational_factor = (self.k / r**2) if (r > 1e-1) else 0.0
        force = - gravitational_factor * direcion - self.c * state.v + np.random.randn(2) * self.eps
        state.v += force * dt 
        state.x += state.v * dt
        return state

@dataclass
class GoalCondition(object):
    attractor: Attractor
    r: float

    def isInside(self, state):
        if np.linalg.norm(state.v) > 0.4:
            return False
        return np.linalg.norm(state.x - self.attractor.center) < self.r

@dataclass
class SequentialAttractor(object):
    attractors: List[Attractor]
    goal_conditions: List[GoalCondition]

    def propagate(self, state, phase, dt):
        attractor = self.attractors[phase]
        state = attractor.propagate(state, dt) 

        phase_new = phase
        gr = self.goal_conditions[phase]
        is_last_phase = (len(self.goal_conditions)-1 == phase)

        if gr.isInside(state):
            if is_last_phase:
                return state, None
            phase_new += 1
        return state, phase_new

def create_dataset(N_data=100) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    attractor1 = Attractor(np.array([0., 1.]))
    gc1 = GoalCondition(attractor1, 0.2)
    attractor2 = Attractor(np.array([1.5, 1.]))
    gc2 = GoalCondition(attractor2, 0.2)
    seqatr = SequentialAttractor([attractor1, attractor2], [gc1, gc2])

    Xs: List[np.ndarray] = []
    Ps: List[np.ndarray] = []
    for i in tqdm.tqdm(range(N_data)):
        s = State(np.array([0, 0.]), np.array([0.2, 0.]))
        phase = 0

        dt = 0.05
        state_list = [copy.deepcopy(s)]
        partitions = []
        counter = 0
        while phase is not None:
            s, phase_new = seqatr.propagate(s, phase, dt)
            if phase != phase_new:
                partitions.append(counter)
            phase = phase_new
            state_list.append(copy.deepcopy(s))
            counter += 1
        X = np.array([[s.x[0], s.x[1]] for s in state_list])
        Xs.append(X)
        Ps.append(np.array(partitions))
    return Xs, Ps

if __name__=='__main__':
    from mimic.datatype import CommandDataChunk

    project_name = 'attractor2d'
    Xs, _ = create_dataset()
    chunk = CommandDataChunk()
    print(type(Xs))
    for cmd in Xs:
        chunk.push_epoch(cmd)
    chunk.dump(project_name)

    data = np.array(Xs[0])
    plt.plot(data[:, 0], data[:, 1])
    plt.show()
