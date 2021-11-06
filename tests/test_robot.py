from mimic.robot import KukaSpec
import pickle
import numpy as np

def test_robot_spec():
    spec = KukaSpec()
    fksolver = spec.create_fksolver()
    coords = fksolver(np.random.randn(100, 7))
    assert list(coords.shape) == [100, 6]

    spec_str = pickle.dumps(spec)
    spec2 = pickle.loads(spec_str)


