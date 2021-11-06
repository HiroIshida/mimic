from abc import ABC, abstractproperty
import os
from typing import List
from typing import Callable
from typing import Optional
import numpy as np
import tinyfk
import math

class RobotSpecBase(ABC):
    @abstractproperty
    def featured_link_names(self) -> List[str]: ...
    @abstractproperty
    def joint_names(self) -> List[str]: ...
    @abstractproperty
    def urdf_path(self) -> str: ...

    def create_fksolver(self) -> Callable[[np.ndarray], np.ndarray]:
        kin_solver = tinyfk.RobotModel(self.urdf_path)
        link_ids = kin_solver.get_link_ids(self.featured_link_names)
        joint_ids = kin_solver.get_joint_ids(self.joint_names)
        def fksolver(angle_vectors: np.ndarray) -> np.ndarray:
            assert angle_vectors.ndim == 2
            coords, _ = kin_solver.solve_forward_kinematics(
                    angle_vectors, link_ids, joint_ids, with_rot=True)
            coords = coords.reshape((-1, len(self.featured_link_names) * 6))
            return coords
        return fksolver

    @property
    def n_joint(self) -> int: return len(self.joint_names)

    @property
    def n_out(self) -> int: return len(self.featured_link_names) * 6

    def sample_from_cspace(self, n_sample: Optional[int]=None) -> np.ndarray:
        kin_solver = tinyfk.RobotModel(self.urdf_path)
        joint_ids = kin_solver.get_joint_ids(self.joint_names)
        joint_limits = kin_solver.get_joint_limits(joint_ids)

        for i in range(len(joint_limits)):
            if joint_limits[i][0] == None:
                joint_limits[i][0] = -math.pi * 1.5
                joint_limits[i][1] = math.pi * 1.5
        lowers = np.array([limit[0] for limit in joint_limits])
        uppers = np.array([limit[1] for limit in joint_limits])

        if n_sample is None:
            n_sample = 8 ** self.n_joint
        points = np.random.random((n_sample, self.n_joint)) * (uppers - lowers) + lowers
        return points

class KukaSpec(RobotSpecBase):
    @property
    def featured_link_names(self): return ['lbr_iiwa_link_7']
    @property
    def joint_names(self): return ['lbr_iiwa_joint_{}'.format(i+1) for i in range(7)]
    @property
    def urdf_path(self):
        try:
            import pybullet_data
        except:
            raise RuntimeError('pybullet must be installed')
        pbdata_path = pybullet_data.getDataPath()
        urdf_path = os.path.join(pbdata_path, 'kuka_iiwa', 'model.urdf')
        return urdf_path
