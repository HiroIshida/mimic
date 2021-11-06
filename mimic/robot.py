from abc import ABC, abstractproperty
import os
from typing import List
from typing import Callable
import numpy as np
import tinyfk

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
