from abc import ABC, abstractproperty
import os
from typing import List

class RobotSpecBase(ABC):
    @abstractproperty
    def featured_link_names(self) -> List[str]: ...
    @abstractproperty
    def joint_names(self) -> List[str]: ...
    @abstractproperty
    def urdf_path(self) -> str: ...

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
