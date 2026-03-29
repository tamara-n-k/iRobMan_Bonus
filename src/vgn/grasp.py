import enum


class Label(enum.IntEnum):
    FAILURE = 0
    SUCCESS = 1


class Grasp:
    """Grasp parameterized as pose of a 2-finger robot hand."""

    def __init__(self, pose, width):
        self.pose = pose
        self.width = width


def to_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation /= voxel_size
    width = grasp.width / voxel_size
    return Grasp(pose, width)


def from_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation *= voxel_size
    width = grasp.width * voxel_size
    return Grasp(pose, width)
