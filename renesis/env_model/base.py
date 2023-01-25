from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    The base model
    """

    def __init__(self):
        self.steps = 0

    @property
    @abstractmethod
    def action_space(self):
        """Returns the space of action."""
        return None

    @property
    @abstractmethod
    def observation_space(self):
        """Returns the space of observation."""
        return None

    @abstractmethod
    def reset(self):
        """Reset model"""
        raise NotImplementedError()

    @abstractmethod
    def is_finished(self):
        """Returns True if there is no more changes to be applied."""
        raise NotImplementedError()

    @abstractmethod
    def is_robot_empty(self):
        """Returns True if robot has at no voxels."""
        raise NotImplementedError()

    def step(self, action):
        """Change the robot design"""
        raise NotImplementedError()

    def observe(self):
        """
        Observe and get view
        """
        raise NotImplementedError()

    def get_robot(self):
        """
        Returns the current robot, in voxels.

        Returns:
            sizes:
            A tuple of size containing (x, y, z)

            representation:
            A list of tuples of length z (voxel max height), each tuple is of form
            (material, amplitude, frequency, phase shift), and each element in tuple
            is a list of length x*y, where x and y are the bounding box sizes
            of all voxels.
        """
        raise NotImplementedError

    def get_state_data(self):
        """
        Returns any additional state data to be saved with robots
        """
        return None
