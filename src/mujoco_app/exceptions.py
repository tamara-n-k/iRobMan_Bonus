"""Application-specific exceptions."""


class CollisionDetectedError(RuntimeError):
    """Raised when the robot collides with an obstacle during execution."""
