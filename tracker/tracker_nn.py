from .associate import PruneAssociate
from .target_tracker import BayesianTargetTracker


class NNTracker(BayesianTargetTracker, PruneAssociate):
    """Nearest neighbor tracker."""
