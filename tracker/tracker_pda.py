from .associate import MergeAssociate
from .target_tracker import BayesianTargetTracker


class PDATracker(BayesianTargetTracker, MergeAssociate):
    """PDA tracker."""
