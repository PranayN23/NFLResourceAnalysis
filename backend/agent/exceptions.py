class UngradablePlayerError(Exception):
    """Exception raised when a player cannot be graded due to sparse data (NaN results)."""
    pass
