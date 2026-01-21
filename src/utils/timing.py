"""
Timing utilities for performance measurement.
"""

import time


def print_timing(start_time: float, operation: str) -> float:
    """
    Measure and display elapsed time for an operation.

    Args:
        start_time: Start time (return value of time.time())
        operation: Description of the operation

    Returns:
        New start time for chaining
    """
    elapsed = time.time() - start_time
    print(f"Time for {operation}: {elapsed:.2f} seconds")
    return time.time()
