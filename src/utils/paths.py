"""
Path configuration utilities.
"""

from pathlib import Path


def setup_paths(base_dir: str = None):
    """
    Configure paths for data and figures.

    Args:
        base_dir: Base directory (default: script directory)

    Returns:
        Dictionary with configured paths
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent

    base_path = Path(base_dir)

    # Default paths for data
    data_paths = [
        Path("/home/mgallet/Documents/Dataset/RIVER_DISCHARGES/8e4f4b6e5b63588b0b2a86786684fa64/data_version-5.nc"),
        Path("/home/mgallet/Documents/Dataset/RIVER_DISCHARGES/7707e06702921ffa66a8559ac266887d/data_version-5.nc"),
        base_path / ".." / "Downloads" / "8e4f4b6e5b63588b0b2a86786684fa64" / "data_version-5.nc",
        base_path / ".." / "Downloads" / "7707e06702921ffa66a8559ac266887d" / "data_version-5.nc"
    ]

    # Find the first file that exists
    data_file = None
    for path in data_paths:
        if path.exists():
            data_file = path
            break

    if data_file is None:
        raise FileNotFoundError("No data file found at default locations")

    results_dir = base_path / "results" / "geographic_barycenter"
    results_dir.mkdir(parents=True, exist_ok=True)

    return {
        'data_file': str(data_file),
        'figures_dir': str(results_dir),
        'base_dir': str(base_path)
    }
