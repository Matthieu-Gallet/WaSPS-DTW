"""
NetCDF data loading functions.
"""

import time
import numpy as np
import xarray as xr

from .series_extraction import extract_lambda_series
from .preprocessing import split_train_test


def load_data(file_path, n_patches=200, predict_length=90, train_ratio=0.8,
              patch_size=4, time_window=4, slice_lat=None, slice_lon=None,
              return_raw_dataset=False, random_seed=42):
    """
    Main function to load and prepare data.

    Args:
        file_path: Path to the NetCDF file
        n_patches: Number of patches to extract
        predict_length: Prediction length
        train_ratio: Train/test ratio
        patch_size: Spatial patch size
        time_window: Temporal window size
        slice_lat: Latitude slice (optional)
        slice_lon: Longitude slice (optional)
        return_raw_dataset: If True, return raw dataset values
        random_seed: Random seed

    Returns:
        tuple: (X_train, Y_train, X_test, Y_test, ds) or (None, None, None, None, None) on error
    """
    try:
        print("Opening data...")
        start_open = time.time()
        
        # Try opening the file with different options
        try:
            ds = xr.open_dataset(file_path, engine='netcdf4', decode_cf=True)
        except Exception as e1:
            print(f"Failed with netcdf4 engine: {e1}")
            try:
                ds = xr.open_dataset(file_path, engine='h5netcdf')
            except Exception as e2:
                print(f"Failed with h5netcdf engine: {e2}")
                raise Exception(f"Unable to open file with available engines: netcdf4={e1}, h5netcdf={e2}")
        
        data = ds['dis06']
        print(f"Time to open NetCDF file: {time.time() - start_open:.2f} seconds")

        if return_raw_dataset:
            X_train, Y_train, X_test, Y_test = split_train_test(
                data.values, train_ratio=train_ratio, predict_length=predict_length, random_seed=random_seed
            )
            return X_train, Y_train, X_test, Y_test, ds

        else:
            print("\nExtracting time series...")
            start_extract = time.time()
            lambda_series_list = extract_lambda_series(
                data, n_patches=n_patches, patch_size=patch_size,
                time_window=time_window, slice_lat=slice_lat,
                slice_lon=slice_lon, random_seed=random_seed
            )
            print(f"Time for patch extraction: {time.time() - start_extract:.2f} seconds")

            if len(lambda_series_list) < 10:
                print("Error: Not enough valid series")
                ds.close()
                return None, None, None, None, None

            print("\nTrain/test split...")
            X_train, Y_train, X_test, Y_test = split_train_test(
                lambda_series_list, train_ratio=train_ratio,
                predict_length=predict_length, random_seed=random_seed
            )

            if X_train is None:
                ds.close()
                return None, None, None, None, None

            return X_train, Y_train, X_test, Y_test, ds

    except Exception as e:
        print(f"Error loading data: {e}")
        # Ensure dataset is closed on error
        try:
            if 'ds' in locals():
                ds.close()
        except Exception:
            pass
        return None, None, None, None, None
