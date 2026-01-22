#!/usr/bin/env python3
"""
Script to extract hydrological regimes and coordinates of Explore2 stations
From the individual diagnostic sheets of the dataset doi:10.57745/LNTOKL

Author: Script generated for automatic extraction
Date: 2026-01-21
"""

import requests
import re
import pandas as pd
from pathlib import Path
import time
from PyPDF2 import PdfReader
import io
from tqdm import tqdm
import unicodedata

# Configuration
DATASET_URL = "https://entrepot.recherche.data.gouv.fr/api/datasets/:persistentId/?persistentId=doi:10.57745/LNTOKL"
OUTPUT_CSV = "stations_regimes_explore2.csv"
DOWNLOAD_DIR = Path("fiches_diagnostic")

# Mapping of hydrological regimes
REGIMES = {
    "Pluvial modérément contrasté": {"code": "PM"},
    "Pluvial contrasté": {"code": "PC"},
    "Pluvio-nival": {"code": "PN"},
    "Nivo-pluvial": {"code": "NP"},
    "Nival & nivo-glaciaire": {"code": "NN"},
    "Nival": {"code": "NG"},
    "nivo-glaciaire": {"code": "NG"},
}

def validate_station_code(code_station):
    """
    Check that the station code is composed of one uppercase letter followed by 9 digits
    """
    return re.match(r'^[A-Z]\d{9}$', code_station) is not None

def get_dataset_files():
    """
    Retrieve the list of PDF files from the Dataverse dataset
    """
    try:
        response = requests.get(DATASET_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        files = []
        if 'data' in data and 'latestVersion' in data['data']:
            version_files = data['data']['latestVersion']['files']
            
            for file_info in version_files:
                file_data = file_info.get('dataFile', {})
                filename = file_data.get('filename', '')
                file_id = file_data.get('id')
                
                # Keep only diagnostic sheets (format: AXXXXXXXXX_diagnostic_datasheet.pdf)
                if filename.endswith('_diagnostic_datasheet.pdf'):
                    code_station = filename.split('_')[0]
                    if validate_station_code(code_station):
                        files.append({
                            'filename': filename,
                            'file_id': file_id,
                            'code_station': code_station
                        })
        
        print(f"OK {len(files)} diagnostic sheets found")
        return files
    
    except Exception as e:
        print(f"Error retrieving files: {e}")
        return []

def download_pdf(file_id, filename):
    """
    Download a PDF file from Dataverse
    """
    url = f"https://entrepot.recherche.data.gouv.fr/api/access/datafile/{file_id}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return io.BytesIO(response.content)
    
    except Exception as e:
        print(f"  Warning: Download error {filename}: {e}")
        return None

def extract_info_from_pdf(pdf_stream, code_station):
    """
    Extract information from a diagnostic PDF sheet
    
    Returns a dictionary with:
    - code_station
    - station_name
    - hydrological_regime
    - regime_code
    - X (Lambert93)
    - Y (Lambert93)
    - area
    - altitude
    - hydro_region
    """
    try:
        reader = PdfReader(pdf_stream)
        
        # Extract text from all pages
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Extract hydrological regime
        # Format: "Pluvial contrasté 5" (with a number at the end)
        hydrological_regime = None
        regime_code = None
        
        for regime_name in REGIMES.keys():
            # Search for the regime followed by a number
            pattern = rf'{re.escape(regime_name)}\s+\d+'
            if re.search(pattern, text, re.IGNORECASE):
                hydrological_regime = regime_name
                regime_code = REGIMES[regime_name]["code"]
                break
        
        # Normalize text for other extractions
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        
        # Extract station code and name (first line)
        # Format: "A105003001 - L'Ill à Altkirch"
        match_station = re.search(r'^([A-Z]\d+)\s*-\s*(.+?)(?:\n|$)', text, re.MULTILINE)
        station_name = match_station.group(2).strip() if match_station else "Unknown"
        
        # Extract Lambert93 coordinates
        # Format: "X = 1018591 m (Lambert93)"
        match_x = re.search(r'X\s*=\s*([\d\s]+)\s*m.*Lambert', text)
        match_y = re.search(r'Y\s*=\s*([\d\s]+)\s*m.*Lambert', text)
        
        x_lambert93 = None
        y_lambert93 = None
        
        if match_x:
            x_lambert93 = float(match_x.group(1).replace(' ', ''))
        if match_y:
            y_lambert93 = float(match_y.group(1).replace(' ', ''))
        
        # Extract area
        match_superficie = re.search(r'Superficie\s*:\s*([\d.]+)', text, re.IGNORECASE)
        superficie = float(match_superficie.group(1)) if match_superficie else None
        
        # Extract altitude
        match_altitude = re.search(r'Altitude\s*:\s*([\d.]+)\s*m', text)
        altitude = float(match_altitude.group(1)) if match_altitude else None
        
        # Extract hydrographic region
        match_region = re.search(r'Region hydrographique\s*:\s*(.+?)(?:\n|$)', text)
        region_hydro = match_region.group(1).strip() if match_region else None
        
        # Extract start and end dates
        match_dates = re.findall(r'Date(?: de)? (debut|defin|fin)\s*:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        date_debut = None
        date_fin = None
        for label, date in match_dates:
            if label == 'debut':
                date_debut = date.strip()
            elif label in ('defin', 'fin'):
                date_fin = date.strip()
        
        return {
            'code_station': code_station,
            'station_name': station_name,
            'hydrological_regime': hydrological_regime,
            'regime_code': regime_code,
            'X_Lambert93': x_lambert93,
            'Y_Lambert93': y_lambert93,
            'superficie_km2': superficie,
            'altitude_m': altitude,
            'region_hydrographique': region_hydro,
            'date_debut': date_debut,
            'date_fin': date_fin
        }
    
    except Exception as e:
        print(f"  Warning: Extraction error {code_station}: {e}")
        return None

def main():
    """
    Main function
    """
    print("=" * 80)
    print("EXTRACTION OF HYDROLOGICAL REGIMES - EXPLORE2 PROJECT")
    print("Dataset: Regional diagnostic sheets (doi:10.57745/LNTOKL)")
    print("=" * 80)
    print()
    
    # Create download directory
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    
    csv_path = Path(OUTPUT_CSV)
    existing_df = None
    if csv_path.exists() and csv_path.stat().st_size > 0:
        print("Existing CSV found. Loading and cleaning...")
        existing_df = pd.read_csv(csv_path, encoding='utf-8')
        # Clean invalid station codes
        existing_df = existing_df[existing_df['code_station'].apply(validate_station_code)]
        print(f"Valid stations in existing CSV: {len(existing_df)}")
        
        # Check if all required columns are present
        required_columns = ['code_station', 'station_name', 'hydrological_regime', 'regime_code', 
                           'X_Lambert93', 'Y_Lambert93', 'superficie_km2', 'altitude_m', 
                           'region_hydrographique', 'date_debut', 'date_fin']
        if not all(col in existing_df.columns for col in required_columns):
            print("Missing columns in CSV. Complete reprocessing.")
            existing_df = None
    
    # Retrieve file list
    files = get_dataset_files()
    
    if not files:
        print("No files found. Check dataset access.")
        return
    
    # If existing CSV, identify stations to retry
    to_retry = []
    if existing_df is not None:
        existing_codes = set(existing_df['code_station'])
        all_codes = set(f['code_station'] for f in files)
        # Missing stations or with missing regime
        missing_or_no_regime = existing_df[existing_df['hydrological_regime'].isna()]['code_station'].tolist()
        to_retry = [f for f in files if f['code_station'] in missing_or_no_regime or f['code_station'] not in existing_codes]
        print(f"Stations to retry: {len(to_retry)}")
    else:
        to_retry = files
    
    if not to_retry:
        print("All stations already processed.")
        return
    
    # Extract information from each sheet
    print(f"\nExtracting information from {len(to_retry)} sheets...\n")
    
    results = []
    errors = []
    
    for file_info in tqdm(to_retry[:2], desc="Processing sheets"):
        code_station = file_info['code_station']
        filename = file_info['filename']
        file_id = file_info['file_id']
        
        # Download PDF
        pdf_stream = download_pdf(file_id, filename)
        
        if pdf_stream is None:
            errors.append(code_station)
            continue
        
        # Extract information
        info = extract_info_from_pdf(pdf_stream, code_station)
        
        if info is not None:
            results.append(info)
        else:
            errors.append(code_station)
        
        # Pause to not overload the server
        time.sleep(0.1)
    
    # Combine with existing data
    if existing_df is not None:
        # Remove old entries for retried stations
        retry_codes = set(f['code_station'] for f in to_retry)
        existing_df = existing_df[~existing_df['code_station'].isin(retry_codes)]
        # Add new ones
        new_df = pd.DataFrame(results)
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("EXTRACTION RESULTS")
    print("=" * 80)
    print(f"Stations successfully processed: {len(results)}")
    print(f"Stations in error: {len(errors)}")
    
    if len(df) > 0:
        print(f"\nDistribution of hydrological regimes:")
        print(df['hydrological_regime'].value_counts().to_string())
        
        # Statistics on missing data
        print(f"\nMissing coordinates: {df['X_Lambert93'].isna().sum()}")
        print(f"Missing altitudes: {df['altitude_m'].isna().sum()}")
        
        # Save to CSV
        csv_path = Path(OUTPUT_CSV)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\nData saved: {csv_path.absolute()}")
        
        # Show preview
        print(f"\nData preview (first 5 lines):")
        print(df.head().to_string())
    
    else:
        print("No data extracted")
    
    if errors:
        print(f"\nStations in error ({len(errors)}):")
        print(", ".join(errors[:10]))
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more")
    
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
