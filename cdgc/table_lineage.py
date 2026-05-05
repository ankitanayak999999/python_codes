# ============================================================
# CDGC Lineage Export
# Author:
# Date: 2026-05-05
# Description: Extract table lineage and column list
#              from CDGC using REST API
# Version: 4.0 - Functional
# ============================================================

import pandas as pd
import requests
import configparser
import logging
import os
import threading
import concurrent.futures
from datetime import datetime

# ─── GLOBALS ────────────────────────────────────────────────
lock           = threading.Lock()
jwt_lock       = threading.Lock()
progress_lock  = threading.Lock()

table_lineage_data = []
column_list_data   = []
error_log_data     = []

headers            = {}
session_id         = None
org_id             = None
jwt_url            = None

total_assets       = 0
processed_count    = 0
success_count      = 0
failed_count       = 0

logger             = None
timestamp          = datetime.now().strftime(
                     "%Y%m%d_%H%M%S_%f")


# ============================================================
# AUTH FUNCTIONS
# ============================================================

def get_session_id(base_url, username, password):
    """Get session ID from IDMC login API"""
    try:
        url = f"{base_url}/ma/api/v2/user/login"
        payload = {
            "username": username,
            "password": password
        }
        resp = requests.post(
            url, json=payload,
            verify=False, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        _session_id = data.get(
            'userInfo', {}).get('sessionId')
        server_url  = data.get(
            'userInfo', {}).get('serverUrl')
        _org_id     = data.get(
            'userInfo', {}).get('orgId')

        logger.info("Session ID obtained successfully!")
        return _session_id, server_url, _org_id

    except Exception as e:
        logger.error(f"Session ID failed: {str(e)}")
        raise


def get_jwt_token(_session_id, _jwt_url):
    """Get JWT token using session ID"""
    try:
        _headers = {
            "cookie"         : f"USER_SESSION="
                               f"{_session_id}",
            "IDS-SESSION-ID" : _session_id
        }
        resp = requests.get(
            _jwt_url,
            headers=_headers,
            verify=False, timeout=60)
        resp.raise_for_status()

        jwt_token = resp.json()["jwt_token"]
        logger.info("JWT Token obtained successfully!")
        return jwt_token

    except Exception as e:
        logger.error(f"JWT Token failed: {str(e)}")
        raise


def refresh_jwt_token():
    """
    Refresh JWT token when expired
    Thread safe
    """
    global headers

    with jwt_lock:
        try:
            logger.warning("JWT expired - Refreshing...")
            new_jwt = get_jwt_token(session_id, jwt_url)
            headers["Authorization"] = f"Bearer {new_jwt}"
            logger.info("JWT refreshed successfully!")
        except Exception as e:
            logger.error(f"JWT Refresh failed: {str(e)}")
            raise


# ============================================================
# API FUNCTIONS
# ============================================================

def get_table_lineage(asset_id, iics_api_url,
                      lineage_distance):
    """
    Call CDGC lineage + hierarchy API
    Auto refreshes JWT on 401
    Returns response JSON
    """
    url = (
        f"{iics_api_url}"
        f"/data360/search/v1/assets"
        f"/{asset_id}"
        f"?scheme=internal"
        f"&segments=lineage"
        f",lineage-direction:all"
        f",lineage-distance:{lineage_distance}"
        f",hierarchy:all"
    )

    resp = requests.get(
        url,
        headers=headers,
        verify=False,
        timeout=300)

    # JWT expired - refresh and retry once
    if resp.status_code == 401:
        logger.warning(
            f"401 for asset: {asset_id} "
            f"- Refreshing JWT...")
        refresh_jwt_token()

        # Retry with new token
        resp = requests.get(
            url,
            headers=headers,
            verify=False,
            timeout=300)

    if resp.status_code != 200:
        try:
            error_msg = resp.json().get(
                'message', resp.text)
        except:
            error_msg = resp.text
        raise Exception(
            f"HTTP_{resp.status_code} | {error_msg}")

    return resp.json()


# ============================================================
# PARSE FUNCTIONS
# ============================================================

def parse_lineage(asset_id, asset_name, response):
    """Parse lineage section from API response"""
    rows = []
    lineage = response.get('lineage', [])

    for lin in lineage:
        direction = lin.get('direction', '')
        for hop in lin.get('hops', []):
            distance = hop.get('distance', '')
            for item in hop.get('items', []):
                rows.append({
                    'ASSET_ID'      : asset_id,
                    'ASSET_NAME'    : asset_name,
                    'DIRECTION'     : direction,
                    'DISTANCE'      : distance,
                    'FROM_NAME'     : item.get(
                        'from', ''),
                    'FROM_IDENTITY' : item.get(
                        'fromIdentity', ''),
                    'FROM_TYPE'     : item.get(
                        'fromType', ''),
                    'FROM_LOCATION' : item.get(
                        'fromLocation', ''),
                    'TO_NAME'       : item.get(
                        'to', ''),
                    'TO_IDENTITY'   : item.get(
                        'toIdentity', ''),
                    'TO_TYPE'       : item.get(
                        'toType', ''),
                    'TO_LOCATION'   : item.get(
                        'toLocation', '')
                })

    return rows


def parse_hierarchy(asset_id, asset_name, response):
    """Parse hierarchy section from API response"""
    rows = []
    hierarchy = response.get('hierarchy', [])

    for col in hierarchy:
        rows.append({
            'TABLE_ASSET_ID' : asset_id,
            'TABLE_NAME'     : asset_name,
            'COL_ASSET_ID'   : col.get(
                'core.identity', ''),
            'COL_NAME'       : col.get(
                'summary', {}).get('core.name', '')
        })

    return rows


# ============================================================
# PROGRESS TRACKING
# ============================================================

def update_progress(success=True):
    """Update and log progress"""
    global processed_count, success_count, failed_count

    with progress_lock:
        processed_count += 1
        if success:
            success_count += 1
        else:
            failed_count += 1

        if processed_count % 10 == 0 or \
           processed_count == total_assets:
            pct = (processed_count / total_assets) * 100
            logger.info(
                f"Progress: "
                f"{processed_count}/{total_assets} "
                f"({pct:.1f}%) - "
                f"Success: {success_count} - "
                f"Failed: {failed_count}")


# ============================================================
# PROCESS FUNCTIONS
# ============================================================

def process_asset(row, iics_api_url, lineage_distance):
    """Process single asset"""
    asset_id   = row['ASSET_ID']
    asset_name = row['ASSET_NAME']

    try:
        logger.info(
            f"Processing: {asset_name} - {asset_id}")

        # Call API
        response = get_table_lineage(
            asset_id, iics_api_url, lineage_distance)

        # Parse lineage
        lineage_rows = parse_lineage(
            asset_id, asset_name, response)

        # Parse hierarchy
        column_rows = parse_hierarchy(
            asset_id, asset_name, response)

        # Thread safe append
        with lock:
            table_lineage_data.extend(lineage_rows)
            column_list_data.extend(column_rows)

        update_progress(success=True)

        logger.info(
            f"Completed: {asset_name} - "
            f"Lineage: {len(lineage_rows)} - "
            f"Columns: {len(column_rows)}")

    except Exception as e:
        logger.error(
            f"Failed: {asset_name} - "
            f"{asset_id} - {str(e)}")

        with lock:
            error_log_data.append({
                'ASSET_ID'   : asset_id,
                'ASSET_NAME' : asset_name,
                'ERROR_CODE' : type(e).__name__,
                'ERROR_DESC' : str(e),
                'TIMESTAMP'  : datetime.now()
                              .strftime('%Y-%m-%d %H:%M:%S')
            })

        update_progress(success=False)


def run_parallel(df_assets, iics_api_url,
                 max_workers, lineage_distance):
    """Run parallel processing for all assets"""
    global total_assets
    total_assets = len(df_assets)

    logger.info(
        f"Starting parallel processing - "
        f"Total   : {total_assets} - "
        f"Workers : {max_workers} - "
        f"Distance: {lineage_distance}")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(max_workers)
    ) as executor:
        futures = [
            executor.submit(
                process_asset, row,
                iics_api_url, lineage_distance)
            for _, row in df_assets.iterrows()
        ]

        for future in \
            concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Thread error: {str(e)}")

    logger.info("Parallel processing completed!")


# ============================================================
# VALIDATION
# ============================================================

def validate_input_file(df):
    """Validate input Excel file"""
    required_cols = ['ASSET_ID', 'ASSET_NAME']

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Missing column: {col}")

    if len(df) == 0:
        raise ValueError("Input file is empty!")

    # Remove duplicates
    dupes = df[df.duplicated(['ASSET_ID'])]
    if len(dupes) > 0:
        logger.warning(
            f"Duplicates found: {len(dupes)} - "
            f"Removing...")
        df.drop_duplicates(
            subset=['ASSET_ID'], inplace=True)

    logger.info(
        f"Input validated - "
        f"Total assets: {len(df)}")
    return df


# ============================================================
# OUTPUT
# ============================================================

def save_output(output_path):
    """Save all output files to CSV — always 3 files"""
    os.makedirs(output_path, exist_ok=True)

    # ─── TABLE LINEAGE ────────────────────────────
    file = (
        f"{output_path}/"
        f"table_lineage_{timestamp}.csv")
    df_lineage = pd.DataFrame(
        table_lineage_data,
        columns=[
            'ASSET_ID', 'ASSET_NAME',
            'DIRECTION', 'DISTANCE',
            'FROM_NAME', 'FROM_IDENTITY',
            'FROM_TYPE', 'FROM_LOCATION',
            'TO_NAME', 'TO_IDENTITY',
            'TO_TYPE', 'TO_LOCATION'
        ])
    df_lineage.to_csv(file, index=False)
    logger.info(
        f"Table lineage saved: {file} - "
        f"{len(df_lineage)} rows")

    # ─── COLUMN LIST ──────────────────────────────
    file = (
        f"{output_path}/"
        f"column_list_{timestamp}.csv")
    df_columns = pd.DataFrame(
        column_list_data,
        columns=[
            'TABLE_ASSET_ID', 'TABLE_NAME',
            'COL_ASSET_ID', 'COL_NAME'
        ])
    df_columns.to_csv(file, index=False)
    logger.info(
        f"Column list saved: {file} - "
        f"{len(df_columns)} rows")

    # ─── ERROR LOG ────────────────────────────────
    file = (
        f"{output_path}/"
        f"error_log_{timestamp}.csv")
    df_errors = pd.DataFrame(
        error_log_data,
        columns=[
            'ASSET_ID', 'ASSET_NAME',
            'ERROR_CODE', 'ERROR_DESC',
            'TIMESTAMP'
        ])
    df_errors.to_csv(file, index=False)
    logger.info(
        f"Error log saved: {file} - "
        f"{len(df_errors)} errors")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ─── CONFIG ───────────────────────────────────
    config = configparser.ConfigParser()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.environ.get(
        "cdgc_config",
        os.path.join(base_dir, "config/cdgc_config.ini"))
    config.read(config_path)

    # ─── PATHS ────────────────────────────────────
    output_path   = config["paths"]["output_path"]
    input_path    = config["paths"]["input_path"]
    log_file_path = config["paths"]["log_file_path"]

    # ─── LOGGER ───────────────────────────────────
    os.makedirs(log_file_path, exist_ok=True)
    log_file_name = (
        f"{log_file_path}"
        f"/cdgc_lineage_{timestamp}.log")
    logger = get_logger(
        name="cdgc_logger",
        log_file=log_file_name,
        level=logging.INFO
    )
    logger.info("=" * 60)
    logger.info("CDGC Lineage Export Started!")
    logger.info("=" * 60)

    # ─── CREDENTIALS ──────────────────────────────
    iics_username = config[
        "iics_login_prod_cdgc"]["iics_username"]
    iics_password = config[
        "iics_login_prod_cdgc"]["iics_password"]
    iics_base_url = config[
        "iics_login_prod_cdgc"]["iics_base_url"]
    jwt_url       = config[
        "iics_login_prod_cdgc"]["iics_jwt_url"]
    iics_api_url  = config[
        "iics_login_prod_cdgc"]["iics_api_url"]

    # ─── SCHEDULING ───────────────────────────────
    max_workers      = config[
        "scheduling"]["max_worker_table"]
    lineage_distance = config[
        "scheduling"]["lineage_distance"]

    # ─── AUTH ─────────────────────────────────────
    session_id, server_url, org_id = get_session_id(
        iics_base_url, iics_username, iics_password)

    jwt_token = get_jwt_token(session_id, jwt_url)

    headers = {
        "Authorization" : f"Bearer {jwt_token}",
        "X-INFA-ORG-ID" : org_id,
        "Content-Type"  : "application/json"
    }

    # ─── LOAD & VALIDATE ──────────────────────────
    df_assets = pd.read_excel(input_path)
    df_assets = validate_input_file(df_assets)
    logger.info(f"Total assets: {len(df_assets)}")

    # ─── RUN ──────────────────────────────────────
    run_parallel(
        df_assets, iics_api_url,
        max_workers, lineage_distance)

    # ─── SAVE ─────────────────────────────────────
    save_output(output_path)

    # ─── SUMMARY ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("CDGC Lineage Export Completed!")
    logger.info(f"Total assets : {total_assets}")
    logger.info(f"Success      : {success_count}")
    logger.info(f"Failed       : {failed_count}")
    logger.info(
        f"Lineage rows : {len(table_lineage_data)}")
    logger.info(
        f"Column rows  : {len(column_list_data)}")
    logger.info(
        f"Errors       : {len(error_log_data)}")
    logger.info("=" * 60)
