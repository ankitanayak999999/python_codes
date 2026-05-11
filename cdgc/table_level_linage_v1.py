# ============================================================
# CDGC Lineage Export
# Author:
# Date: 2026-05-05
# Description: Extract table lineage and column list
#              from CDGC using REST API
# Version: 4.1 - Monitoring Enhanced
# ============================================================

import pandas as pd
import requests
import configparser
import logging
import os
import threading
import concurrent.futures
from datetime import datetime
import urllib3
import sys
import json
from queue import Queue

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_dir)

from common import iics_metadata_cm as cm
from common.logger_helper import get_logger

# ─── GLOBALS ────────────────────────────────────────────────
lock      = threading.Lock()
jwt_lock  = threading.Lock()
headers   = {}
logger    = None
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")


# ============================================================
# AUTH FUNCTIONS
# ============================================================

def get_session_id(base_url, user_name, password):

    login_url_v3 = f"{base_url}/saas/public/core/v3/login"

    _headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    payload = {"username": user_name, "password": password}

    resp = requests.post(login_url_v3, headers=_headers, json=payload, verify=False)

    logger.info(f"API Status Code | Login API | {resp.status_code}")

    try:

        resp.raise_for_status()

        data       = resp.json()
        session_id = data['userInfo']['sessionId']
        org_id     = data['userInfo']['orgId']
        server_url = data['products'][0]['baseApiUrl']

        logger.debug(f"Login OK | server_url={server_url} | session_id_len={len(session_id)}")

        return session_id, server_url, org_id

    except Exception:

        safe_body = None

        try:
            safe_body = json.dumps(resp.json())[:1000]
        except Exception:
            safe_body = resp.text[:1000]

        logger.exception(f"Login failed | status={resp.status_code} | body={safe_body}")

        raise


def get_jwt_token(session_id, jwt_url):
    """Get JWT token using session ID"""

    try:

        _headers = {
            "cookie"         : f"USER_SESSION={session_id}",
            "IDS-SESSION-ID" : session_id
        }

        resp = requests.get(jwt_url, headers=_headers, verify=False, timeout=60)

        resp.raise_for_status()

        jwt_token = resp.json()["jwt_token"]

        logger.info("JWT Token obtained successfully!")

        return jwt_token

    except Exception as e:

        logger.error(f"JWT Token failed: {str(e)}")

        raise


def refresh_cdgc_api_headers(job_run_config):
    """Build/Refresh CDGC API headers"""

    global headers

    iics_base_url = job_run_config['iics_base_url']
    iics_username = job_run_config['iics_username']
    iics_password = job_run_config['iics_password']
    jwt_url       = job_run_config['jwt_url']

    current_token = headers.get("Authorization")

    with jwt_lock:

        try:

            if headers.get("Authorization") != current_token:
                logger.info("Headers already refreshed by another thread - skipping!")
                return

            new_session_id, _, new_org_id = get_session_id(
                iics_base_url,
                iics_username,
                iics_password
            )

            new_jwt = get_jwt_token(new_session_id, jwt_url)

            headers = {
                "Authorization": f"Bearer {new_jwt}",
                "X-INFA-ORG-ID": new_org_id,
                "Content-Type" : "application/json"
            }

            logger.info("CDGC API headers refreshed successfully!")

        except Exception as e:

            logger.error(f"Header refresh failed: {str(e)}")

            raise


# ============================================================
# API FUNCTIONS
# ============================================================

def get_table_lineage_json(asset_id, job_run_config):
    """
    Call CDGC lineage + hierarchy API
    Auto refreshes JWT on 401
    Returns response JSON
    """

    cdgc_base_url    = job_run_config['cdgc_base_url']
    lineage_distance = job_run_config['lineage_distance']
    api_timeout_secs = int(job_run_config['api_timeout_secs'])

    url = (
        f"{cdgc_base_url}"
        f"/data360/search/v1/assets/{asset_id}"
        f"?scheme=internal"
        f"&segments=lineage"
        f",lineage-direction:all"
        f",lineage-distance:{lineage_distance}"
        f",hierarchy:all"
    )

    resp = requests.get(url, headers=headers, verify=False, timeout=api_timeout_secs)

    if resp.status_code == 401:

        logger.warning(f"401 for asset: {asset_id} - Re-authenticating...")

        refresh_cdgc_api_headers(job_run_config)

        resp = requests.get(url, headers=headers, verify=False, timeout=api_timeout_secs)

    if resp.status_code != 200:

        try:
            error_msg = resp.json().get('message', resp.text)
        except:
            error_msg = resp.text

        raise Exception(f"HTTP_{resp.status_code} | {error_msg}")

    return resp.json()


# ============================================================
# PARSE FUNCTIONS
# ============================================================

def parse_lineage(asset_id, asset_name, response):

    rows    = []
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
                    'FROM_NAME'     : item.get('from', ''),
                    'FROM_IDENTITY' : item.get('fromIdentity', ''),
                    'FROM_TYPE'     : item.get('fromType', ''),
                    'FROM_LOCATION' : item.get('fromLocation', ''),
                    'TO_NAME'       : item.get('to', ''),
                    'TO_IDENTITY'   : item.get('toIdentity', ''),
                    'TO_TYPE'       : item.get('toType', ''),
                    'TO_LOCATION'   : item.get('toLocation', '')
                })

    return rows


def parse_hierarchy(asset_id, asset_name, response):

    rows      = []
    hierarchy = response.get('hierarchy', [])

    for col in hierarchy:

        rows.append({
            'TABLE_ASSET_ID' : asset_id,
            'TABLE_NAME'     : asset_name,
            'COL_ASSET_ID'   : col.get('core.identity', ''),
            'COL_NAME'       : col.get('summary', {}).get('core.name', '')
        })

    return rows


# ============================================================
# BACKGROUND THREAD MONITOR
# ============================================================

def monitor_parallel_execution(job_run_config, monitor_tracker):

    log_interval_secs = int(job_run_config['log_interval_secs'])
    max_workers       = int(job_run_config['max_workers'])

    while monitor_tracker["is_running"]:

        try:

            total_assets = monitor_tracker["total_assets"]
            processed    = monitor_tracker["processed"]
            success      = monitor_tracker["success"]
            failed       = monitor_tracker["failed"]

            remaining = total_assets - processed
            running   = min(max_workers, remaining)
            pending   = max(remaining - running, 0)

            pct = (processed / total_assets) * 100 if total_assets > 0 else 0

            elapsed_total = (
                datetime.now() - monitor_tracker["start_time"]
            ).total_seconds()

            rate = processed / elapsed_total if elapsed_total > 0 else 0

            eta_secs = (
                (total_assets - processed) / rate
                if rate > 0 else 0
            )

            elapsed_mins = int(elapsed_total // 60)
            elapsed_secs = int(elapsed_total % 60)

            eta_mins   = int(eta_secs // 60)
            eta_secs_r = int(eta_secs % 60)

            logger.info(
                f"[THREAD-MONITOR] "
                f"Processed={processed}/{total_assets} ({pct:.1f}%) | "
                f"Running={running} | "
                f"Pending={pending} | "
                f"Remaining={remaining} | "
                f"Success={success} | "
                f"Failed={failed} | "
                f"Rate={rate:.2f}/sec | "
                f"Elapsed={elapsed_mins}m {elapsed_secs}s | "
                f"ETA={eta_mins}m {eta_secs_r}s"
            )

            threading.Event().wait(log_interval_secs)

        except Exception as e:

            logger.error(f"Monitor thread failed: {str(e)}")


# ============================================================
# PROCESS FUNCTIONS
# ============================================================

def process_dataset_lineage(row, job_run_config):

    asset_id   = row['ASSET_ID']
    asset_name = row['ASSET_NAME']

    try:

        logger.info(f"Processing: {asset_name} - {asset_id}")

        response = get_table_lineage_json(asset_id, job_run_config)

        lineage_rows = parse_lineage(asset_id, asset_name, response)
        column_rows  = parse_hierarchy(asset_id, asset_name, response)

        logger.info(
            f"Completed: {asset_name} - "
            f"Lineage: {len(lineage_rows)} - "
            f"Columns: {len(column_rows)}"
        )

        return lineage_rows, column_rows, []

    except Exception as e:

        logger.error(f"Failed: {asset_name} - {asset_id} - {str(e)}")

        error_row = [{
            'ASSET_ID'   : asset_id,
            'ASSET_NAME' : asset_name,
            'ERROR_CODE' : type(e).__name__,
            'ERROR_DESC' : str(e),
            'TIMESTAMP'  : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }]

        return [], [], error_row


# ============================================================
# PARALLEL EXECUTOR
# ============================================================

def parallel_run_executor(input_data, call_function_name, max_workers, job_run_config):

    logger.info(
        f"Starting parallel execution - "
        f"Total: {len(input_data)} - "
        f"Workers: {max_workers} - "
        f"Function: {call_function_name.__name__}"
    )

    table_lineage_data = []
    column_list_data   = []
    error_log_data     = []

    total_assets    = len(input_data)
    processed_count = 0
    success_count   = 0
    failed_count    = 0

    start_time = datetime.now()

    # ─── LOCAL MONITOR TRACKER ───────────────────
    monitor_tracker = {
        "total_assets" : total_assets,
        "processed"    : 0,
        "success"      : 0,
        "failed"       : 0,
        "running"      : 0,
        "start_time"   : datetime.now(),
        "is_running"   : True
    }

    # ─── START MONITOR THREAD ────────────────────
    monitor_thread = threading.Thread(
        target=monitor_parallel_execution,
        args=(job_run_config, monitor_tracker),
        daemon=True
    )

    monitor_thread.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

        if hasattr(input_data, 'iterrows'):

            futures = [
                executor.submit(call_function_name, row, job_run_config)
                for _, row in input_data.iterrows()
            ]

        else:

            futures = [
                executor.submit(call_function_name, item, job_run_config)
                for item in input_data
            ]

        monitor_tracker["running"] = min(int(max_workers), total_assets)

        for future in concurrent.futures.as_completed(futures):

            try:

                lineage_rows, column_rows, error_rows = future.result()

                table_lineage_data.extend(lineage_rows)
                column_list_data.extend(column_rows)
                error_log_data.extend(error_rows)

                processed_count += 1
                success_count   += 1 if not error_rows else 0
                failed_count    += 1 if error_rows else 0

                # ─── MONITOR UPDATE ────────────────
                monitor_tracker["processed"] = processed_count
                monitor_tracker["success"]   = success_count
                monitor_tracker["failed"]    = failed_count

            except Exception as e:

                logger.error(f"Thread error: {str(e)}")

                processed_count += 1
                failed_count    += 1

                # ─── MONITOR UPDATE ────────────────
                monitor_tracker["processed"] = processed_count
                monitor_tracker["success"]   = success_count
                monitor_tracker["failed"]    = failed_count

    # ─── STOP MONITOR THREAD ─────────────────────
    monitor_tracker["is_running"] = False

    total_time = (datetime.now() - start_time).total_seconds()

    total_mins = int(total_time // 60)
    total_secs = int(total_time % 60)

    logger.info(
        f"Parallel execution completed - "
        f"Function: {call_function_name.__name__} - "
        f"Total time: {total_mins}m {total_secs}s"
    )

    return table_lineage_data, column_list_data, error_log_data


def run_parallel(df_assets, job_run_config):

    max_workers        = int(job_run_config['max_workers'])
    call_function_name = process_dataset_lineage

    table_lineage_data, column_list_data, error_log_data = (
        parallel_run_executor(
            df_assets,
            call_function_name,
            max_workers,
            job_run_config
        )
    )

    logger.info("Parallel processing completed!")

    return table_lineage_data, column_list_data, error_log_data
# ============================================================
# VALIDATION
# ============================================================

def validate_input_file(df):

    required_cols = ['ASSET_ID', 'ASSET_NAME', 'LINEAGE_REFRESH_FLAG']

    for col in required_cols:

        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    if len(df) == 0:
        raise ValueError("Input file is empty!")

    dupes = df[df.duplicated(['ASSET_ID'])]

    if len(dupes) > 0:

        logger.warning(f"Duplicates found: {len(dupes)} - Removing...")

        df.drop_duplicates(subset=['ASSET_ID'], inplace=True)

    total_before = len(df)

    df = df[
        df['LINEAGE_REFRESH_FLAG'].str.upper() == 'Y'
    ]

    total_after = len(df)

    logger.info(
        f"Input validated - "
        f"Total: {total_before} - "
        f"To process (Y): {total_after} - "
        f"Skipped (N): {total_before - total_after}"
    )

    return df


# ============================================================
# OUTPUT
# ============================================================

def save_output(output_path, table_lineage_data, column_list_data, error_log_data):

    os.makedirs(output_path, exist_ok=True)

    # ─── TABLE LINEAGE ────────────────────────────
    file = f"{output_path}/CDGC_TABLE_LVL_LINEAGE_{timestamp}.csv"

    column_list = [
        'ASSET_ID',
        'ASSET_NAME',
        'DIRECTION',
        'DISTANCE',
        'FROM_NAME',
        'FROM_IDENTITY',
        'FROM_TYPE',
        'FROM_LOCATION',
        'TO_NAME',
        'TO_IDENTITY',
        'TO_TYPE',
        'TO_LOCATION'
    ]

    df_lineage = pd.DataFrame(
        table_lineage_data,
        columns=column_list
    )

    df_lineage.to_csv(file, index=False)

    logger.info(
        f"Table lineage saved: {file} - "
        f"{len(df_lineage)} rows"
    )

    # ─── COLUMN LIST ──────────────────────────────
    file = f"{output_path}/CDGC_TABLE_COLUMNS_{timestamp}.csv"

    df_columns = pd.DataFrame(
        column_list_data,
        columns=[
            'TABLE_ASSET_ID',
            'TABLE_NAME',
            'COL_ASSET_ID',
            'COL_NAME'
        ]
    )

    df_columns.to_csv(file, index=False)

    logger.info(
        f"Column list saved: {file} - "
        f"{len(df_columns)} rows"
    )

    # ─── ERROR LOG ────────────────────────────────
    file = f"{output_path}/CDGC_ERROR_LOG_TABLE_LVL_{timestamp}.csv"

    column_list = [
        'ASSET_ID',
        'ASSET_NAME',
        'ERROR_CODE',
        'ERROR_DESC',
        'TIMESTAMP'
    ]

    df_errors = pd.DataFrame(
        error_log_data,
        columns=column_list
    )

    df_errors.to_csv(file, index=False)

    logger.info(
        f"Error log saved: {file} - "
        f"{len(df_errors)} errors"
    )

    logger.info("save file function run completed")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ─── CONFIG ───────────────────────────────────
    config      = configparser.ConfigParser()
    base_dir    = os.path.dirname(os.path.abspath(__file__))

    config_path = os.environ.get(
        "cdgc_config",
        os.path.join(base_dir, "config/cdgc_config.ini")
    )

    config.read(config_path)

    # ─── PATHS ────────────────────────────────────
    output_path   = config["paths"]["output_path"]
    input_path    = config["paths"]["input_path"]
    log_file_path = config["paths"]["log_file_path"]

    # ─── LOGGER ───────────────────────────────────
    os.makedirs(log_file_path, exist_ok=True)

    log_file_name = (
        f"{log_file_path}/cdgc_lineage_{timestamp}.log"
    )

    logger = get_logger(
        name="cdgc_logger",
        log_file=log_file_name,
        level=logging.INFO
    )

    logger.info("=" * 60)
    logger.info("CDGC Lineage Export Started!")
    logger.info("=" * 60)

    # ─── CREDENTIALS ──────────────────────────────
    iics_username = config["iics_login_prod_cdgc"]["iics_username"]
    iics_password = config["iics_login_prod_cdgc"]["iics_password"]

    # ─── URL ──────────────────────────────────────
    iics_base_url = config["urls"]["iics_base_url"]
    jwt_url       = config["urls"]["cdgc_jwt_url"]
    cdgc_base_url = config["urls"]["cdgc_base_url"]

    # ─── JOB CONTROL ──────────────────────────────
    max_workers       = config["cdgc_job_control"]["max_worker_table"]
    lineage_distance  = config["cdgc_job_control"]["lineage_distance"]
    api_timeout_secs  = config["cdgc_job_control"]["api_timeout_secs"]
    log_interval_secs = config["cdgc_job_control"]["log_interval_secs"]

    # ─── AUTH ─────────────────────────────────────
    job_run_config = {
        "iics_base_url"     : iics_base_url,
        "iics_username"     : iics_username,
        "iics_password"     : iics_password,
        "jwt_url"           : jwt_url,
        "cdgc_base_url"     : cdgc_base_url,
        "max_workers"       : max_workers,
        "api_timeout_secs"  : api_timeout_secs,
        "lineage_distance"  : lineage_distance,
        "log_interval_secs" : log_interval_secs
    }

    refresh_cdgc_api_headers(job_run_config)

    # ─── LOAD & VALIDATE ──────────────────────────
    asset_file = f'{input_path}/cdgc_object_asset_list.xlsx'

    df_assets = pd.read_excel(asset_file)

    df_assets = validate_input_file(df_assets)

    logger.info(f"Total assets to process: {len(df_assets)}")

    # ─── RUN ──────────────────────────────────────
    table_lineage_data, column_list_data, error_log_data = (
        run_parallel(df_assets, job_run_config)
    )

    # ─── SAVE ─────────────────────────────────────
    save_output(
        output_path,
        table_lineage_data,
        column_list_data,
        error_log_data
    )

    # ─── SUMMARY ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("CDGC Lineage Export Completed!")
    logger.info(f"Total assets  : {len(df_assets)}")
    logger.info(f"Lineage rows  : {len(table_lineage_data)}")
    logger.info(f"Column rows   : {len(column_list_data)}")
    logger.info(f"Errors        : {len(error_log_data)}")
    logger.info("=" * 60)
