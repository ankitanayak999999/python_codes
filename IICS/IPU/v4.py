
# iics_idmc_ipu_extract_v1.py
# (Patched to use generic logger_helper with hardcoded params in main_run)

import io
import json
import zipfile
import csv  # <-- kept from your version
import requests
import time
import urllib3
import urllib3.exceptions
import pandas as pd
from datetime import datetime, timedelta
import logging
import os

import iics_login_cred as iics_cred
from logger_helper import get_logger  # 👈 generic helper import

# Disable SSL warnings (your original behavior)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# We'll initialize this in main_run() with hardcoded values
logger: logging.Logger | None = None


def login_v3(user_name, password, base_url):
    login_url_v3 = f"{base_url}/saas/public/core/v3/login"
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    payload = {"username": user_name, "password": password}

    resp = requests.post(login_url_v3, headers=headers, json=payload, verify=False)
    logger.info(f"API Status Code | Login API | {resp.status_code}")

    try:
        resp.raise_for_status()
        data = resp.json()
        session_id = data['userInfo']['sessionId']
        server_url = data['products'][0]['baseApiUrl']
        logger.debug(f"Login OK | server_url={server_url} | session_id_len={len(session_id)}")
        return session_id, server_url
    except Exception:
        # Keep body short to avoid PII leakage
        safe_body = None
        try:
            safe_body = json.dumps(resp.json())[:1000]
        except Exception:
            safe_body = resp.text[:1000]
        logger.exception(f"Login failed | status={resp.status_code} | body={safe_body}")
        raise


def check_export_status(input_param_dict):
    session_id = input_param_dict['session_id']
    status_url = input_param_dict['status_url']
    headers = {"INFA-SESSION-ID": session_id, "Accept": "application/json"}

    resp = requests.get(status_url, headers=headers, verify=False)
    logger.info(f"API Status Code | JOB Status API | {resp.status_code}")
    resp.raise_for_status()
    status = resp.json().get("status")
    logger.debug(f"Export status payload: {resp.json()}")
    return status


# ---- patched cleaner (no row skipping) ----
def clean_csv_content(text_io: io.TextIOBase) -> io.StringIO:
    """
    Fix rows that have extra commas inside text by collapsing the overflow back
    into a free-text column based on header column count. Keeps all rows.
    """
    text_io.seek(0)
    raw = text_io.read()

    # normalize newlines
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")
    if not lines or not lines[0].strip():
        return io.StringIO(raw)

    header = lines[0]
    hdr_cols = header.split(",")
    ncols = len(hdr_cols)

    # choose a likely free-text column to absorb overflow
    try:
        text_idx = hdr_cols.index("MeterName")
    except ValueError:
        candidates = ["Meter Type", "Meter", "Name", "Asset Name"]
        text_idx = next((hdr_cols.index(c) for c in candidates if c in hdr_cols), 1)
        if text_idx < 0 or text_idx >= ncols:
            text_idx = 1  # fallback

    fixed = [",".join(hdr_cols)]
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split(",")

        if len(parts) == ncols:
            fixed.append(",".join(parts))
            continue

        if len(parts) < ncols:
            # pad missing fields
            parts += [""] * (ncols - len(parts))
            fixed.append(",".join(parts))
            continue

        # too many fields → collapse middle into the text column
        right_needed = ncols - (text_idx + 1)
        left = parts[:text_idx]
        if right_needed > 0:
            right = parts[-right_needed:]
            middle = parts[text_idx: len(parts) - right_needed]
        else:
            right = []
            middle = parts[text_idx:]
        merged_text = ",".join(middle)  # preserve commas inside the text value
        fixed.append(",".join(left + [merged_text] + right))

    return io.StringIO("\n".join(fixed))
# -------------------------------------------


def download_export_zip(input_param_dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_id = input_param_dict["session_id"]
    download_url = input_param_dict["download_url"]
    path = input_param_dict["path"]
    headers = {"INFA-SESSION-ID": session_id}

    resp = requests.get(download_url, headers=headers, verify=False)
    logger.info(f"API Status Code | Download Status API | {resp.status_code}")
    resp.raise_for_status()

    os.makedirs(path, exist_ok=True)
    output_file = f"{path}/{timestamp}_ipu_usage.zip"
    with open(output_file, "wb") as f:
        f.write(resp.content)
    logger.info(f"Download completed | File saved at: {output_file}")

    dfs = []
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            for name in z.namelist():
                if name.lower().endswith(".csv"):
                    logger.debug(f"Processing CSV in ZIP: {name}")
                    with z.open(name) as f:
                        csv_file = io.TextIOWrapper(f, encoding="utf-8-sig")
                        try:
                            cleaned_csv = clean_csv_content(csv_file)
                            df = pd.read_csv(
                                cleaned_csv,
                                sep=",",
                                engine="python",
                                quotechar='"'
                            )  # no on_bad_lines → don't drop rows
                        except pd.errors.EmptyDataError:
                            logger.warning(f"Empty CSV - Skipping file: {name}")
                            df = pd.DataFrame()
                        except Exception:
                            logger.exception(f"Failed to read CSV: {name}")
                            df = pd.DataFrame()
                        if not df.empty:
                            dfs.append(df)
    except zipfile.BadZipFile:
        logger.exception("Downloaded content is not a valid ZIP")
        raise

    if not dfs:
        logger.warning("No valid CSV found in ZIP")
        return

    final_df = pd.concat(dfs, ignore_index=True)
    excel_file_name = f"{path}/{timestamp}_ipu_usages.xlsx"
    try:
        final_df.to_excel(excel_file_name, index=False)
        logger.info(f"Excel written: {excel_file_name} | rows={len(final_df)} cols={len(final_df.columns)}")
    except Exception:
        logger.exception(f"Failed to write Excel: {excel_file_name}")
        raise


def get_job_id(input_param_dict, payload):
    session_id = input_param_dict["session_id"]
    server_url = input_param_dict["serverUrl"]

    # Start export job
    # Option A (all linked orgs across region)
    # url = f"{server_url}/public/core/v3/license/metering/ExportMeteringDataAllLinkedOrgsAcrossRegion"
    # Option B (generic)
    url = f"{server_url}/public/core/v3/license/metering/ExportMeteringData"

    headers = {"INFA-SESSION-ID": session_id, "Accept": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, verify=False)
    logger.info(f"API Status Code | Export API | {resp.status_code}")

    try:
        resp.raise_for_status()
        data = resp.json()
        job_id = (data.get("jobId") or "").strip()
        if not job_id:
            msg = f"Export request OK but jobId missing | payload={data}"
            logger.error(msg)
            raise RuntimeError(msg)
        logger.info(f"Export started | Job ID: {repr(job_id)}")
        return job_id
    except Exception:
        body = None
        try:
            body = json.dumps(resp.json())[:1000]
        except Exception:
            body = resp.text[:1000]
        logger.exception(f"Export request failed | status={resp.status_code} | body={body}")
        raise


def create_export(input_param_dict, payload):
    server_url = input_param_dict["serverUrl"]
    job_id = get_job_id(input_param_dict, payload)
    input_param_dict["job_id"] = job_id
    input_param_dict["status_url"] = f"{server_url}/public/core/v3/license/metering/ExportMeteringData/{job_id}"
    input_param_dict["download_url"] = f"{server_url}/public/core/v3/license/metering/ExportMeteringData/{job_id}/download"

    # check if export creation is completed
    timeout_limit_min = input_param_dict["timeout_limit_min"]
    interval = input_param_dict["interval"]
    deadline = datetime.now() + timedelta(minutes=timeout_limit_min)

    for check_num in range(1, 9999):
        if datetime.now() > deadline:
            logger.error(f"Timed out after {timeout_limit_min} minutes")
            break
        try:
            export_status = check_export_status(input_param_dict)
        except Exception:
            logger.exception("Status check failed")
            break

        logger.info(f"{datetime.now():%H:%M:%S} | Attempt {check_num} | Status: {export_status}")

        if export_status == "SUCCESS":
            logger.info("Export created successfully | downloading ZIP")
            try:
                download_export_zip(input_param_dict)
            except Exception:
                logger.exception("Download/processing failed")
                raise
            break
        elif export_status in ("FAILED", "CANCELLED"):
            logger.error(f"Export finished with status: {export_status}")
            break
        time.sleep(interval)


def main_run():
    global logger

    # 👇 Hardcode your logging here (helper stays generic)
    logger = get_logger(
        name="iics_ipu_extract",
        log_file=r"C:\Users\raksahu\Downloads\python\logs\iics_ipu_extract.log",
        level=logging.INFO  # change to logging.DEBUG for deep troubleshooting
    )

    logger.info("Starting IPU usage export flow")
    try:
        username, password, base_url = iics_cred.iics_prd_cred()
        session_id, serverUrl = login_v3(username, password, base_url)

        # ---- Your parameters (kept as-is) ----
        startDate = "2025-10-01T00:00:00Z"      # start time from when data needed
        endDate   = "2025-10-02T23:59:59Z"      # End time from when data needed
        jobType = "SUMMARY"                     # "SUMMARY","ASSET","PROJECT_FOLDER"
        combinedMeterUsage = "FALSE"            # "TRUE" combine meters, "FALSE" separate
        allLinkedOrgs = "TRUE"                  # "TRUE" include sub orgs
        path = r"C:\Users\raksahu\Downloads\python\json_output"
        # --------------------------------------

        os.makedirs(path, exist_ok=True)

        export_payload = {
            "startDate": startDate,
            "endDate": endDate,
            "jobType": jobType,
            "combinedMeterUsage": combinedMeterUsage,
            "allLinkedOrgs": allLinkedOrgs
        }

        input_param_dict = {
            "path": path,
            "session_id": session_id,
            "serverUrl": serverUrl,
            "timeout_limit_min": 15,
            "interval": 120
        }

        create_export(input_param_dict, export_payload)
        logger.info("Flow completed ✅")
    except Exception:
        logger.exception("Fatal error in main_run")
        raise


if __name__ == "__main__":
    main_run()
