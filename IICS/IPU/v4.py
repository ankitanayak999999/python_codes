# iics_idmc_ipu_extract_v1.py
# Production-friendly: adds logging while preserving your inputs/flow.

import iics_login_cred as iics_cred
import requests, time, json, os, logging
import urllib3
import urllib3.exceptions
from datetime import datetime, timedelta
import pandas as pd
import io
import zipfile
import csv

# =========================
# Logging helper
# =========================
def get_logger(name: str, log_file: str | None = None, level: str = "INFO") -> logging.Logger:
    """
    Create a console logger (and optional file logger) with timestamped messages.
    Level can be overridden by environment variable LOG_LEVEL.
    """
    lvl = os.getenv("LOG_LEVEL", level).upper()
    logger = logging.getLogger(name)
    if logger.handlers:
        # Reuse existing logger (avoid duplicate handlers on reruns)
        logger.setLevel(lvl)
        return logger

    logger.setLevel(lvl)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(lvl)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(lvl)
        logger.addHandler(fh)

    return logger

# Create module-level logger (console only by default).
# If you want a file too, set LOG_FILE env var, or edit here.
LOGGER = get_logger(__name__, os.getenv("LOG_FILE"), level="INFO")

# =========================
# Networking tweaks
# =========================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def login_v3(user_name, password, base_url):
    login_url_v3 = f"{base_url}/saas/public/core/v3/login"
    login_headers = { 'Content-Type': 'application/json', 'Accept': 'application/json' }
    login_payload = { "username": user_name, "password": password }
    resp = requests.post(login_url_v3, headers=login_headers, json=login_payload, verify=False)
    LOGGER.info(f"API Status Code | Login API | {resp.status_code}")
    resp.raise_for_status()
    sessionId = resp.json()['userInfo']['sessionId']
    serverUrl = resp.json()['products'][0]['baseApiUrl']
    return sessionId, serverUrl

def check_export_status(input_param_dict):
    session_id = input_param_dict['session_id']
    status_url = input_param_dict['status_url']
    status_header = { "INFA-SESSION-ID": session_id, "Accept": "application/json" }
    resp = requests.get(status_url, headers=status_header, verify=False)
    LOGGER.info(f"API Status Code | JOB Status API | {resp.status_code}")
    resp.raise_for_status()
    export_status = resp.json()["status"]
    return export_status

# (kept) tiny cleaner—safe and fast; combined with pandas tolerant read below
def clean_csv_content(text_io: io.TextIOBase) -> io.StringIO:
    """
    Re-read CSV with Python's csv.reader so commas inside text values are handled
    correctly. Returns a cleaned StringIO for pandas to read. Original ZIP bytes untouched.
    """
    text_io.seek(0)
    reader = csv.reader(text_io, quotechar='"')
    cleaned_lines = []
    for row in reader:
        cleaned_lines.append(",".join(row))
    return io.StringIO("\n".join(cleaned_lines))

def download_export_zip(input_param_dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_id = input_param_dict["session_id"]
    download_url = input_param_dict["download_url"]
    path = input_param_dict["path"]

    # Ensure target folder exists (no change to input semantics)
    os.makedirs(path, exist_ok=True)

    headers = { "INFA-SESSION-ID": session_id }
    resp = requests.get(download_url, headers=headers, verify=False)
    LOGGER.info(f"API Status Code | Download Status API | {resp.status_code}")
    resp.raise_for_status()

    output_file = f"{path}/{timestamp}_ipu_usage.zip"
    with open(output_file, "wb") as f:
        f.write(resp.content)
    LOGGER.info(f"Download Completed | File saved at: {output_file}")

    dfs = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                with z.open(name) as f:
                    csv_file = io.TextIOWrapper(f, encoding="utf-8-sig")
                    LOGGER.debug(f"Reading CSV from ZIP member: {name}")
                    try:
                        # Clean then read; tolerate malformed lines so job never crashes
                        cleaned_csv = clean_csv_content(csv_file)
                        df = pd.read_csv(
                            cleaned_csv,
                            sep=",",
                            engine="python",
                            quotechar='"',
                            on_bad_lines="skip",     # tolerate any rogue rows
                            error_bad_lines=False     # for older pandas
                        )
                    except pd.errors.EmptyDataError:
                        LOGGER.warning(f"Empty CSV — Skipping file: {name}")
                        df = pd.DataFrame()
                    except Exception as e:
                        LOGGER.error(f"Failed reading CSV {name}: {e}")
                        df = pd.DataFrame()

                    if not df.empty:
                        dfs.append(df)
                    else:
                        LOGGER.debug(f"No rows read from: {name}")

    if not dfs:
        LOGGER.warning("No valid CSVs found in ZIP folder")
        return

    final_df = pd.concat(dfs, ignore_index=True)
    excel_file_name = f"{path}/{timestamp}_ipu_usages.xlsx"
    try:
        final_df.to_excel(excel_file_name, index=False)
        LOGGER.info(f"Excel saved: {excel_file_name} (rows={len(final_df)})")
    except Exception as e:
        LOGGER.error(f"Failed writing Excel {excel_file_name}: {e}")
        raise

def get_job_id(input_param_dict, payload):
    session_id = input_param_dict["session_id"]
    serverUrl = input_param_dict["serverUrl"]
    job_id = input_param_dict.get("job_id")

    # Start export job
    # url = f"{serverUrl}/public/core/v3/license/metering/ExportMeteringDataAllLinkedOrgsAcrossRegion"
    url = f"{serverUrl}/public/core/v3/license/metering/ExportMeteringData"

    header = { "INFA-SESSION-ID": session_id, "Accept": "application/json" }
    export_payload = payload
    resp = requests.post(url, headers=header, json=export_payload, verify=False)
    LOGGER.info(f"API Status Code | Export API | {resp.status_code}")

    # Log response body if 4xx/5xx to aid diagnosis
    if resp.status_code >= 400:
        try:
            LOGGER.error(f"Export API error body: {resp.json()}")
        except Exception:
            LOGGER.error(f"Export API error text: {resp.text}")

    try:
        data = resp.json()
        job_id = data.get("jobId")
    except Exception:
        data = resp.text
        job_id = None

    if job_id:
        job_id = job_id.strip()
        LOGGER.info(f"Export Started | Job ID: {repr(job_id)}")
        return job_id
    else:
        raise RuntimeError(f"Export Request Failed | Status Code: {resp.status_code} | Response: {data}")

def create_export(input_param_dict, payload):
    serverUrl = input_param_dict["serverUrl"]
    job_id = get_job_id(input_param_dict, payload)
    input_param_dict["job_id"] = job_id
    input_param_dict["status_url"] = f"{serverUrl}/public/core/v3/license/metering/ExportMeteringData/{job_id}"
    input_param_dict["download_url"] = f"{serverUrl}/public/core/v3/license/metering/ExportMeteringData/{job_id}/download"

    # check if export creation is completed
    timeout_limit_min = input_param_dict["timeout_limit_min"]
    interval = input_param_dict["interval"]
    deadline = datetime.now() + timedelta(minutes=timeout_limit_min)

    for check_num in range(1, 9999):
        if datetime.now() > deadline:
            LOGGER.error(f"Timed Out after {timeout_limit_min} minutes")
            break

        try:
            export_status = check_export_status(input_param_dict)
        except Exception as e:
            LOGGER.error(f"Status check failed: {e}")
            time.sleep(interval)
            continue

        LOGGER.info(f"{datetime.now():%H:%M:%S} | Status check Attempt: {check_num} | Status: {export_status}")

        if export_status == "SUCCESS":
            LOGGER.info("Export created successfully — Downloading the ZIP file")
            download_export_zip(input_param_dict)
            break
        elif export_status in ("FAILED", "CANCELLED"):
            LOGGER.error(f"Export failed with status: {export_status}")
            break

        time.sleep(interval)

def main_run():
    # ---- inputs remain exactly as you had them ----
    username, password, base_url = iics_cred.iics_prd_cred()
    session_id, serverUrl = login_v3(username, password, base_url)

    startDate = "2025-10-01T00:00:00Z"
    endDate   = "2025-10-02T23:59:59Z"
    jobType = "SUMMARY"          # "SUMMARY" / "ASSET" / "PROJECT_FOLDER"
    combinedMeterUsage = "FALSE" # "TRUE" combine meters; "FALSE" separate rows per meter
    allLinkedOrgs = "TRUE"       # "TRUE" all sub orgs; "FALSE" only current org
    path = "C:/Users/raksahu/Downloads/python/json_output"

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

if __name__ == "__main__":
    main_run()
