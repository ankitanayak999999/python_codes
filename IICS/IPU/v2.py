# iics_idmc_ipu_extract_v1.py
# (Transcribed exactly from your screenshots)

import iics_login_cred as iics_cred
import requests, time, json
import urllib3
import urllib3.exceptions
from datetime import datetime, timedelta
import pandas as pd
import io
import zipfile
import csv  # <-- added

# disable warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def login_v3(user_name, password, base_url):
    login_url_v3 = f"{base_url}/saas/public/core/v3/login"
    login_headers = { 'Content-Type': 'application/json', 'Accept': 'application/json' }
    login_payload = { "username": user_name, "password": password }
    resp = requests.post(login_url_v3, headers=login_headers, json=login_payload, verify=False)
    print(f"API Status Code | Login API | {resp.status_code}")
    sessionId = resp.json()['userInfo']['sessionId']
    serverUrl = resp.json()['products'][0]['baseApiUrl']
    return sessionId, serverUrl

def check_export_status(input_param_dict):
    session_id = input_param_dict['session_id']
    status_url = input_param_dict['status_url']
    status_header = { "INFA-SESSION-ID": session_id, "Accept": "application/json" }
    resp = requests.get(status_url, headers=status_header, verify=False)
    print(f" API Status Code | JOB Status API | {resp.status_code}")
    export_status = resp.json()["status"]
    return export_status

# ---- tiny cleaner (added) ----
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
# ------------------------------

def download_export_zip(input_param_dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_id = input_param_dict["session_id"]
    download_url = input_param_dict["download_url"]
    path = input_param_dict["path"]
    headers = { "INFA-SESSION-ID": session_id }
    resp = requests.get(download_url, headers=headers, verify=False)
    print(f" API Status Code | Download Status API | {resp.status_code}")
    output_file = f"{path}\\{timestamp}_ipu_usage.zip"
    with open(output_file, "wb") as f:
        f.write(resp.content)
    print(f"Download Completed , File saved at:{output_file}")

    dfs = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                with z.open(name) as f:
                    csv_file = io.TextIOWrapper(f, encoding="utf-8-sig")
                    print(csv_file)
                    try:
                        # patched: clean the CSV text before pandas reads it
                        cleaned_csv = clean_csv_content(csv_file)
                        df = pd.read_csv(
                            cleaned_csv,
                            sep=",",
                            engine="python",
                            quotechar='"',
                            on_bad_lines="warn"
                        )
                    except pd.errors.EmptyDataError:
                        print(f"Empty csv - Skipping file:{name}")
                        df = pd.DataFrame()
                    if not df.empty:
                        dfs.append(df)

    if not dfs:
        print("No Valid CCS found in Zip folder")
        return
    final_df = pd.concat(dfs, ignore_index=True)
    excel_file_name = f"{path}\\{timestamp}_ipu_usages.xlsx"
    final_df.to_excel(excel_file_name, index=False)

def get_job_id(input_param_dict, payload):
    session_id = input_param_dict["session_id"]
    serverUrl = input_param_dict["serverUrl"]
    job_id = input_param_dict.get("job_id")

    # Start export job
    # Option A (all linked orgs across region)
    # url = f"{serverUrl}/public/core/v3/license/metering/ExportMeteringDataAllLinkedOrgsAcrossRegion"
    # Option B (generic)
    url = f"{serverUrl}/public/core/v3/license/metering/ExportMeteringData"

    header = { "INFA-SESSION-ID": session_id, "Accept": "application/json" }
    export_payload = payload
    resp = requests.post(url, headers=header, json=export_payload, verify=False)
    print(f" API Status Code | Export API | {resp.status_code}")
    try:
        data = resp.json()
        job_id = data.get("jobId")
    except:
        data = resp.text
        job_id = None

    if job_id:
        job_id = job_id.strip()
        print(f"Export Started and Job ID:{repr(job_id)}")
        return job_id
    else:
        raise RuntimeError(f"Export Request Failed|Status Code :{resp.status_code}|Response:{data}")

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
            print(f"Timed Out after {timeout_limit_min} minutes")
            break
        export_status = check_export_status(input_param_dict)
        print(f"{datetime.now():%H:%M:%S} | Status check Attempt:{check_num} | Status :{export_status}")
        if export_status == "SUCCESS":
            print("Export created successfully , Downloading the Zip file")
            download_export_zip(input_param_dict)
            break
        elif export_status in ("FAILED", "CANCELLED"):
            print(f"Export failed with status :{export_status}")
            break
        time.sleep(interval)

def main_run():
    username,password,base_url = iics_cred.iics_prd_cred()
    session_id, serverUrl = login_v3(username,password,base_url)

    startDate = "2025-10-01T00:00:00Z"      # start time from when data needed
    endDate   = "2025-10-02T23:59:59Z"      # End time from when data needed
    jobType = "SUMMARY"                     # "SUMMARY" Aggregated Usages , "ASSET","PROJECT_FOLDER"
    combinedMeterUsage = "FALSE"            # "TRUE" Combine data across meters  "FALSE" seperate rows for meter
    allLinkedOrgs = "TRUE"                  # "TRUE" for all sub org  "FALSE" for only current org
    path = "C:/Users/rakashu/Downloads/python/json_output"

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

main_run() was
