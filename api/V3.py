import requests, time, json

# ---------- Config ----------
POD       = "usw3"  # your IDMC pod (usw3, use1, eu1, etc.)
USERNAME  = "you@example.com"
PASSWORD  = "*****"
START     = "2025-07-01T00:00:00Z"
END       = "2025-09-30T23:59:59Z"
OUT_FILE  = "ipu_usage.csv"
# ----------------------------

# 1. Login
login_url = f"https://dm-{POD}.informaticacloud.com/saas/public/core/v3/login"
resp = requests.post(login_url, json={"username": USERNAME, "password": PASSWORD})
resp.raise_for_status()
data = resp.json()
session_id = data["sessionId"]
base_api_url = data["baseApiUrl"]

headers = {"INFA-SESSION-ID": session_id, "Accept": "application/json"}

# 2. Request export
export_url = f"{base_api_url}/public/core/v3/license/metering/ExportMeteringData"
payload = {
    "startDate": START,
    "endDate": END,
    "jobType": "SUMMARY",
    "combinedMeterUsage": "TRUE",
    "allLinkedOrgs": "TRUE"
}
resp = requests.post(export_url, headers=headers, json=payload)
resp.raise_for_status()
job_id = resp.json()["jobId"]
print("Job ID:", job_id)

# 3. Poll status
status_url = f"{export_url}/jobs/{job_id}"
while True:
    status = requests.get(status_url, headers=headers).json()["status"]
    print("Status:", status)
    if status == "SUCCEEDED":
        break
    elif status in ("FAILED", "CANCELLED"):
        raise RuntimeError("Export failed with status " + status)
    time.sleep(5)

# 4. Download CSV
download_url = f"{status_url}/download"
csv_resp = requests.get(download_url, headers={"INFA-SESSION-ID": session_id, "Accept": "text/csv"})
csv_resp.raise_for_status()
with open(OUT_FILE, "wb") as f:
    f.write(csv_resp.content)

print("Saved to", OUT_FILE)
