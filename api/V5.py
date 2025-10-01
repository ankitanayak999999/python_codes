import requests

# ---- v3 LOGIN ----
def login_v3():
    login_url = "https://dm-us.informaticacloud.com/saas/public/core/v3/login"
    r = requests.post(login_url,
                      json={"username": "YOUR_USER", "password": "YOUR_PASS"},
                      verify=False)
    print("login status", r.status_code)
    j = r.json()
    # IMPORTANT: take baseApiUrl from products[0]
    base = j["products"][0]["baseApiUrl"]     # e.g. https://usw3.dm-us.informaticacloud.com/saas
    sid  = j["userInfo"]["sessionId"]
    print("DEBUG baseApiUrl:", base)
    print("DEBUG sessionId :", sid[:12]+"...")
    return sid, base

# ---- CREATE EXPORT (same base, same session) ----
def create_export(session_id, baseApiUrl):
    url = f"{baseApiUrl}/public/core/v3/license/metering/ExportMeteringData"
    hdr = {"INFA-SESSION-ID": session_id, "Accept": "application/json"}
    payload = {
        "startDate": "2025-10-01T00:00:00Z",
        "endDate"  : "2025-10-01T23:59:59Z",
        "jobType"  : "SUMMARY",
        "combinedMeterUsage": "TRUE",
        "allLinkedOrgs": "TRUE"
    }
    r = requests.post(url, headers=hdr, json=payload, verify=False)
    print("export api status", r.status_code)
    print("export body      ", r.text[:300])
    r.raise_for_status()
    job_id = r.json()["jobId"].strip()
    print("Job ID:", repr(job_id))
    return job_id

# ---- STATUS CHECK (single probe, no loop yet) ----
def check_export_status(session_id, baseApiUrl, job_id):
    url = f"{baseApiUrl}/public/core/v3/license/metering/ExportMeteringData/jobs/{job_id}"
    hdr = {"INFA-SESSION-ID": session_id, "Accept": "application/json"}  # no Content-Type on GET
    print("DEBUG status_url :", url)
    print("DEBUG headers    :", hdr)
    r = requests.get(url, headers=hdr, verify=False)
    print("request status", r.status_code)
    print(r.text[:500])

session_id, baseApiUrl = login_v3()
job_id = create_export(session_id, baseApiUrl)
check_export_status(session_id, baseApiUrl, job_id)
