import requests, json, time

# you must already have these from the same run:
# baseApiUrl, session_id, job_id

export_url = f"{baseApiUrl}/public/core/v3/license/metering/ExportMeteringData"
status_url = f"{export_url}/jobs/{job_id.strip()}"
headers = {"INFA-SESSION-ID": session_id, "Accept": "application/json"}

print("DEBUG baseApiUrl:", baseApiUrl)
print("DEBUG job_id repr:", repr(job_id))
print("DEBUG status_url:", status_url)
print("DEBUG headers:", headers)

# --- single probe, no redirects so we see the real URL that returns 404 ---
r = requests.get(status_url, headers=headers, verify=False, allow_redirects=False)
print("HTTP:", r.status_code)
print("Final URL:", r.url)
print("History:", [h.status_code for h in r.history])
print("Content-Type:", r.headers.get("content-type"))
print("Body:", r.text[:500])

# If it's 200, you're good. If it's 404, keep polling a bit to rule out timing.
for i in range(6):
    if r.status_code == 200:
        break
    time.sleep(5)
    r = requests.get(status_url, headers=headers, verify=False, allow_redirects=False)
    print(f"[retry {i+1}] HTTP:", r.status_code, "| Body:", r.text[:200])
