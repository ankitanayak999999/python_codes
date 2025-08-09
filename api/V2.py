from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd  # NEW

RETRY_STATUS = {408, 429, 500, 502, 503, 504}

def fetch_json_by_id(v_list=None, url=None, workers=30, max_attempts=3, base_sleep=1.5):
    if not v_list:
        return [], [], [], [], pd.DataFrame()

    total_cnt = len(v_list)
    print(f'Fetch json by id started for {total_cnt} - mtts')

    success_cnt = 0
    failed_cnt = 0
    success_json_list = []
    failed_json_list = []
    error_detail_list = []
    summary_list = []

    def do_login():
        sid, server_url = cm.api_login_v2()
        hdrs = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'icSessionId': sid
        }
        return hdrs, server_url

    headers, serverUrl = do_login()
    base_url = f"{serverUrl}{url}"

    def call_with_attempts(item):
        id_val = item.get("id", "NA")
        attempt = 0
        local_headers = headers.copy()
        codes_seen = []

        while attempt < max_attempts:
            attempt += 1
            results = cm.api_call_get_generic(base_url, local_headers, id_val)
            sc = results.get("status_code")

            try:
                sc_int = int(sc)
            except Exception:
                sc_int = None

            codes_seen.append(str(sc) if sc is not None else "NA")

            if sc_int == 200:
                return ('ok', results, codes_seen)

            if sc_int == 401:
                local_headers, _ = do_login()
                time.sleep(0.3)
                attempt -= 1  # free retry
                continue

            if sc_int in RETRY_STATUS:
                time.sleep(min(base_sleep ** attempt, 15))
                continue

            return ('err', results, codes_seen)

        return ('err', results, codes_seen)

    start_time = time.time()
    printed_pct = -1

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(call_with_attempts, row): row for row in v_list}
        for i, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            status, results, codes_seen = fut.result()

            if status == 'ok':
                json_data = results.get("json_data", {})
                success_json_list.append(json_data)
                success_cnt += 1
                final_status = "success"
            else:
                failed_json_list.append(row)
                error_detail_list.append(results)
                failed_cnt += 1
                final_status = "failed"

            summary_list.append({
                "id": row.get("id", "NA"),
                "final_status": final_status,
                "retry_codes": ",".join(codes_seen),
                "attempt_count": len(codes_seen)
            })

            pct = int(i * 100 / total_cnt)
            if pct // 5 > printed_pct:
                printed_pct = pct // 5
                elapsed_sec = time.time() - start_time
                elapsed_min = elapsed_sec / 60.0
                rate = i / elapsed_sec if elapsed_sec > 0 else 0
                eta_min = ((total_cnt - i) / rate / 60.0) if rate > 0 else 0
                print(f"{pct}% ({i}/{total_cnt}) in {elapsed_min:.2f} min | ETA {eta_min:.2f} min")

    elapsed_time = (time.time() - start_time) / 60.0
    first_attempt_success = sum(1 for s in summary_list if s["final_status"] == "success" and s["attempt_count"] == 1)

    print(f"Completed in {elapsed_time:.2f} min | Success: {success_cnt} | Failed: {failed_cnt} | "
          f"First-attempt success: {first_attempt_success}")

    all_json_list = success_json_list + failed_json_list
    summary_df = pd.DataFrame(summary_list)  # convert here

    return all_json_list, success_json_list, failed_json_list, error_detail_list, summary_df
