import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Retry these HTTP status codes
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}

def fetch_by_id_mt_with_retry(
    v_list,
    url_path,
    workers=30,
    max_retries=4,
    base_sleep=1.5,
    save_prefix=None
):
    """
    Multithreaded fetch using cm.api_call_get_generic with retries.

    Args:
        v_list:      list[dict]  -> each item must contain 'id'
        url_path:    str         -> base path WITHOUT trailing '/{id}', e.g. '/saas/api/v2/mttask'
        workers:     int         -> thread pool size
        max_retries: int         -> max attempts per id
        base_sleep:  float       -> exponential backoff base
        save_prefix: str|None    -> if set, saves *_SUCCESS.json and *_ERRORS.json via cm.json_file_save

    Returns:
        (success_json_list, failed_json_list, error_detail_list)
          success_json_list: list of JSON payloads (only 'json_data' from cm result)
          failed_json_list:  list of the original input dicts that failed
          error_detail_list: list of cm results for failures (status_code/message/etc.)
    """
    if not v_list:
        return [], [], []

    # --- login once; build base url + headers ---
    session_id, serverUrl = cm.api_login_v2()
    base_url = f"{serverUrl}{url_path}"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'INFA-SESSION-ID': session_id,
        'icSessionId': session_id,  # harmless if not required
    }

    def _relogin():
        nonlocal headers
        sid, _ = cm.api_login_v2()
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'INFA-SESSION-ID': sid,
            'icSessionId': sid,
        }

    ids = [(row.get('id'), row) for row in v_list if row.get('id')]

    def _call_with_retry(mid):
        """Call cm.api_call_get_generic(base_url, headers, id=mid) with retries."""
        attempt = 0
        while attempt <= max_retries:
            attempt += 1
            res = cm.api_call_get_generic(base_url, headers, id=mid)

            sc = res.get('status_code')
            try:
                sc_int = int(sc)
            except Exception:
                sc_int = None

            if sc_int == 200:
                return ('ok', mid, res)

            if sc_int == 401 and attempt <= max_retries:
                _relogin()
                time.sleep(0.3)
                continue

            if (sc_int in _RETRY_STATUS) and (attempt <= max_retries):
                time.sleep(min((base_sleep ** attempt), 15))
                continue

            return ('err', mid, res)

    success_json_list, failed_json_list, error_detail_list = [], [], []

    start = time.time()
    printed = -1
    total = len(ids)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_call_with_retry, mid): (mid, row) for mid, row in ids}
        for i, fut in enumerate(as_completed(futs), 1):
            mid, row = futs[fut]
            status, _, res = fut.result()

            if status == 'ok':
                success_json_list.append(res.get('json_data', {}))
            else:
                failed_json_list.append(row)
                error_detail_list.append(res)

            pct = int(i * 100 / total)
            if pct // 5 > printed:
                printed = pct // 5
                elapsed = round((time.time() - start) / 60, 2)
                print(f"{pct}% ({i}/{total}) in {elapsed} min")

    elapsed = round((time.time() - start) / 60, 2)
    print(f"Done. Success: {len(success_json_list)} | Failed: {len(failed_json_list)} | Time: {elapsed} min")

    if save_prefix:
        if success_json_list:
            cm.json_file_save(success_json_list, f"{save_prefix}_SUCCESS")
        if error_detail_list:
            cm.json_file_save(error_detail_list, f"{save_prefix}_ERRORS")

    return success_json_list, failed_json_list, error_detail_list
