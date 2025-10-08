import pandas as pd
from decimal import Decimal

# 1) Rename columns to match Oracle
rename_map = {
    "OrgId": "ORGID",
    "MeterId": "METER_ID",
    "MeterName": "METER_NAME",
    "Date": "DATE",
    "BillingPeriodStartDate": "BILLING_PERIOD_START_DATE",
    "BillingPeriodEndDate": "BILLING_PERIOD_END_DATE",
    "MeterUsage": "METER_USAGES",     # <- plural in DB
    "IPU": "IPU",
    "Scalar": "SCALAR",
    "MetricCategory": "METRIC_CATEGORY",
    "OrgName": "ORG_NAME",
    "OrgType": "ORG_TYPE",
    "IPURate": "IPU_RATE",
}
df = df.rename(columns=rename_map)

# 2) Timestamps (coerce bad values to NaT)
for c in ["DATE", "BILLING_PERIOD_START_DATE", "BILLING_PERIOD_END_DATE"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")
        # If they came in timezone-aware, strip tz to naive UTC for Oracle:
        if getattr(df[c].dt, "tz", None) is not None:
            df[c] = df[c].dt.tz_convert("UTC").dt.tz_localize(None)

# 3) Numerics: use Decimal for exact NUMBER(38,12) (or use to_numeric if float is fine)
for c in ["METER_USAGES", "IPU", "SCALAR", "IPU_RATE"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # Optional: convert to Decimal for exactness
        df[c] = df[c].apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)

# 4) Strings: ensure dtype is str/object and trim if needed
str_cols = ["ORGID","METER_ID","METER_NAME","METRIC_CATEGORY","ORG_NAME","ORG_TYPE"]
for c in [x for x in str_cols if x in df.columns]:
    df[c] = df[c].astype("string").str.strip()
