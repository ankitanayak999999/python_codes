from dateutil.parser import parse
import pandas as pd
import openpyxl

def is_probably_date(val):
    try:
        parse(str(val))
        return True
    except:
        return False

def df_clean(df):
    for col in df.columns:
        if df[col].dtypes == 'object':
            # Unescape Excel HTML strings (if any)
            df[col] = df[col].astype(str).apply(openpyxl.utils.escape.unescape)

            try:
                # Sample 10 non-null values to test if column looks like dates
                sample = df[col].dropna().astype(str).head(10)
                date_like_count = sum(is_probably_date(v) for v in sample)

                # If at least 70% of sample values are date-like, convert the column
                if date_like_count >= len(sample) * 0.7:
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
            except:
                pass

    # Clean up line breaks and tabs across entire DataFrame
    df = df.replace(r'\r|\n|\r\n|\t+', ' ', regex=True)
    return dfthe
