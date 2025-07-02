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
    return df
import pandas as pd

def convert_date_columns(df):
    for col in df.columns:
        try:
            # Attempt to convert the column to datetime
            converted = pd.to_datetime(df[col], errors='coerce')

            # Only update if at least 70% of non-null values are valid dates
            non_nulls = df[col].notna().sum()
            if non_nulls > 0 and converted.notna().sum() / non_nulls >= 0.7:
                # Format all valid dates to 'YYYY-MM-DD'
                df[col] = converted.dt.strftime('%Y-%m-%d')
                print(f"✅ Converted column to date format: {col}")
        except Exception:
            pass  # Safe fallback
    return df
    import pandas as pd
import warnings

def convert_date_columns(df):
    # Define allowed date-like keywords in column names
    date_keywords = ['DATE', 'DT', 'DTE', 'DTTM', 'TIME', 'TS', 'TIMESTAMP']

    for col in df.columns:
        col_upper = col.upper()

        # Only convert if column name contains any date keyword
        if not any(keyword in col_upper for keyword in date_keywords):
            continue  # Skip this column

        try:
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=UserWarning)
                converted = pd.to_datetime(df[col], errors='coerce')

            non_nulls = df[col].notna().sum()
            if non_nulls > 0 and converted.notna().sum() / non_nulls >= 0.7:
                df[col] = converted.dt.strftime('%Y-%m-%d')
                print(f"✅ Converted column to date format: {col}")

        except Exception:
            pass
    return df

