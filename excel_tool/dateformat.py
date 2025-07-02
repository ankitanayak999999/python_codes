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
    
    import pandas as pd
import warnings

def convert_date_columns(df):
    # Define column name keywords that likely indicate date/datetime columns
    date_keywords = ['DATE', 'DT', 'DTE', 'DTTM', 'TIME', 'TS', 'TIMESTAMP']

    for col in df.columns:
        col_upper = col.upper()

        # Only proceed if column name contains a date-related keyword
        if not any(keyword in col_upper for keyword in date_keywords):
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=UserWarning)
                converted = pd.to_datetime(df[col], errors='coerce')

            non_nulls = df[col].notna().sum()
            valid_dates = converted.notna().sum()

            # Convert only if enough values look like dates
            if non_nulls > 0 and valid_dates / non_nulls >= 0.7:
                # Force format even for large dates like 9999-12-31
                df[col] = converted.dt.strftime('%Y-%m-%d')
                print(f"✅ Converted column to date format: {col}")

        except Exception as e:
            print(f"⚠️ Skipped {col} due to error: {e}")
            continue

    return df
    

import pandas as pd
import warnings

def df_date_clean_v2(df):
    date_keywords = ['DATE', 'DTE', 'DT', 'DTTM', 'TIME', 'TS', 'TIMESTAMP']
    
    for col in df.columns:
        col_upper = col.upper()

        # Step 1: Filter by column name
        if not any(keyword in col_upper for keyword in date_keywords):
            continue

        try:
            # First parsing attempt (automatic)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                converted = pd.to_datetime(df[col], errors='coerce')

            non_nulls = df[col].notna().sum()
            valid_dates = converted.notna().sum()

            # Step 2: If too few valid dates, try common fallback format like MM/DD/YYYY
            if non_nulls > 0 and valid_dates / non_nulls < 0.7:
                fallback_converted = pd.to_datetime(df[col], format='%m/%d/%Y', errors='coerce')
                fallback_valid = fallback_converted.notna().sum()

                if fallback_valid / non_nulls >= 0.7:
                    converted = fallback_converted
                    valid_dates = fallback_valid

            # Step 3: Accept if result is valid enough
            if valid_dates / non_nulls >= 0.7:
                df[col] = converted.dt.strftime('%Y-%m-%d')
                print(f"✅ Converted column to date format: {col}")
            else:
                print(f"⚠️ Skipped column {col}: only {valid_dates}/{non_nulls} valid date values")

        except Exception as e:
            print(f"❌ Error processing {col}: {e}")
            continue

    return df



