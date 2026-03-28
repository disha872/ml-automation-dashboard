import pandas as pd

def clean_data(df):
    for col in df.select_dtypes(include=['float64', 'int64']):
        df[col] = df[col].fillna(df[col].mean())

    for col in df.select_dtypes(include=['object']):
        if df[col].mode().empty:
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop_duplicates()

    return df