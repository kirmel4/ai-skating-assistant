import pandas as pd

def parse_excel(path, sheet_name=0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)

    time_range = df["t_start"].astype(str).str.extract(r"\[(.*?)\s*-\s*(.*?)\]")
    df["t_start_raw"] = time_range[0]
    df["t_end_raw"] = time_range[1]

    def parse_time(x):
        if pd.isna(x):
            return pd.NaT
        x = str(x).strip()
        parts = x.split(":")
        if len(parts) == 2:
            x = f"00:{parts[0]}:{parts[1]}"
        return pd.to_datetime(x, format="%H:%M:%S").time()

    df["t_start_val"] = df["t_start_raw"].apply(parse_time)
    df["t_end_val"] = df["t_end_raw"].apply(parse_time)

    df.drop(columns=["t_start", "t_end", "t_start_raw", "t_end_raw"], inplace=True)

    df = df.sort_values(by=["video_id", "t_start_val"]).reset_index(drop=True)

    return df
