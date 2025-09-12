import os
import pandas as pd

DATA_FOLDER = "."

def main():
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".csv"):
            path = os.path.join(DATA_FOLDER, filename)
            try:
                # Try to read with parse_dates if 'Date' column exists, else read normally
                with open(path, 'r') as f:
                    header = f.readline()
                columns = [col.strip() for col in header.split(',')]
                if "Date" in columns:
                    df = pd.read_csv(path, parse_dates=["Date"])
                    min_date = df["Date"].min()
                    max_date = df["Date"].max()
                    print(f"{filename}: {len(df)} rows, {min_date} to {max_date}, columns: {list(df.columns)}")
                else:
                    df = pd.read_csv(path)
                    print(f"{filename}: {len(df)} rows, columns: {list(df.columns)} (no 'Date' column)")
            except Exception as e:
                print(f"{filename}: Error reading file - {e}")

if __name__ == "__main__":
    main()
