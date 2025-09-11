# news_updater.py
# Lightweight placeholder. Writes an empty CSV (no news blocked).
import csv, datetime, os

OUT = "news_block.csv"
with open(OUT, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["start_utc","end_utc","impact","title"])
print(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} news_updater: wrote empty {OUT}")