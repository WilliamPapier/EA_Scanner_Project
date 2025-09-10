# infer_and_write_params.py
# For now: acts as a passthrough/adjustment step. Could load ML model later.
import csv, os, datetime, math

INFILE = "model_params.csv"
OUTFILE = "model_params.csv"  # we will rewrite same file after light adjustments

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def main():
    if not os.path.exists(INFILE):
        print("No scanner output to infer.")
        return
    rows = []
    with open(INFILE, "r", newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                prob = int(r["probability"])
            except:
                continue
            # Example: if probability very high, suggest slightly higher risk (but do not exceed 5%)
            suggested = float(r.get("suggested_risk_percent", 1.0))
            if prob >= 90:
                suggested = clamp(suggested * 1.5, 0.2, 5.0)
            elif prob >= 80:
                suggested = clamp(suggested * 1.1, 0.2, 5.0)
            else:
                suggested = clamp(suggested, 0.2, 5.0)
            r["suggested_risk_percent"] = round(suggested,2)
            rows.append(r)
    # rewrite file
    with open(OUTFILE, "w", newline='') as f:
        fieldnames = ["symbol","direction","probability","entry","sl","tp","suggested_risk_percent","entry_types","timestamp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"{datetime.datetime.now().isoformat()} inference updated {OUTFILE} with {len(rows)} rows")

if __name__ == "__main__":
    main()
