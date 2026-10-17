import openpyxl
import statistics as st
import csv

SRC = "/home/claude/work/Merged_Human_Ratings.xlsx"
OUT_CSV = "/home/claude/work/Calibrated_Human_Ratings.csv"

wb = openpyxl.load_workbook(SRC, data_only=True)
ws = wb["Merged Ratings"]
rows = list(ws.iter_rows(values_only=True))
header = rows[0]
rater_cols = header[1:]
data_rows = rows[1:]

# Step 1: প্রতিটা রেটারের নিজের mean এবং std বের করা
rater_vals = {h: [] for h in rater_cols}
for r in data_rows:
    for h, v in zip(rater_cols, r[1:]):
        if v is not None:
            rater_vals[h].append(v)

rater_mean = {h: sum(v)/len(v) for h, v in rater_vals.items() if v}
rater_std = {h: (st.pstdev(v) if len(v) > 1 else 1.0) for h, v in rater_vals.items() if v}
# std শূন্য হলে (যে রেটার সবসময় একই নম্বর দিয়েছে) ভাগ করতে সমস্যা এড়াতে
rater_std = {h: (s if s > 1e-6 else 1.0) for h, s in rater_std.items()}

# Step 2: প্রতিটা রেটিংকে z-score এ রূপান্তর করে প্রতি ছবির গড় বের করা
results = []
for r in data_rows:
    img = r[0]
    raw_vals = [v for v in r[1:] if v is not None]
    z_vals = []
    for h, v in zip(rater_cols, r[1:]):
        if v is not None:
            z = (v - rater_mean[h]) / rater_std[h]
            z_vals.append(z)
    if not raw_vals:
        continue
    raw_mean = sum(raw_vals) / len(raw_vals)
    raw_range = max(raw_vals) - min(raw_vals)
    z_mean = sum(z_vals) / len(z_vals)
    results.append({
        "Image": img,
        "N-Raters": len(raw_vals),
        "Raw-Mean-1to10": round(raw_mean, 3),
        "Raw-Range": raw_range,
        "Calibrated-Z-Mean": round(z_mean, 4),
    })

# Step 3: z-mean গুলোকে আবার পড়ার-উপযোগী 1-10 স্কেলে ফিরিয়ে আনা (min-max rescale)
all_z = [r["Calibrated-Z-Mean"] for r in results]
z_min, z_max = min(all_z), max(all_z)
for r in results:
    r["Calibrated-Rating-1to10"] = round(1 + 9 * (r["Calibrated-Z-Mean"] - z_min) / (z_max - z_min), 3)

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Image", "N-Raters", "Raw-Mean-1to10", "Raw-Range", "Calibrated-Z-Mean", "Calibrated-Rating-1to10"])
    writer.writeheader()
    writer.writerows(results)

print(f"Done. {len(results)} images written to {OUT_CSV}")
print()
print("যেসব ছবিতে raw rating-এ সবচেয়ে বেশি মতভেদ ছিল, সেগুলো calibration-এর পর কেমন দেখাচ্ছে:")
high_disagreement = sorted(results, key=lambda r: -r["Raw-Range"])[:6]
print(f"{'Image':10s} {'N':>3s} {'Raw-Mean':>9s} {'Raw-Range':>10s} {'Calibrated(1-10)':>18s}")
for r in high_disagreement:
    print(f"{r['Image']:10s} {r['N-Raters']:3d} {r['Raw-Mean-1to10']:9.2f} {r['Raw-Range']:10d} {r['Calibrated-Rating-1to10']:18.2f}")
