import nbformat
import os
import re
import base64
import sys
from bs4 import BeautifulSoup

NB_PATH = sys.argv[1] if len(sys.argv) > 1 else "traffic_project/notebooks/Traffic_Forecasting_standard_executed.ipynb"
OUT_DIR = os.path.join(os.path.dirname(__file__), "notebook_extract")
os.makedirs(OUT_DIR, exist_ok=True)

nb = nbformat.read(NB_PATH, as_version=4)
headings = []
text_outputs = []
images = []
tables = []

for c_i, cell in enumerate(nb.cells):
    if cell.get("cell_type") == "markdown":
        for line in cell.get("source", "").splitlines():
            m = re.match(r"^(#{1,6})\s+(.*)", line)
            if m:
                level = len(m.group(1))
                text = m.group(2).strip()
                headings.append((level, text))
    elif cell.get("cell_type") == "code":
        for out_i, out in enumerate(cell.get("outputs", [])):
            otype = out.get("output_type")
            # plain text / stream
            if otype == "stream":
                text = out.get("text", "")
                if text:
                    text_outputs.append(text)
            # execute_result or display_data
            if otype in ("execute_result", "display_data"):
                data = out.get("data", {})
                # text/plain
                if "text/plain" in data:
                    text_outputs.append(data["text/plain"]) 
                # html -> try extract tables
                if "text/html" in data:
                    html = data["text/html"]
                    soup = BeautifulSoup(html, "html.parser")
                    for t in soup.find_all("table"):
                        # extract first few rows
                        rows = []
                        for tr in t.find_all("tr")[:10]:
                            cols = [td.get_text(strip=True) for td in tr.find_all(["th","td"])]
                            rows.append(cols)
                        tables.append(rows)
                # images (png)
                if "image/png" in data:
                    b64 = data["image/png"]
                    imgdata = base64.b64decode(b64)
                    fname = os.path.join(OUT_DIR, f"cell{c_i}_out{out_i}.png")
                    with open(fname, "wb") as f:
                        f.write(imgdata)
                    images.append(fname)

# Search for metric-like lines
joined_text = "\n".join(text_outputs)
metric_lines = []
for line in joined_text.splitlines():
    if re.search(r"MAE|RMSE|R2|R\^2|mean absolute|mean squared|Random Forest|RandomForest|Decision Tree|LinearRegression|Saved final|submission", line, re.I):
        metric_lines.append(line.strip())

# Title (first H1)
title = None
for lvl, txt in headings:
    if lvl == 1:
        title = txt
        break

print(f"Report title: {title or os.path.basename(NB_PATH)}\n")

print("--- Headings (level,text) ---\n")
for lvl, txt in headings[:40]:
    print(f"H{lvl}: {txt}")

print("\n--- Key metric lines / snippets ---\n")
for l in (metric_lines[:40] if metric_lines else text_outputs[:40]):
    print(l)

print("\n--- Table previews (first 2 tables) ---\n")
for t in tables[:2]:
    for row in t[:10]:
        print(" | ".join(row))
    print("---")

print("\n--- Images saved (count) ---")
print(len(images))
for p in images[:10]:
    print(p)

print("\n--- Sample console/text outputs (first 20 lines) ---\n")
sample_lines = []
for t in text_outputs:
    sample_lines.extend(t.splitlines())
for ln in sample_lines[:200]:
    print(ln)

print("\nExtraction complete.")
