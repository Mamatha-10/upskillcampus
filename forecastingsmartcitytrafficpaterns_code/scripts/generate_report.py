import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import importlib.util
from pathlib import Path as _P

# Import train_models by path to avoid module path issues
root = _P(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_models", str(root / "train_models.py"))
train_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_models)

OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)

print('Running training pipeline (this will re-run training)...')
results_df, trained_models, final_model, submission = train_models.main()

# Save metrics table
metrics_path = OUT_DIR / "metrics.csv"
results_df.to_csv(metrics_path, index=False)

# Feature importances (if available)
fi_path = OUT_DIR / "feature_importances.png"
if hasattr(final_model, 'feature_importances_'):
    importances = pd.Series(final_model.feature_importances_, index=train_models.FEATURE_COLS).sort_values(ascending=True)
    plt.figure(figsize=(6,4))
    importances.plot.barh(color='C0')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig(fi_path)
    plt.close()
else:
    fi_path = None

# Save sample submission preview
preview_path = OUT_DIR / "submission_preview.csv"
submission.head(50).to_csv(preview_path, index=False)

# Create simple HTML report
html_path = OUT_DIR / "report_generated.html"
with open(html_path, 'w', encoding='utf-8') as f:
    f.write('<html><head><meta charset="utf-8"><title>Traffic Forecast Report</title></head><body>')
    f.write('<h1>Traffic Forecast Report</h1>')
    f.write('<h2>Model comparison (validation)</h2>')
    f.write(results_df.to_html(index=False))
    if fi_path and fi_path.exists():
        f.write('<h2>Top Feature Importances</h2>')
        f.write(f'<img src="{fi_path.name}" alt="feature importances">')
    f.write('<h2>Submission preview (first 50 rows)</h2>')
    f.write(submission.head(50).to_html(index=False))
    f.write('</body></html>')

print(f'Report written: {html_path}')
print(f'Metrics saved: {metrics_path}')
print(f'Submission preview saved: {preview_path}')
print('\n=== Generated output files ===')
print(f'Trained model: {root / "models" / "traffic_prediction_model.pkl"}')
print(f'Submission file: data/submission.csv')
print(f'Metrics table: {metrics_path}')
print(f'Preview data: {preview_path}')
if fi_path and fi_path.exists():
    print(f'Feature importance plot: {fi_path}')
print(f'HTML report: {html_path}')
print('Open the HTML report in your browser to view text, images, and visualizations together.')
