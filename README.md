# MedSnitch

A supervised learning system for detecting fraudulent medical billing practices such as phantom billing, unbundling, and duplicate claims. Designed for insurers and regulators, the project includes a frontend dashboard, RESTful API, and multiple classification models. The models are trained on real healthcare billing data from the National Institute of Allergy and Infectious Diseases.

## Data Preprocessing

1. **Download Data**
   cd src/data
   python download_data.py  # Downloads raw data to data/raw/

2. **Get CCS Mapping**
   - Download "Single Level CCS" from [HCUP CCS Tools](https://hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp)
   - Move `$dxref 2015.csv` to `data/mappings/`

3. **Run Preprocessor**
   python preprocess.py

The preprocessor:
- Combines inpatient/outpatient claims
- Processes diagnosis/procedure codes (keeps first code + counts)
- Converts dates to features
- Adds beneficiary details
- Calculates costs
- Adds fraud labels
- Scales features
- Removes IDs

Output: `data/processed/processed_claims.csv` and `data/processed/scaler.joblib`

To use the scaler in your model:
```python
import joblib
scaler = joblib.load('data/processed/scaler.joblib')
```