import pandas as pd
import json
import os

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mapping_file = os.path.join(BASE_DIR, 'icd_to_ccs_mapping.json')
csv_file = os.path.join(BASE_DIR, '$dxref 2015.csv')

# Load the CSV file
df = pd.read_csv(csv_file, skiprows=1)

# Remove leading and trailing whitespace from column names
df.columns = df.columns.str.strip().str.replace("'", "")

df = df.dropna(subset=['ICD-9-CM CODE', 'CCS CATEGORY'])

df['ICD-9-CM CODE'] = df['ICD-9-CM CODE'].str.strip("'").str.strip()
df['CCS CATEGORY'] = df['CCS CATEGORY'].str.strip("'").str.strip()

print(df.head(10))


# Create the mapping dictionary
icd_to_ccs_mapping = {}
for _, row in df.iterrows():
    icd_code = str(row['ICD-9-CM CODE']).strip()
    ccs_category = str(row['CCS CATEGORY']).strip()
    
    # Check if the CCS category is a valid digit and icd_code is not empty
    if icd_code and ccs_category.isdigit():
        icd_to_ccs_mapping[icd_code] = int(ccs_category)


# Save the mapping as a JSON file
with open(mapping_file, 'w') as f:
    json.dump(icd_to_ccs_mapping, f)

print("Mapping saved successfully.")