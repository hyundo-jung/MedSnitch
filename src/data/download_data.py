import kagglehub
import os
import shutil
from pathlib import Path

def download_dataset():
    """Download the healthcare provider fraud detection dataset from Kaggle"""
    print("Downloading dataset...")
    
    # Download the dataset
    path = kagglehub.dataset_download("rohitrox/healthcare-provider-fraud-detection-analysis")
    print(f"Dataset downloaded to: {path}")
    
    # Create data directory if it doesn't exist
    data_dir = Path("../../data/raw")  # Go up two levels from src/data to reach the root data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define file name mappings
    file_mappings = {
        "Train_Beneficiarydata-1542865627584.csv": "beneficiary.csv",
        "Train_Inpatientdata-1542865627584.csv": "inpatient.csv",
        "Train_Outpatientdata-1542865627584.csv": "outpatient.csv",
        "Train-1542865627584.csv": "labels.csv"
    }
    
    files_to_remove = ["Test_Beneficiarydata-1542969243754.csv",
                       "Test_Inpatientdata-1542969243754.csv",
                       "Test_Outpatientdata-1542969243754.csv",
                       "Test-1542969243754.csv"]
    # Move and rename files to data/raw directory
    print("Moving and renaming files to data/raw directory...")
    for file in Path(path).glob("*"):
        if file.is_file():
            # Skip files that are marked for removal
            if files_to_remove and file.name in files_to_remove:
                print(f"Skipping {file.name} as it is marked for removal")
                continue
                
            # Get the new name from the mapping, or keep original if not in mapping
            new_name = file_mappings.get(file.name, file.name)
            shutil.copy(file, data_dir / new_name)
            print(f"Moved and renamed {file.name} to {new_name}")
    
    print("Download and setup complete!")

if __name__ == "__main__":
    download_dataset()