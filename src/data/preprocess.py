import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareDataPreprocessor:
    def __init__(self, data_dir: str = "../../../../backend/data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.mappings_dir = self.data_dir / "mappings"
        
        # Create necessary directories
        self.processed_dir.mkdir(exist_ok=True)
        self.mappings_dir.mkdir(exist_ok=True)
        
        # Initialize scaler
        self.scaler = StandardScaler()

    def process_diagnosis_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process diagnosis codes into CCS categories"""
        df = df.copy()
        
        # Count number of non-empty and non-NA diagnoses per claim using vectorized operations
        diagnosis_columns = [f'ClmDiagnosisCode_{i}' for i in range(1, 11)]
        df['num_diagnoses'] = (~df[diagnosis_columns].isna() & (df[diagnosis_columns] != 'NA')).sum(axis=1)
        
        # Keep the first diagnosis code (and strip whitespace)
        df['first_diagnosis'] = df['ClmDiagnosisCode_1'].astype(str).str.strip()
        
        # Load the mapping file
        mapping_df = pd.read_csv(self.mappings_dir / "$dxref 2015.csv", skiprows=1)
        
        # Clean up the data: strip quotes and whitespace
        mapping_df["'ICD-9-CM CODE'"] = mapping_df["'ICD-9-CM CODE'"].str.strip("'").str.strip()
        mapping_df["'CCS CATEGORY'"] = mapping_df["'CCS CATEGORY'"].str.strip("'").str.strip()
        
        # Create a mapping dictionary from ICD-9 code to CCS category
        dx_to_category = dict(zip(
            mapping_df["'ICD-9-CM CODE'"],
            mapping_df["'CCS CATEGORY'"].astype(int)
        ))
        
        # Map the first diagnosis code to its category
        df['DiagnosisCategory'] = df['first_diagnosis'].map(dx_to_category)
        df['DiagnosisCategory'] = df['DiagnosisCategory'].fillna(-1).astype(int) + 1
        
        # Drop original diagnosis code columns
        df = df.drop(columns=diagnosis_columns + ['first_diagnosis'])
        
        return df


    def process_procedure_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process procedure codes, counting non-empty and non-NA codes per claim"""
        df = df.copy()
        
        # Count number of non-empty and non-NA procedures per claim using vectorized operations
        procedure_columns = [f'ClmProcedureCode_{i}' for i in range(1, 7)]
        df['num_procedures'] = (~df[procedure_columns].isna() & (df[procedure_columns] != 'NA')).sum(axis=1)
        
        # Keep the first procedure code
        df['first_procedure'] = df['ClmProcedureCode_1']
        df["first_procedure"] = df["first_procedure"].fillna(0)
        
        # Drop original procedure code columns
        df = df.drop(columns=procedure_columns)
        
        return df

    def process_claim_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process claim dates into day of year, weekend indicator, and claim duration"""
        df = df.copy()
        
        # Convert to datetime
        df['ClaimStartDt'] = pd.to_datetime(df['ClaimStartDt'])
        df['ClaimEndDt'] = pd.to_datetime(df['ClaimEndDt'])
        df['ClaimDayOfYear'] = df['ClaimStartDt'].dt.dayofyear
        
        # Add day of year (use sin and cos to encode cyclic nature)
        df["ClaimDay_sin"] = np.sin(2 * np.pi * df["ClaimDayOfYear"] / 365)
        df["ClaimDay_cos"] = np.cos(2 * np.pi * df["ClaimDayOfYear"] / 365)
        
        # Add weekend indicator (1 for weekend, 0 for weekday)
        df['isWeekend'] = df['ClaimStartDt'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Add claim duration in days
        df['ClaimDuration'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days
        
        # Drop original date columns
        df = df.drop(columns=['ClaimStartDt', 'ClaimEndDt', 'ClaimDayOfYear'])
        
        return df

    def enrich_with_beneficiary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich claims with beneficiary details"""
        # Load beneficiary data
        beneficiary_df = pd.read_csv(self.raw_dir / "beneficiary.csv")
        
        # Convert DOB to datetime
        beneficiary_df['DOB'] = pd.to_datetime(beneficiary_df['DOB'])
        
        # Merge with claims data
        df = df.merge(
            beneficiary_df[['BeneID', 'Gender', 'Race', 'DOB']],
            on='BeneID',
            how='left'
        )
        
        # Calculate age at time of claim
        df['Age'] = (
            pd.to_datetime(df['ClaimStartDt']).dt.year - 
            df['DOB'].dt.year
        )
        
        # Drop DOB as we don't need it anymore
        df = df.drop(columns=['DOB'])
        
        return df

    def calculate_total_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate total cost from reimbursement and deductible amounts"""
        df = df.copy()
        
        # Fill any missing values with 0
        df['InscClaimAmtReimbursed'] = df['InscClaimAmtReimbursed'].fillna(0)
        df['DeductibleAmtPaid'] = df['DeductibleAmtPaid'].fillna(0)
        
        # Calculate total cost
        df['cost'] = df['InscClaimAmtReimbursed'] + df['DeductibleAmtPaid']
        
        # Drop original columns
        df = df.drop(columns=['InscClaimAmtReimbursed', 'DeductibleAmtPaid'])
        
        return df

    def process_fraud_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fraud labels based on provider fraud status"""
        df = df.copy()
        
        # Load fraud labels
        labels_df = pd.read_csv(self.raw_dir / "labels.csv")
        
        # Create a set of fraudulent providers for faster lookup
        fraudulent_providers = set(labels_df[labels_df['PotentialFraud'] == 'Yes']['Provider'])
        
        # Add fraud label column
        df['is_fraudulent'] = df['Provider'].isin(fraudulent_providers).astype(int)

        return df

    def scale_numerical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale all numerical columns using StandardScaler"""
        df = df.copy()
        
        # Identify numerical columns (excluding is_fraudulent as it's our target)
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        exclude_cols = ['claimType', 'is_fraudulent', 'DiagnosisCategory', 'first_procedure']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Scale numerical columns
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        return df

    def combine_claims(self, inpatient_df: pd.DataFrame, outpatient_df: pd.DataFrame) -> pd.DataFrame:
        """Combine inpatient and outpatient claims with claim type indicator and stay duration"""
        # Define columns to keep
        columns_to_keep = [
            'BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
            'InscClaimAmtReimbursed', 'DeductibleAmtPaid'
        ]
        # Add all diagnosis code columns
        diagnosis_columns = [f'ClmDiagnosisCode_{i}' for i in range(1, 11)]
        columns_to_keep.extend(diagnosis_columns)
        # Add procedure code columns
        procedure_columns = [f'ClmProcedureCode_{i}' for i in range(1, 7)]
        columns_to_keep.extend(procedure_columns)
        
        # Add claim type column
        inpatient_df['claimType'] = 'inpatient'
        outpatient_df['claimType'] = 'outpatient'
        
        # Calculate stay duration for inpatient
        inpatient_df['StayDuration'] = (
            pd.to_datetime(inpatient_df['DischargeDt']) - 
            pd.to_datetime(inpatient_df['AdmissionDt'])
        ).dt.days
        # Set stay duration to 0 for outpatient
        outpatient_df['StayDuration'] = 0
        
        # Add new columns to keep list
        columns_to_keep.extend(['claimType', 'StayDuration'])
        
        # Select only the specified columns
        inpatient_df = inpatient_df[columns_to_keep]
        outpatient_df = outpatient_df[columns_to_keep]
        
        # Calculate target number of outpatient claims for 90/10 ratio
        target_outpatient = int(len(inpatient_df) * 9)  # 9 outpatient for every 1 inpatient
        
        # If we have more outpatient claims than needed, sample them
        if len(outpatient_df) > target_outpatient:
            outpatient_df = outpatient_df.sample(n=target_outpatient, random_state=42)
        
        # Combine the dataframes
        combined_df = pd.concat([inpatient_df, outpatient_df], ignore_index=True)
        
        # Calculate total cost
        combined_df = self.calculate_total_cost(combined_df)
        
        # Process diagnosis codes
        combined_df = self.process_diagnosis_codes(combined_df)
        
        # Process procedure codes
        combined_df = self.process_procedure_codes(combined_df)
        
        # Add fraud labels
        combined_df = self.process_fraud_labels(combined_df)
        
        # Enrich with beneficiary details
        combined_df = self.enrich_with_beneficiary(combined_df)
        
        # Process dates on the combined dataframe
        combined_df = self.process_claim_dates(combined_df)
        
        # Encode claim type
        combined_df['claimType'] = combined_df['claimType'].map({'inpatient': 0, 'outpatient': 1})
        
        # Remove ID columns
        combined_df = combined_df.drop(columns=['BeneID', 'ClaimID', 'Provider'])
        
        # Scale numerical columns
        combined_df = self.scale_numerical_columns(combined_df)
        
        # Ensure is_fraudulent is the last column
        cols = combined_df.columns.tolist()
        cols.remove('is_fraudulent')
        cols.append('is_fraudulent')
        combined_df = combined_df[cols]
        # Print fraud distribution
        fraud_counts = combined_df['is_fraudulent'].value_counts().sort_index()
        logger.info(f"Fraudulent claims: {fraud_counts.get(1, 0)}")
        logger.info(f"Non-fraudulent claims: {fraud_counts.get(0, 0)}")
        logger.info(f"Fraud ratio: {fraud_counts.get(1, 0) / fraud_counts.sum():.4f}")
        return combined_df

    def preprocess(self) -> None:
        """Main preprocessing function - currently only combining claims"""
        try:
            # Load only the necessary data
            logger.info("Loading inpatient and outpatient data...")
            inpatient_df = pd.read_csv(self.raw_dir / "inpatient.csv")
            outpatient_df = pd.read_csv(self.raw_dir / "outpatient.csv")
            
            # Combine claims
            logger.info("Combining claims...")
            claims_df = self.combine_claims(inpatient_df, outpatient_df)
            
            # Save processed data
            logger.info("Saving processed data...")
            claims_df.to_csv(self.processed_dir / "processed_claims.csv", index=False)
            
            # Save the scaler
            logger.info("Saving scaler...")
            joblib.dump(self.scaler, self.processed_dir / "scaler.joblib")
            
            logger.info("Preprocessing completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise

def preprocess(data_dir: str = "../../backend/data") -> None:
    """Main preprocessing function that combines and processes claims data"""
    # Initialize preprocessor
    preprocessor = HealthcareDataPreprocessor(data_dir)
    
    # Run preprocessing
    preprocessor.preprocess()

if __name__ == "__main__":
    preprocess()