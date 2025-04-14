import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load data from Excel file"""
    return pd.read_excel(file_path)


def encode_categorical_features(df, category_orders=None):
    """Encode categorical features using custom ordered Label Encoding
    
    Args:
        df: DataFrame to encode
        category_orders: Dictionary mapping column names to ordered lists of categories
                        Example: {'marital_status': ['single', 'married', 'divorced', 'widowed']}
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        if col in category_orders:
            # Use custom ordering if specified
            categories = category_orders[col]
            # Create a mapping from category to desired number
            category_map = {cat: idx for idx, cat in enumerate(categories)}
            # Apply the mapping
            df[col] = df[col].map(category_map)
            # Create a LabelEncoder for consistency
            label_encoders[col] = LabelEncoder()
            label_encoders[col].classes_ = np.array(categories)
        else:
            # Use default LabelEncoder if no custom order specified
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])
    
    return df, label_encoders

def scale_numerical_features(df, target_column=None, categorical_cols=None):
    """Scale numerical features using StandardScaler
    
    Args:
        df: DataFrame to scale
        target_column: Name of the target column to exclude from scaling
        categorical_cols: List of categorical columns to exclude from scaling
    """
    # Get numerical columns excluding the target and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if target_column and target_column in numerical_cols:
        numerical_cols = numerical_cols.drop(target_column)
    if categorical_cols:
        numerical_cols = numerical_cols.difference(categorical_cols)
    
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler

def split_data(df, target_column, test_size=0.2, val_size=0.2, random_state=42):
    """Split data into train, validation, and test sets"""
    # First split to separate out the test set
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # Then split the remaining data into train and validation
    train, val = train_test_split(
        train_val, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    # Separate features and target
    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    
    X_val = val.drop(columns=[target_column])
    y_val = val[target_column]
    
    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def process_date_columns(df):
    """Process date columns into numeric features"""
    # Convert to datetime
    df["ClaimDate"] = pd.to_datetime(df["ClaimDate"])
    
    # Create new columns
    # 1. Day of week (0-6, where 0 is Monday)
    df["ClaimDate_day_of_week"] = df["ClaimDate"].dt.dayofweek
    # 2. Year
    df["ClaimDate_year"] = df["ClaimDate"].dt.year
    # 3. Day of year (1-365/366)
    df["ClaimDate_day_of_year"] = df["ClaimDate"].dt.dayofyear

    # Drop original date column
    df = df.drop(columns=["ClaimDate"])
            
    return df

def preprocess_data(file_path, target_column, category_orders, columns_to_drop):
    """Main preprocessing function
    
    Args:
        file_path: Path to the data file
        target_column: Name of the target column
        category_orders: Dictionary mapping column names to ordered lists of categories
        columns_to_drop: List of column names to remove from the dataset
    """
    # Load data
    print("Loading data...")
    df = load_data(file_path)
    
    # Drop specified columns
    print(f"Dropping columns: {columns_to_drop}")
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Process date columns
    df = process_date_columns(df)
    
    # Encode categorical features
    print("Encoding categorical features...")
    df, label_encoders = encode_categorical_features(df, category_orders)
    
    # Get list of categorical columns (including binary ones)
    categorical_cols = list(category_orders.keys())
    
    # Scale numerical features (excluding target and categorical columns)
    print("Scaling numerical features...")
    df, scaler = scale_numerical_features(df, target_column, categorical_cols)
    
    # Split data
    print("Splitting data into train, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, target_column
    )
    
    # Save processed data
    print("Saving processed data...")
    processed_dir = '../../data/processed'
    
    X_train.to_csv(f'{processed_dir}/X_train.csv', index=False)
    X_val.to_csv(f'{processed_dir}/X_val.csv', index=False)
    X_test.to_csv(f'{processed_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{processed_dir}/y_train.csv', index=False)
    y_val.to_csv(f'{processed_dir}/y_val.csv', index=False)
    y_test.to_csv(f'{processed_dir}/y_test.csv', index=False)
    
    # Save preprocessing objects
    import joblib
    joblib.dump(label_encoders, f'{processed_dir}/label_encoders.joblib')
    joblib.dump(scaler, f'{processed_dir}/scaler.joblib')
    
    print("Preprocessing completed successfully!")
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    file_path = "../../data/raw_data.xlsx"
    target_column = "ClaimLegitimacy"
    
    # Define custom ordering for categorical columns
    category_orders = {
        'PatientGender': ['F', 'M'],
        'ClaimStatus': ['Pending', 'Denied', 'Approved'],
        'PatientMaritalStatus': ['Widowed', 'Divorced', 'Single', 'Married'],
        'PatientEmploymentStatus': ['Student', 'Unemployed', 'Retired', 'Employed'],
        'ClaimType': ['Emergency', 'Inpatient', 'Outpatient', 'Routine'],
        'ClaimSubmissionMethod': ['Phone', 'Paper', 'Online']
    }
    
    # Specify columns to remove
    columns_to_drop = [
        'ClaimID',
        'PatientID',
        'ProviderID',
        'ProviderLocation',
        'Cluster'
    ]

    preprocess_data(file_path, target_column, category_orders, columns_to_drop)