import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Corrected line
from sklearn.compose import ColumnTransformer # <--- THIS IS THE MISSING LINE!
from sklearn.cluster import KMeans
import os
# No xverse imports should be present anymore

# --- DATA PATH ---
DATA_PATH = "data/raw/data.csv" 
# -----------------

# --- TASK 4: Proxy Target Variable Engineering (RFM and K-Means) ---
def calculate_rfm_and_proxy(df):
    """Calculates RFM metrics, clusters customers, and assigns the 'is_high_risk' proxy label."""
    
    # Ensure 'TransactionStartTime' is datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # 1. Define Snapshot Date
    snapshot_date = df['TransactionStartTime'].max() + pd.DateOffset(days=1)
    
    # 2. Calculate RFM Metrics
    rfm_df = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Value', 'sum') 
    )
    
    # 3. Scale RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

    # 4. K-Means Clustering (k=3) 
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # 5. Define High-Risk Label 
    cluster_means = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_means['Monetary'].idxmin() # Lowest Monetary cluster = Highest Risk

    # 6. Assign binary target variable ('is_high_risk')
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
    
    return rfm_df[['is_high_risk']]

# --- TASK 3: Feature Engineering Pipeline ---
# --- TASK 3: Feature Engineering Pipeline (Corrected) ---
# --- TASK 3: Feature Engineering Pipeline (One-Hot Encoding FIX) ---
def preprocess_and_feature_engineer(df, proxy_df):
    
    # 1. Merge the Target Variable 
    df = df.merge(proxy_df, on='CustomerId', how='left')
    
    # 2. Customer-Level Aggregation (Feature Engineering)
    customer_df = df.groupby('CustomerId').agg(
        is_high_risk=('is_high_risk', 'max'),
        
        # Aggregate Numerical Features
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        transaction_count=('TransactionId', 'count'),
        std_amount=('Amount', 'std'),
        
        # Categorical Features 
        most_frequent_category=('ProductCategory', lambda x: x.mode()[0] if not x.empty else 'Missing'),
        most_frequent_provider=('ProviderId', lambda x: x.mode()[0] if not x.empty else 'Missing'),
        most_frequent_channel=('ChannelId', lambda x: x.mode()[0] if not x.empty else 'Missing'),
        
    ).reset_index()

    customer_df = customer_df.drop('CustomerId', axis=1)

    X = customer_df.drop('is_high_risk', axis=1)
    y = customer_df['is_high_risk']
    
    # Handle NaNs from std_amount before preprocessing
    X['std_amount'] = X['std_amount'].fillna(0)

    # Define features by type
    numerical_features = ['total_amount', 'avg_amount', 'transaction_count', 'std_amount']
    categorical_features = ['most_frequent_category', 'most_frequent_provider', 'most_frequent_channel']

    # 3. Create the robust Scikit-learn Preprocessing Pipeline 
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaling', StandardScaler(), numerical_features), # Standardization
            # One-Hot Encoding for categorical features
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) 
        ],
        remainder='passthrough'
    )
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X, y)
    
    # Get the feature names for the final dataframe
    feature_names = numerical_features + list(preprocessor.named_transformers_['onehot'].get_feature_names_out(categorical_features))
    
    # Convert back to DataFrame for clean output
    X_final = pd.DataFrame(X_processed, columns=feature_names)
    
    return X_final, y
# --- FIX ENDS HERE ---

def get_model_ready_data():
    """Main function to run all data processing steps."""
    print("Starting data loading...")
    try:
        df_raw = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Raw data not found at {DATA_PATH}. Please ensure 'data.csv' is in the data/raw/ folder.")
        return None, None
        
    print("Creating proxy target variable (Task 4)...")
    proxy_target_df = calculate_rfm_and_proxy(df_raw.copy())
    
    print("Running feature engineering (Task 3)...")
    X, y = preprocess_and_feature_engineer(df_raw, proxy_target_df)
    
    print(f"Data processing complete. X shape: {X.shape}, y shape: {y.shape}")
    return X, y

if __name__ == '__main__':
    os.makedirs("data/processed", exist_ok=True) 
    X_final, y_final = get_model_ready_data()
    if X_final is not None:
        print("Processed data saved to data/processed/")


