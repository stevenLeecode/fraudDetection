import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def load_and_clean_data(file_path):
    """
    Load data from CSV file and clean it by removing null values and duplicates.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df = pd.read_csv(file_path)
    print(f"Total missing values in dataset: {df.isnull().sum().sum()}")
    print(f"Total duplicated rows: {df.duplicated().sum()}")
    #Drop all rows where values are null.
    df.dropna(inplace=True)
    #Drop all rows where values are duplicated.
    df.drop_duplicates(inplace=True)
    return df

def encode_categorical_features(df, categorical_columns):
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorical_columns (list): List of categorical column names
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical features
    """
    #Encode categorical variables using LabelEncoder
    #LabelEncoder converts each unique value into an integer that represents the text it was before encoding.
    label_encoder = LabelEncoder()
    df_encoded = df.copy()
    for column in categorical_columns:
        df_encoded[column] = label_encoder.fit_transform(df[column])
    return df_encoded

def prepare_data(df_encoded):
    """
    Prepare data for model training by dropping irrelevant columns and splitting into features and target.
    
    Args:
        df_encoded (pd.DataFrame): Encoded dataframe
        
    Returns:
        tuple: Features (X) and target (y)
    """
    # Drop columns that shouldn't be used for training
    columns_to_drop = [
        'Transaction_ID',  # Identifier, not useful for prediction
        'Time_of_Transaction',  # Raw datetime column (we've already extracted features from it)
    ]
    
    # Only drop columns that exist in the dataframe
    columns_to_drop = [col for col in columns_to_drop if col in df_encoded.columns]
    df_encoded = df_encoded.drop(columns_to_drop, axis=1)

    # Ensure all remaining columns are numeric
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.kind in ['M', 'm']:
            print(f"Warning: Converting {col} to numeric...")
            try:
                df_encoded[col] = pd.to_numeric(df_encoded[col])
            except:
                print(f"Error: Could not convert {col} to numeric. Dropping column.")
                df_encoded = df_encoded.drop(col, axis=1)

    x = df_encoded.drop('Fraudulent', axis=1)
    y = df_encoded['Fraudulent']
    
    # Print feature names for debugging
    print("\nFeatures used for training:")
    print(x.columns.tolist())
    print("\nFeature types:")
    print(x.dtypes)
    
    return x, y

def train_test_data_split(x, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        x (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: Training and testing data splits
    """
    #Split dataset into 80% train and 20% test.
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def define_model():
    """
    Define SVM model for fraud detection with balanced class weights.
    """
    return svm.SVC(
        class_weight='balanced',  # Automatically adjust weights based on class frequencies
        kernel='rbf',
        probability=True,
        random_state=42,  # Regularization parameter - lower C means more regularization
    )

def analyze_class_distribution(y):
    """
    Analyze the class distribution
    """
    # Count the occurrences of each class
    class_counts = pd.Series(y).value_counts()
    imbalance_ratio = class_counts[0] / class_counts[1]  # legitimate / fraud
    
    print("Class distribution:")
    print(f"Legitimate transactions: {class_counts[0]} ({class_counts[0]/len(y)*100:.2f}%)")
    print(f"Fraudulent transactions: {class_counts[1]} ({class_counts[1]/len(y)*100:.2f}%)")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    return imbalance_ratio

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Train and evaluate the SVM model, printing performance metrics.
    
    Args:
        model: SVM model object
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training target
        y_test (pd.Series): Testing target
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model.__class__.__name__}")
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print labeled confusion matrix with TN, FP, FN, TP
    print("Confusion Matrix:")
    print("                  | Predicted Legitimate | Predicted Fraud")
    print("------------------|--------------------|----------------")
    print(f"Actual Legitimate |         TN={cm[0][0]}          |       FP={cm[0][1]}")
    print(f"Actual Fraud      |         FN={cm[1][0]}          |       TP={cm[1][1]}")
    
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision: ", precision_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred), "\n\n")

def create_temporal_features(df):
    """Add time-based features that may indicate fraud patterns"""
    
    # Convert timestamp to datetime if needed
    if 'Time_of_Transaction' in df.columns:
        df['Time_of_Transaction'] = pd.to_datetime(df['Time_of_Transaction'])
        
        # Extract numeric features from datetime
        df['Hour'] = df['Time_of_Transaction'].dt.hour
        df['Is_Night'] = ((df['Hour'] >= 0) & (df['Hour'] < 6)).astype(int)
        df['Is_Weekend'] = df['Time_of_Transaction'].dt.dayofweek.isin([5, 6]).astype(int)
        df['Day_Of_Month'] = df['Time_of_Transaction'].dt.day
        df['Is_Month_End'] = (df['Day_Of_Month'] >= 28).astype(int)
        df['Month'] = df['Time_of_Transaction'].dt.month
        df['Year'] = df['Time_of_Transaction'].dt.year
        
        # Convert time to seconds since midnight for a numeric representation
        df['Time_Of_Day_Seconds'] = (df['Time_of_Transaction'].dt.hour * 3600 + 
                                   df['Time_of_Transaction'].dt.minute * 60 +
                                   df['Time_of_Transaction'].dt.second)
    
    return df

def create_velocity_features(df):
    """Create features that detect unusual transaction patterns for each user"""
    
    # Make a copy to avoid modifying the original dataframe
    velocity_df = df.copy()
    
    if 'User_ID' in df.columns and 'Time_of_Transaction' in df.columns:
        # Ensure Time_of_Transaction is datetime
        velocity_df['Time_of_Transaction'] = pd.to_datetime(velocity_df['Time_of_Transaction'])
        
        # Sort by user and time
        velocity_df = velocity_df.sort_values(['User_ID', 'Time_of_Transaction'])
        
        # Calculate time since last transaction for each user
        velocity_df['Time_Since_Last_Txn'] = velocity_df.groupby('User_ID')['Time_of_Transaction'].diff().dt.total_seconds()
        
        # Calculate total amount in last 24 hours for each user
        if 'Transaction_Amount' in df.columns:
            # Create a unique index within each group to handle duplicate timestamps
            velocity_df['temp_index'] = velocity_df.groupby('User_ID').cumcount()
            
            def calculate_24h_amount(group):
                # Create a 24-hour window mask
                current_time = group['Time_of_Transaction']
                time_window = pd.Timedelta(hours=24)
                
                # Calculate rolling sum manually
                amounts = []
                for idx in range(len(group)):
                    end_time = current_time.iloc[idx]
                    start_time = end_time - time_window
                    # Sum amounts within the 24-hour window
                    window_mask = (group['Time_of_Transaction'] > start_time) & (group['Time_of_Transaction'] <= end_time)
                    window_sum = group.loc[window_mask, 'Transaction_Amount'].sum()
                    amounts.append(window_sum)
                
                return pd.Series(amounts, index=group.index)
            
            # Apply the calculation for each user
            velocity_df['Amount_24h'] = velocity_df.groupby('User_ID', group_keys=False).apply(calculate_24h_amount)
            
            # Drop temporary index
            velocity_df = velocity_df.drop('temp_index', axis=1)
    
    return velocity_df

def create_amount_features(df):
    """Create features based on transaction amounts"""
    
    if 'Transaction_Amount' in df.columns and 'User_ID' in df.columns:
        # Calculate user's average transaction amount
        user_avg = df.groupby('User_ID')['Transaction_Amount'].transform('mean')
        
        # Calculate how much this transaction deviates from user's average
        df['Amount_User_Deviation'] = df['Transaction_Amount'] / user_avg
        
        # Flag high-value transactions (above 95th percentile)
        high_value_threshold = df['Transaction_Amount'].quantile(0.95)
        df['Is_High_Value'] = (df['Transaction_Amount'] > high_value_threshold).astype(int)
        
        # Flag round amounts (often suspicious)
        df['Is_Round_Amount'] = ((df['Transaction_Amount'] % 10) == 0).astype(int)
        
        # Transaction amount percentile (relative to all transactions)
        df['Amount_Percentile'] = df['Transaction_Amount'].rank(pct=True)
    
    return df

def create_user_profile_features(df):
    """Create features that capture user behavior patterns"""
    
    if 'User_ID' in df.columns:
        # Account age and previous fraudulent transactions are already in the dataset
        
        # User's distinct locations
        if 'Location' in df.columns:
            location_counts = df.groupby('User_ID')['Location'].transform('nunique')
            df['User_Location_Count'] = location_counts
        
        # User's distinct device types
        if 'Device_Used' in df.columns:
            device_counts = df.groupby('User_ID')['Device_Used'].transform('nunique')
            df['User_Device_Count'] = device_counts
        
        # Check if transaction location is unusual for this user
        if 'Location' in df.columns:
            # Get common locations for each user (those used more than once)
            user_common_locations = df.groupby(['User_ID', 'Location']).size().reset_index()
            user_common_locations = user_common_locations[user_common_locations[0] > 1]
            user_common_locations = user_common_locations[['User_ID', 'Location']]
            
            # Merge with original dataframe
            df = df.merge(user_common_locations, on=['User_ID', 'Location'], how='left', indicator=True)
            
            # If _merge is 'left_only', it's an uncommon location
            df['Is_Uncommon_Location'] = (df['_merge'] == 'left_only').astype(int)
            df = df.drop('_merge', axis=1)
    
    return df

def engineer_features(df):
    """Apply all feature engineering functions"""
    
    df = create_temporal_features(df)
    df = create_velocity_features(df)
    df = create_amount_features(df)
    df = create_user_profile_features(df)
    
    # Fill missing values created during feature engineering
    df = df.fillna(0)
    
    return df

def main():
    """
    Main function to run the fraud detection process.
    """
    # Load and clean data
    print("Starting data loading...")
    df = load_and_clean_data('Fraud Detection Dataset.csv')
    print("Data loading complete.")
    
    # Engineer features
    print("\nStarting feature engineering...")
    df = engineer_features(df)
    print("Feature engineering complete.")
    
    # Encode categorical features
    print("\nStarting categorical feature encoding...")
    categorical_columns = ['Transaction_Type', 'Device_Used', 'Location', 'Payment_Method']
    df_encoded = encode_categorical_features(df, categorical_columns)
    print("Categorical feature encoding complete.")
    
    # Prepare data
    print("\nStarting data preparation...")
    x, y = prepare_data(df_encoded)
    print("Data preparation complete.")
    
    # Analyze class distribution
    print("\nStarting class distribution analysis...")
    imbalance_ratio = analyze_class_distribution(y)
    print("Class distribution analysis complete.")
    
    # Split data
    print("\nStarting train-test split...")
    X_train, X_test, y_train, y_test = train_test_data_split(x, y)
    print("Train-test split complete.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Analyze class distribution in training set
    print("Class distribution in training set:")
    print(y_train.value_counts(normalize=True))
    

    # Define the model with a more appropriate scale_pos_weight
    # Use approximately the imbalance ratio, but not as extreme
    scale_pos_weight_value = min(imbalance_ratio, 10)  # Cap at 10 to avoid extreme predictions
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight_value,
        random_state=42
    )

    # Train and evaluate
    xgb_model.fit(X_train, y_train)
    
    # Get probability scores for threshold tuning
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    
    # Try different thresholds to find optimal balance
    print("\nPrecision-Recall trade-off at different thresholds:")
    print("Threshold | Precision | Recall | Accuracy | F1 Score")
    print("----------|-----------|--------|----------|--------")
    
    best_f1 = 0
    best_threshold = 0.5
    
    # Try different thresholds from 0.1 to 0.9
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        
        # Calculate F1 score (harmonic mean of precision and recall)
        f1 = 0 if (prec + rec == 0) else 2 * (prec * rec) / (prec + rec)
        
        # Print results
        print(f"{threshold:.1f}      | {prec:.4f}    | {rec:.4f} | {acc:.4f}   | {f1:.4f}")
        
        # Keep track of best F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Use the threshold with the best F1 score
    print(f"\nUsing best threshold: {best_threshold:.1f} with F1 score: {best_f1:.4f}")
    y_pred = (y_prob >= best_threshold).astype(int)
    
    # Evaluate with the best threshold
    print("\nXGBoost Results with optimized threshold:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("F1 Score:", best_f1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("                  | Predicted Legitimate | Predicted Fraud")
    print("------------------|--------------------|----------------")
    print(f"Actual Legitimate |         TN={cm[0][0]}          |       FP={cm[0][1]}")
    print(f"Actual Fraud      |         FN={cm[1][0]}          |       TP={cm[1][1]}")

    # Feature importance
    feature_names = X_train.columns
    importances = xgb_model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    # Grid search for XGBoost
    print("\nPerforming grid search for optimal hyperparameters...")
    
    # Define the model
    xgb_model_for_grid = xgb.XGBClassifier(random_state=42)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'scale_pos_weight': [1, 5, scale_pos_weight_value]  # Include our calculated value
    }

    # Perform grid search optimizing for F1 score instead of just recall
    grid_search = GridSearchCV(xgb_model_for_grid, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    
    # Get the best model
    best_xgb = grid_search.best_estimator_
    
    # Get probability scores
    y_prob_best = best_xgb.predict_proba(X_test)[:, 1]
    
    # Use the same threshold optimization for the best model
    y_pred_best = (y_prob_best >= best_threshold).astype(int)

    # Evaluate
    print("\nTuned XGBoost Results:")
    acc_best = accuracy_score(y_test, y_pred_best)
    prec_best = precision_score(y_test, y_pred_best, zero_division=0)
    rec_best = recall_score(y_test, y_pred_best)
    f1_best = 2 * (prec_best * rec_best) / (prec_best + rec_best) if (prec_best + rec_best > 0) else 0
    
    print("Accuracy:", acc_best)
    print("Recall:", rec_best)
    print("Precision:", prec_best)
    print("F1 Score:", f1_best)

    # Confusion Matrix for best model
    cm_best = confusion_matrix(y_test, y_pred_best)
    print("\nConfusion Matrix for Best Model:")
    print("                  | Predicted Legitimate | Predicted Fraud")
    print("------------------|--------------------|----------------")
    print(f"Actual Legitimate |         TN={cm_best[0][0]}          |       FP={cm_best[0][1]}")
    print(f"Actual Fraud      |         FN={cm_best[1][0]}          |       TP={cm_best[1][1]}")

if __name__ == "__main__":
    main()