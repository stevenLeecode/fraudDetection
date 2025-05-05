import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import numpy as np

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

def define_models():
    """
    Define machine learning models for fraud detection with balanced class weights.
    """
    svm_model = svm.SVC(
        class_weight='balanced',
        kernel='rbf',
        probability=True,
        random_state=42
    )
    
    logistic_model = LogisticRegression(
        class_weight='balanced',
        max_iter=10000,
        solver='liblinear',
        random_state=42
    )
    
    decision_tree_model = DecisionTreeClassifier(
        class_weight='balanced',
        criterion='entropy',
        random_state=42
    )
    
    return [logistic_model, decision_tree_model, svm_model]

def analyze_class_distribution(y):
    """
    Analyze and visualize the class distribution
    """
    # Count the occurrences of each class
    class_counts = pd.Series(y).value_counts()
    imbalance_ratio = class_counts[0] / class_counts[1]  # legitimate / fraud
    
    print("Class distribution:")
    print(f"Legitimate transactions: {class_counts[0]} ({class_counts[0]/len(y)*100:.2f}%)")
    print(f"Fraudulent transactions: {class_counts[1]} ({class_counts[1]/len(y)*100:.2f}%)")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # If you want to visualize it
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.bar(['Legitimate', 'Fraud'], class_counts)
    plt.title('Class Distribution in Fraud Detection Dataset')
    plt.ylabel('Count')
    plt.show()
    
    return imbalance_ratio

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Train and evaluate models, printing performance metrics.
    
    Args:
        models (list): List of model objects
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training target
        y_test (pd.Series): Testing target
    """
    #Run through each model and print out the confusion matrix, accuracy, precision and recall.
    for model in models:
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


from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN

def apply_resampling(X_train, y_train, technique='smote', sampling_strategy=None):
    """
    Apply various resampling techniques to address class imbalance
    
    Args:
        X_train: Training features
        y_train: Training labels
        technique: Resampling technique to use ('smote', 'adasyn', 'undersample', 'smote_tomek', 'smote_enn')
        sampling_strategy: Ratio of minority to majority class (default is 'auto')
    
    Returns:
        X_resampled, y_resampled: Resampled feature and label datasets
    """
    # If no sampling strategy is provided, use 'auto'
    if sampling_strategy is None:
        sampling_strategy = 'auto'
    
    # Select the resampling technique
    if technique == 'smote':
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    elif technique == 'adasyn':
        sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
    elif technique == 'undersample':
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    elif technique == 'smote_tomek':
        sampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=42)
    elif technique == 'smote_enn':
        sampler = SMOTEENN(sampling_strategy=sampling_strategy, random_state=42)
    else:
        raise ValueError(f"Unknown resampling technique: {technique}")
    
    # Apply the resampling
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    # Calculate class distribution after resampling
    resampled_counts = pd.Series(y_resampled).value_counts()
    print("\nResampled class distribution:")
    print(f"Legitimate transactions: {resampled_counts[0]} ({resampled_counts[0]/len(y_resampled)*100:.2f}%)")
    print(f"Fraudulent transactions: {resampled_counts[1]} ({resampled_counts[1]/len(y_resampled)*100:.2f}%)")
    print(f"New ratio: {resampled_counts[0]/resampled_counts[1]:.2f}:1")
    
    return X_resampled, y_resampled


from sklearn.metrics import classification_report, f1_score

def compare_resampling_techniques(X_train, X_test, y_train, y_test):
    """
    Compare different resampling techniques using a simple model
    """
    # Define the techniques to test
    techniques = ['smote', 'adasyn', 'undersample', 'smote_tomek', 'smote_enn']
    
    # Define a simple model for comparison
    model = LogisticRegression(class_weight='balanced', max_iter=10000, random_state=42)
    
    # Store results
    results = []
    
    # Baseline (no resampling)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    baseline_f1 = f1_score(y_test, y_pred)
    print("Baseline (No resampling):")
    print(classification_report(y_test, y_pred))
    results.append(('baseline', baseline_f1))
    
    # Test each resampling technique
    for technique in techniques:
        print(f"\nTesting: {technique}")
        X_resampled, y_resampled = apply_resampling(X_train, y_train, technique=technique)
        
        # Train the model
        model.fit(X_resampled, y_resampled)
        # Evaluate
        y_pred = model.predict(X_test)
        technique_f1 = f1_score(y_test, y_pred)
        print(classification_report(y_test, y_pred))
        results.append((technique, technique_f1))
    
    # Compare results
    print("\nF1 Score Comparison:")
    for technique, f1 in results:
        print(f"{technique}: {f1:.4f}")
    
    # Find the best technique
    best_technique = max(results, key=lambda x: x[1])
    print(f"\nBest technique: {best_technique[0]} (F1 = {best_technique[1]:.4f})")
    
    return best_technique[0]

def find_optimal_sampling_ratio(X_train, X_test, y_train, y_test, best_technique):
    """
    Find the optimal sampling ratio for the best resampling technique
    """
    if best_technique == 'baseline':
        print("Baseline (no resampling) performed best. Skipping ratio optimization.")
        return None
        
    # Get class counts
    n_majority = np.sum(y_train == 0)
    n_minority = np.sum(y_train == 1)
    
    if best_technique == 'undersample':
        # For undersampling, ratios represent how much of the majority class to keep
        # We can't have more samples than the minority class
        ratios = [0.5, 1.0, 2.0, 5.0, 10.0]  # multiplier of minority class size
        print("\nFor undersampling, ratios represent majority:minority ratio")
    else:
        # For oversampling, ratios represent minority:majority ratio
        ratios = [0.1, 0.25, 0.5, 0.75, 1.0]
        print("\nFor oversampling, ratios represent minority:majority ratio")
    
    # Simple model for testing
    model = LogisticRegression(class_weight='balanced', max_iter=10000, random_state=42)
    
    # Store results
    results = []
    
    for ratio in ratios:
        print(f"\nTesting ratio: {ratio}")
        
        if best_technique == 'undersample':
            # For undersampling, we're reducing majority class
            n_samples_majority = int(n_minority * ratio)
            sampling_strategy = {0: n_samples_majority}  # 0 is majority class
        else:
            # For oversampling, we're increasing minority class
            n_samples_minority = int(n_majority * ratio)
            sampling_strategy = {1: n_samples_minority}  # 1 is minority class
        
        # Apply resampling
        X_resampled, y_resampled = apply_resampling(
            X_train, y_train, 
            technique=best_technique,
            sampling_strategy=sampling_strategy
        )
        
        # Train and evaluate
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test)
        ratio_f1 = f1_score(y_test, y_pred)
        
        print(f"F1 Score: {ratio_f1:.4f}")
        results.append((ratio, ratio_f1))
    
    # Find the best ratio
    best_ratio = max(results, key=lambda x: x[1])
    print(f"\nBest ratio: {best_ratio[0]} (F1 = {best_ratio[1]:.4f})")
    
    return best_ratio[0]

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
    df = load_and_clean_data('Fraud Detection Dataset.csv')
    
    # Engineer features
    print("\nEngineering features...")
    df = engineer_features(df)
    
    # Encode categorical features
    print("\nEncoding categorical features...")
    categorical_columns = ['Transaction_Type', 'Device_Used', 'Location', 'Payment_Method']
    df_encoded = encode_categorical_features(df, categorical_columns)
    
    # Prepare data
    print("\nPreparing data for training...")
    x, y = prepare_data(df_encoded)
    
    # Analyze class distribution
    print("\nAnalyzing class distribution...")
    analyze_class_distribution(y)
    
    # Split data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_data_split(x, y)
    
    # Compare different resampling techniques and find the best one
    print("\nComparing resampling techniques...")
    best_technique = compare_resampling_techniques(X_train, X_test, y_train, y_test)
    
    # Find optimal sampling ratio for the best technique
    print("\nFinding optimal sampling ratio...")
    best_ratio = find_optimal_sampling_ratio(X_train, X_test, y_train, y_test, best_technique)
    
    # Apply the best resampling technique
    if best_technique == 'baseline':
        print("\nUsing original data since baseline performed best")
        X_train_resampled, y_train_resampled = X_train, y_train
    else:
        print(f"\nApplying {best_technique} with ratio {best_ratio}")
        if best_technique == 'undersample':
            # For undersampling, ratio represents how many majority samples per minority sample
            n_samples_majority = int(np.sum(y_train == 1) * best_ratio)  # multiply minority count by ratio
            sampling_strategy = {0: n_samples_majority}  # 0 is majority class
        else:
            # For oversampling, ratio represents how many minority samples per majority sample
            n_samples_minority = int(np.sum(y_train == 0) * best_ratio)  # multiply majority count by ratio
            sampling_strategy = {1: n_samples_minority}  # 1 is minority class
            
        X_train_resampled, y_train_resampled = apply_resampling(
            X_train, y_train, 
            technique=best_technique,
            sampling_strategy=sampling_strategy
        )
    
    # Define models with balanced class weights
    print("\nDefining models...")
    models = define_models()
    
    # Evaluate models with original imbalanced data (as baseline)
    print("\nEvaluating models with original imbalanced data:")
    evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # Evaluate models with resampled data
    print("\nEvaluating models with resampled data:")
    evaluate_models(models, X_train_resampled, X_test, y_train_resampled, y_test)

if __name__ == "__main__":
    main()