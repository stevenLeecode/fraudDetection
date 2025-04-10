import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

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
    #Drop Transaction_ID since I think they are irrelevant. 
    #Possibly don't need to do this and we can keep this not sure.
    #I think its important to keep the User_ID because we may be able to find a relationship
    #with the same people having fraudulent transactions.
    df_encoded.drop(['Transaction_ID'], axis=1, inplace=True)

    x = df_encoded.drop('Fraudulent', axis=1)
    y = df_encoded['Fraudulent']
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
    Define machine learning models for fraud detection.
    
    Returns:
        list: List of initialized model objects
    """
    #Define our models.
    svm_model = svm.SVC()
    logistic_model = LogisticRegression(max_iter=100000)
    decision_tree_model = DecisionTreeClassifier(criterion='entropy')
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
import numpy as np

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
    # Define sampling ratios to test
    # These are the ratios of minority class to majority class
    ratios = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    # Simple model for testing
    model = LogisticRegression(class_weight='balanced', max_iter=10000, random_state=42)
    
    # Store results
    results = []
    
    for ratio in ratios:
        print(f"\nTesting ratio: {ratio}")
        
        # Convert ratio to sampling_strategy
        # For example, 0.5 means we'll have half as many minority samples as majority
        minority_class = 1  # assuming 1 is the fraud class
        sampling_strategy = {minority_class: int(np.sum(y_train == 0) * ratio)}
        
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

def main():
    """
    Main function integrating class imbalance handling
    """
    # Load and clean data
    df = load_and_clean_data('Fraud Detection Dataset.csv')
    
    # Encode categorical features
    categorical_columns = ['Transaction_Type', 'Device_Used', 'Location', 'Payment_Method']
    df_encoded = encode_categorical_features(df, categorical_columns)
    
    # Prepare data
    X, y = prepare_data(df_encoded)
    
    # Analyze class distribution
    imbalance_ratio = analyze_class_distribution(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_data_split(X, y)
    
    # Compare resampling techniques and find the best one
    best_technique = compare_resampling_techniques(X_train, X_test, y_train, y_test)
    
    # Find optimal sampling ratio for the best technique
    best_ratio = find_optimal_sampling_ratio(X_train, X_test, y_train, y_test, best_technique)
    
    # If baseline performed best, use original data
    if best_technique == 'baseline':
        print("\nUsing original data since baseline performed best")
        X_train_resampled, y_train_resampled = X_train, y_train
    else:
        # Apply the best resampling technique with the optimal ratio
        minority_class = 1  # assuming 1 is the fraud class
        sampling_strategy = {minority_class: int(np.sum(y_train == 0) * best_ratio)}
        X_train_resampled, y_train_resampled = apply_resampling(
            X_train, y_train,
            technique=best_technique,
            sampling_strategy=sampling_strategy
        )
    
    # Define models with class_weight='balanced'
    models = define_models()
    
    # Evaluate models with original imbalanced data (as baseline)
    print("\nEvaluating models with original imbalanced data:")
    evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # Evaluate models with resampled data
    print("\nEvaluating models with resampled data:")
    evaluate_models(models, X_train_resampled, X_test, y_train_resampled, y_test)
if __name__ == "__main__":
    main()