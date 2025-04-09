import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

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

def main():
    """
    Main function to run the fraud detection process.
    """
    # Load and clean data
    df = load_and_clean_data('Fraud Detection Dataset.csv')
    
    # Encode categorical features
    categorical_columns = ['Transaction_Type', 'Device_Used', 'Location', 'Payment_Method']
    df_encoded = encode_categorical_features(df, categorical_columns)
    
    # Prepare data
    x, y = prepare_data(df_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_data_split(x, y)
    
    # Define models
    models = define_models()
    
    # Evaluate models
    evaluate_models(models, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()