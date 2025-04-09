import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Fraud Detection Dataset.csv')
print(f"Total missing values in dataset: {df.isnull().sum().sum()}")
print(f"Total duplicated rows: {df.duplicated().sum()}")
#Drop all rows where values are null.
df.dropna(inplace=True)
#Drop all rows where values are duplicated.
df.drop_duplicates(inplace=True)

#Encode categorical variables using LabelEncoder
#LabelEncoder converts each unique value into an integer that represents the text it was before encoding.
categorical_columns = ['Transaction_Type', 'Device_Used', 'Location', 'Payment_Method']
label_encoder = LabelEncoder()
df_encoded = df.copy()
for column in categorical_columns:
    df_encoded[column] = label_encoder.fit_transform(df[column])

#Drop Transaction_ID since I think they are irrelevant. 
#Possibly don't need to do this and we can keep this not sure.
#I think its important to keep the User_ID because we may be able to find a relationship
#with the same people having fraudulent transactions.
df_encoded.drop(['Transaction_ID'], axis=1, inplace=True)

x = df_encoded.drop('Fraudulent', axis=1)
y = df_encoded['Fraudulent']

#Split dataset into 80% train and 20% test.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

svm = svm.SVC()
logistic_model = LogisticRegression(max_iter=10000)
decision_tree_model = DecisionTreeClassifier()

ml_models = [logistic_model, decision_tree_model, svm]
for model in ml_models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model.__class__.__name__}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision: ", precision_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred), "\n\n")