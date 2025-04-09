import pandas as pd

df = pd.read_csv('Fraud Detection Dataset.csv')
print(f"Total missing values in dataset: {df.isnull().sum().sum()}")
print(f"Total duplicated rows: {df.duplicated().sum()}")
#Drop all rows where values are null.
df.dropna(inplace=True)
#Drop all rows where values are duplicated.
df.drop_duplicates(inplace=True)

#Encode categorical values to booleans 
categorical_columns = ['Transaction_Type', 'Device_Used', 'Location', 'Payment_Method']
# Convert categorical columns to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

#Convert boolean columns to 0/1 for our ML models to easily process.
boolean_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[boolean_cols] = df_encoded[boolean_cols].astype(int)

#Drop Transaction_ID since I think they are irrelevant. 
#Possibly don't need to do this and we can keep this not sure.
#I think its important to keep the User_ID because we may be able to find a relationship
#with the same people having fraudulent transactions.
df_encoded.drop(['Transaction_ID'], axis=1, inplace=True)

print(df_encoded.head())