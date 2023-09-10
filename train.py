import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

df = pd.read_csv('World University Rankings 2023.csv')
df = df.head(198)


def preprocess_data(df):
    # Replace empty strings with NaN
    df['International Student'].replace('%', np.nan, inplace=True)
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df.dropna(inplace=True)
    # Remove commas and convert 'Number of students' to integer
    df['No of student'] = df['No of student'].str.replace(',', '').astype(int)

    # Remove '%' and convert 'International Student' to float, then to proportion
    df['International Student'] = df['International Student'].str.rstrip(
        '%').astype(float) / 100.0

    # Split 'Female:Male Ratio' into two columns
    df['Female Ratio'], df['Male Ratio'] = zip(
        *df['Female:Male Ratio'].str.split(':').tolist())
    df['Female Ratio'] = df['Female Ratio'].astype(float)
    df['Male Ratio'] = df['Male Ratio'].astype(float)

    # Drop the original 'Female:Male Ratio' column
    df = df.drop('Female:Male Ratio', axis=1)

    return df


df = preprocess_data(df)

# Drop the 'Name of University' column
df = df.drop('Name of University', axis=1)

# Split the data into train and test sets
X = df.drop(columns=['University Rank', 'OverAll Score', 'Location'], axis=1)
y = df['University Rank']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
print(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
print(X_test)
y_pred = model.predict(X_test)

# Calculate the RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error: {rmse}")
print(X.columns)
print(X.dtypes)


# Save the model to a file
with open('university_ranking_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('university_ranking_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
