# Import necessary libraries
from sklearn.model_selection import train_test_split
import pandas as pd
from NBMultinomial import NBMultinomial  # Import the NBMultinomial 
from label_encoder import label_encoder  # Import the label_encoder

# Read data from CSV file
data = pd.read_csv(r"Data.csv", encoding='ISO-8859-1')

# Separate features (X) and target variable (y)
X = data.drop("v1", axis=1)
y = data["v1"]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0018, random_state=42)

# Initialize and train the label encoder
lab = label_encoder()
lab.train(X_train)

# Transform the train data using the label encoder
transformed_train_data = lab.transform(X_train)

# Calculate the sum of each feature in the transformed train data
info_train_data = transformed_train_data.sum(axis=0).sort_values(ascending=False)
info_train_data = pd.DataFrame(info_train_data)

# Identify features with low occurrence (<3) and high occurrence (>100)
index_low = info_train_data[0].where(info_train_data[0] < 3).dropna().index
index_high = info_train_data[0].where(info_train_data[0] > 100).dropna().index

# Drop features with low and high occurrences from train and test data
transformed_train_data = transformed_train_data.drop(index_high, axis=1).drop(index_low, axis=1)
transformed_test_data = lab.transform(X_test).drop(index_high, axis=1).drop(index_low, axis=1)

# Initialize and train the NBMultinomial model
model = NBMultinomial()
model.train_data(transformed_train_data, y_train)

# Make predictions on test data
y_pred = model.predict(transformed_test_data)
print(y_pred)

# Evaluate the model's performance
point = model.score(y_pred, y_test)
print(point)

# The commented section below appears to be code for sending email and making predictions on email input

"""
while True:
    email_input = input("Email:\n")
    if email_input == "0":
        break
    email_input = pd.DataFrame([email_input])
    print(email_input)
    email_input_transformed = lab.transform(email_input)
    email_input_transformed = email_input_transformed.drop(index_high, axis=1).drop(index_low, axis=1)
    email_predict = model.predict(email_input_transformed)
    print(email_predict)
"""
