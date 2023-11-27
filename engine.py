import pandas as pd
import numpy as np
import joblib

# General scikitLearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# SVM Model
from sklearn.svm import SVC

# DecisionTree Model
from sklearn.tree import DecisionTreeClassifier

# Neural Netwok Model
from sklearn.neural_network import MLPClassifier

# KNeighbors Model
from sklearn.neighbors import KNeighborsClassifier

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression






def read_data():
    df = pd.read_csv('df_new.csv')
    df_use = df.drop(columns=['city_code', 'Settlement_Council'])

    array_list = []

    # Split the DataFrame into chunks
    chunks = np.array_split(df_use.iloc[:, 2:-1].values, 251)

    # Iterate over chunks and append to the array_list
    for chunk in chunks:
        array_list.append(chunk)

    X_3d = np.stack(array_list)
    X = X_3d.reshape(X_3d.shape[0], -1)

    type_list = []
    for i in range(0,len(df), 21):
        type_list.append(df_use['last_city_type'][i])

    return df, X, type_list


def model_SVM():
    df, X, y = read_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Support Vector Machine (SVM) classifier
    svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

    # Train the classifier on the training data
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = svm_classifier.predict(X_test)

    # Evaluate the performance of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)


def model_DecisionTree():
    df, X, y = read_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Decision Tree classifier
    dt_classifier = DecisionTreeClassifier()

    # Train the classifier on the training data
    dt_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = dt_classifier.predict(X_test)

    # Evaluate the performance of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)


def model_NNC():
    df, X, y = read_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Neural Network classifier
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 60, 30, 3), max_iter=1000,
                                   learning_rate_init=0.001, warm_start=False, random_state=42)

    # Train the classifier on the training data
    mlp_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = mlp_classifier.predict(X_test)

    # Evaluate the performance of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)

    # Save the model to a file
    joblib.dump(mlp_classifier, 'municipal_crime_classifier.pkl')

def model_KNeighbors():
    df, X, y = read_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a K-Nearest Neighbors classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=5)

    # Train the classifier on the training data
    knn_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = knn_classifier.predict(X_test)

    # Evaluate the performance of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)


def model_RandomForest():
    df, X, y = read_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42)

    # Train the classifier on the training data
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the performance of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)


def model_LogisticsRegression():
    df, X, y = read_data()

    # Assuming X and y are your features and labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Logistic Regression classifier
    logreg_classifier = LogisticRegression(random_state=42)

    # Train the classifier
    logreg_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = logreg_classifier.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

