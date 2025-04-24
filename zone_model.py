from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import joblib
import numpy as np
import pandas as pd

def train_model_zone(data, save_model_path):
    # Extracting features and labels
    X = data[['Zone Above', 'Zone Middle', 'Zone Below']].values
    y = data['Zone'].map({'Above': 1, 'Middle': 2, 'Below': 3}).values

    # Initializing SVC model
    svc_model = SVC(C=100, gamma=0.0001, kernel='rbf')

    # Cross-validation
    accuracy_scores = cross_val_score(svc_model, X, y, cv=5)

    # Calculating mean accuracy
    mean_accuracy = np.mean(accuracy_scores)

    # Fitting the model to the entire dataset
    svc_model.fit(X, y)

    # Save the model to a .pkl file
    #joblib.dump(svc_model, save_model_path)

    return mean_accuracy

# # Assuming you have your data stored in a DataFrame called A
# model_save_path = "model_zone.pkl"
# A=pd.read_csv("dataset_csv/dataset.csv")
# accuracy = train_model_zone(A, model_save_path)
# #print("Mean Accuracy:", accuracy)
# #print("Trained SVC Model saved at:", model_save_path)
