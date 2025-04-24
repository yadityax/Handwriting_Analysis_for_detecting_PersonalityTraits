import pandas as pd
import numpy as np
import random
import os
import pickle
from sklearn.metrics.pairwise import rbf_kernel
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import joblib
import numpy as np
import pandas as pd



def train_zone(data):
    #if os.path.exists("model_zone.pkl"):
    #    os.remove("model_zone.pkl")

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

def train_pressure(data):
    #if os.path.exists("model_pressure.pkl"):
    #    os.remove("model_pressure.pkl")
    X = data[['Average']].values
    y = data['Pressure'].map({'Heavy': 1, 'Medium': 2, 'Light': 3}).values

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

def train_margin(data):
    # Extracting features and labels
    X = data[['margin']].values
    y = data['Top_Margin'].map({'Narrow': 1, 'Big': 2}).values

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

def train_letterSize(data):
    # Extracting features and labels
    X = data[['size']].values
    y = data['Letter_Size'].map({'Small': 1, 'Medium': 2, 'Big': 3}).values

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


def predict_zone(x):
    svc_model = joblib.load("model_zone.pkl")
    return svc_model.predict(x)

def predict_pressure(x):
    svc_model = joblib.load("model_pressure.pkl")
    return svc_model.predict(x)

def predict_letterSize(x):
    svc_model = joblib.load("model_letter-size.pkl")
    return svc_model.predict(x)

def predict_margin(x):
    svc_model = joblib.load("model_margin.pkl")
    return svc_model.predict(x)



def result_zone(_class):
    personality = ""
    if _class == 1:
        personality = "The author pays more attention to the spiritual aspects, dreams, hopes and ambitions in his life. The author prefers to do thinking activities and think about his future"
    elif _class == 2:
        personality = "Writers are more concerned with their current lives and find it difficult to make their long-term plans"
    elif _class == 3:
        personality = "The author gives more importance to the physical aspects of life and relies more on his muscles than his brain"

    return personality

def result_pressure(_class):
    personality = ""
    if _class == 1:
        personality = "The author has a high emotional level, is difficult to adapt, is always serious about everything, is firm, and has a strong desire"
    elif _class == 2:
        personality = "The author has the ability to control his emotions well, is comfortable, and does not like to harbor anger"
    elif _class == 3:
        personality = "The author has a calm and relaxed personality, is more sensitive, understanding, and has difficulty making decisions because he is easily influenced"

    return personality

def result_margin(_class):
    personality = ""
    if _class == 1:
        personality = "Author holds the characteristics of informality, directness in approach, along with a lack of respect and indifference are evident."
    elif _class == 2:
        personality = "The display of modesty and formality in the writing reflects the author's respectful demeanor towards the recipient, indicative of their personality traits."
    return personality

def result_letterSize(_class):
    personality = ""
    if _class == 1:
        personality = "Author tend to be introspective, less inclined towards seeking attention, and prefer close communication; often possessing strong academic focus, concentration, and organizational skills, with some displaying surprising independence and a drive for power."
    elif _class == 2:
        personality = "The author has the ability to demonstrate adaptability, practicality, and realism, fitting well into conventional circumstances with balanced minds."
    elif _class == 3:
        personality = "Author reflects a craving for attention and recognition, often accompanied by boldness and enthusiasm, yet prone to boastfulness, restlessness, and a lack of focus and discipline, with a dislike for solitude."
    return personality




#A = pd.read_csv('dAboveet_csv/dAboveet.csv')
#train_zone(A)

#x = np.array([[37,42,52]])
#print(predict_zone(x))
