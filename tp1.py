import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from pandas import json_normalize

def post_retrain(data_frame):
    #Now seperate the dataset as response variable and feature variabes
    X = data_frame.drop('quality', axis = 1)
    y = data_frame['quality']

    #Train and Test splitting of data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    #Applying Standard scaling to get optimized result
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    

def post_predict(data_frame, new_wine):
    #Now seperate the dataset as response variable and feature variabes
    X = data_frame.drop('quality', axis = 1)
    y = data_frame['quality']

    #Train and Test splitting of data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    #Applying Standard scaling to get optimized result
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    rfc = RandomForestClassifier(n_estimators=300)
    rfc.fit(X_train, y_train)
    pred_rfc = rfc.predict(new_wine)
    
    # print(get_description(rfc,y_test,X_test)[2])
    print(pred_rfc[0])

def get_model(model):
    pickle.dump(model, open('model.pkl', 'wb'))

def get_description(model,y_test,X_test):
    pred_model = model.predict(X_test)
    
    param = model.get_params()
    accuracy = accuracy_score(y_test,pred_model)
    class_report = classification_report(y_test, pred_model)
    
    return param, accuracy, class_report

def put_model(data_frame, add_wine):
    new_data_frame = data_frame.append(add_wine, ignore_index=True)
    new_data_frame.to_csv('WineTest.csv', index=False)
    # return new_data_frame

data_frame = pd.read_csv("Wines.csv")

data_frame = data_frame.drop(columns=['Id'])
data_frame.columns = ['fixedAcidity', 'volatileAcidity', 'citricAcid', 'residualSugar', 'chlorides', 'freeSulfurDioxide', 'totalSulfurDioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

new_wine = {'fixedAcidity' : 7.4,
    'volatileAcidity' : 0.7,
    'citricAcid' : 0,
    'residualSugar' : 1.9,
    'chlorides' : 0.076,
    'freeSulfurDioxide' : 11,
    'totalSulfurDioxide' : 34,
    'density' : 0.9978,
    'pH' : 3.51,
    'sulphates' : 0.56,
    'alcohol' : 9.4}

add_wine = {'fixedAcidity' : 7.4,
    'volatileAcidity' : 0.7,
    'citricAcid' : 0,
    'residualSugar' : 1.9,
    'chlorides' : 0.076,
    'freeSulfurDioxide' : 11,
    'totalSulfurDioxide' : 34,
    'density' : 0.9978,
    'pH' : 3.51,
    'sulphates' : 0.56,
    'alcohol' : 9.4,
    'quality' : 5
    }

to_predict_new_wine = json_normalize(new_wine)

#Now seperate the dataset as response variable and feature variabes
# X = wine.drop('quality', axis = 1)
# y = wine['quality']

# #Train and Test splitting of data 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# #Applying Standard scaling to get optimized result
# sc = StandardScaler()

# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)

# rfc = RandomForestClassifier(n_estimators=300)
# rfc.fit(X_train, y_train)
# pred_rfc = rfc.predict(X_test)

# accuracy_score(y_test,pred_rfc)

# print(classification_report(y_test, pred_rfc))

post_predict(data_frame, to_predict_new_wine)

# put_model(data_frame, add_wine)

# get_model(rfc)

load_model = pickle.load(open('model.pkl', 'rb'))
