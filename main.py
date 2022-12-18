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


from pydantic import BaseModel
import base64
from typing import Union
from fastapi import FastAPI


class Wine(BaseModel):
    fixedAcidity: float
    volatileAcidity: float
    citricAcid: float
    residualSugar: float
    chlorides : float
    freeSulfurDioxide : float
    totalSulfurDioxide : float
    density : float
    pH : float
    sulphates : float 
    alcohol : float

    
app = FastAPI()








# 3 
@app.get("/api/model")   
def getModel():
    """Call the get_data_frame() function through the API
        Returns:
            The dataframe as a json format
    """
    df = get_data_frame()
    df_json = df.to_dict(orient='list') 
    return df_json

# 4 
@app.get("/api/model/description")   
def getDescri():
    """Call the get_description() function through the API
    """
    return get_description(load_model,y_test,X_test)




def get_data_frame():
    """Open the csv file and drop the column Id 
        Returns:
            The dataframe as a pandas.core.frame.DataFrame format
    """
    data_frame = pd.read_csv("Wines.csv")
    if 'Id' in data_frame : 
        data_frame = data_frame.drop(columns=['Id'])
    data_frame.columns = ['fixedAcidity', 'volatileAcidity', 'citricAcid', 'residualSugar', 'chlorides', 'freeSulfurDioxide', 'totalSulfurDioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    return data_frame

def put_model(data_frame , add_wine: dict) -> pd.DataFrame:
    """add the wine add_wine to the dataframe
        Args:
        data_frame: the pandas.core.frame.DataFrame dataframe of the wines
        add_wine : type dict, value received from putModel(data: Wine)
        Returns:
            The dataframe as a pandas.core.frame.DataFrame format
    """

    new_data_frame = data_frame.append(add_wine, ignore_index=True)
    new_data_frame.to_csv('Wines.csv', index=False)
    print(type(new_data_frame))
    return new_data_frame


def get_train_test_data() -> dict:
    """create the train and test datas
        Returns:
            The dict of the train and test numpy arrays
    """
    data_frame = get_data_frame()
    
    # from : https://www.kaggle.com/code/omaryassersalaheldin/red-wine-quality-classifier-using-a-neural-network/notebook
    # Now seperate the dataset as response variable and feature variabes
    X = data_frame.drop('quality', axis = 1)
    y = data_frame['quality']

    #Train and Test splitting of data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    #Applying Standard scaling to get optimized result
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    return X_train, X_test, y_train, y_test

def post_retrain(X_train, y_train):
    """create the train and test datas
        Args:
            X_train numpy array of the training values 
            y_train numpy array of the training values 
    """
    train_test_data = get_train_test_data()
    
    X_train = train_test_data[0]
    y_train = train_test_data[2]
    
    rfc = RandomForestClassifier(n_estimators=300)
    rfc.fit(X_train, y_train)
    get_model(rfc)


# 6 
@app.post("/api/model/retrain")
def postRetrain():
    """Call the post_retrain function through the api 
    """
    post_retrain(X_train, y_train)





# 1 
@app.post("/api/predict") 
def post_predict(data : Wine) -> int:
    """create the train and test datas
        Args:
           data : the wine object that you desire to test
        Return : 
            the quality as int of the wine tested  
    """
    rfc= pickle.load(open('model.pkl', 'rb'))
    new_wine = {
        "fixedAcidity" : data.fixedAcidity,
        "volatileAcidity" : data.volatileAcidity,
        "citricAcid" : data.citricAcid,
        "residualSugar" : data.residualSugar,
        "chlorides" : data.chlorides,
        "freeSulfurDioxide" : data.freeSulfurDioxide,
        "totalSulfurDioxide" : data.totalSulfurDioxide,
        "density" : data.density,
        "pH" : data.pH,
        "sulphates" : data.sulphates,
        "alcohol" : data.alcohol
    }
    to_predict_new_wine = json_normalize(new_wine)

    pred_rfc = rfc.predict(to_predict_new_wine)
    return int(pred_rfc[0])


def get_model(model):
    """Call the already trained model 
        Args:
           model : trained model
    """
    pickle.dump(model, open('model.pkl', 'wb'))



def get_description(model,y_test,X_test) -> dict:
    """return some description about the parameters of the model, the accuracy, the classification report 
        Args:
           model : trained model
           y_test : numpy array of the test datas
           X_test : numpy array of the test datas
    """
    pred_model = model.predict(X_test)
    
    param = model.get_params()
    accuracy = accuracy_score(y_test,pred_model)
    class_report = classification_report(y_test, pred_model)
    
    return param, accuracy, class_report

@app.put("/api/model")
def putModel(data: Wine):
    """Call the put_model function through the api 
    Args:
        data: the json body of the request casted as a Wine object
    """
    # create a json object of the class
    add_wine = {
        "fixedAcidity" : data.fixedAcidity,
        "volatileAcidity" : data.volatileAcidity,
        "citricAcid" : data.citricAcid,
        "residualSugar" : data.residualSugar,
        "chlorides" : data.chlorides,
        "freeSulfurDioxide" : data.freeSulfurDioxide,
        "totalSulfurDioxide" : data.totalSulfurDioxide,
        "density" : data.density,
        "pH" : data.pH,
        "sulphates" : data.sulphates,
        "alcohol" : data.alcohol,
        "quality" : 5
    }
    data_frame = get_data_frame()
    put_model(data_frame, add_wine)








add_wine = {
    "fixedAcidity" : 7.4,
    "volatileAcidity" : 0.7,
    "citricAcid" : 0,
    "residualSugar" : 1.9,
    "chlorides" : 0.076,
    "freeSulfurDioxide" : 11,
    "totalSulfurDioxide" : 34,
    "density" : 0.9978,
    "pH" : 3.51,
    "sulphates" : 0.56,
    "alcohol" : 9.4,
    "quality" : 5
}



data_frame = get_data_frame()

train_test_data = get_train_test_data()

X_train = train_test_data[0]
X_test = train_test_data[1]
y_train = train_test_data[2]
y_test = train_test_data[3]


@app.get("/api/predict")
def get_predict():
    """Return the values of the perfect wine according to the model 
    Return:
        best_wine: the json body of the 'perfect' wine 
    """
    data_frame = get_data_frame()
    sorted_data_frame = data_frame.sort_values(by="quality", ascending=False)
    
    pred_fixedAcidiy = 0
    pred_volatileAcidity = 0
    pred_citricAcid = 0
    pred_residualSugar = 0
    pred_chlorides = 0
    pred_freeSulfurDioxide = 0
    pred_totalSulfurDioxide = 0
    pred_density = 0
    pred_pH = 0
    pred_sulphates = 0
    pred_alcohol = 0
 
    for i in range(10):
        pred_fixedAcidiy += sorted_data_frame.iloc[i][0]
        pred_volatileAcidity += sorted_data_frame.iloc[i][1]
        pred_citricAcid += sorted_data_frame.iloc[i][2]
        pred_residualSugar += sorted_data_frame.iloc[i][3]
        pred_chlorides += sorted_data_frame.iloc[i][4]
        pred_freeSulfurDioxide += sorted_data_frame.iloc[i][5]
        pred_totalSulfurDioxide += sorted_data_frame.iloc[i][6]
        pred_density += sorted_data_frame.iloc[i][7]
        pred_pH += sorted_data_frame.iloc[i][8]
        pred_sulphates += sorted_data_frame.iloc[i][9]
        pred_alcohol += sorted_data_frame.iloc[i][10]
    
        best_wine = {
            'fixedAcidity' : pred_fixedAcidiy/10.0,
            'volatileAcidity' : pred_volatileAcidity/10.0,
            'citricAcid' : pred_citricAcid/10.0,
            'residualSugar' : pred_residualSugar/10.0,
            'chlorides' : pred_chlorides/10.0,
            'freeSulfurDioxide' : pred_freeSulfurDioxide/10.0,
            'totalSulfurDioxide' : pred_totalSulfurDioxide/10.0,
            'density' : pred_density/10.0,
            'pH' : pred_pH/10.0,
            'sulphates' : pred_sulphates/10.0,
            'alcohol' : pred_alcohol/10.0}

        return best_wine

import doctest
doctest.testmod()


