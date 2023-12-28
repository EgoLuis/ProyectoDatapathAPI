from fastapi import APIRouter, HTTPException, status
from model import Scores

#Librerias para modelo de Regresión Lineal
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

TEST_SIZE = 0.2
RANDOM_STATE = 2

router = APIRouter()

preds = []

@router.get('/preds')
def get_task():
    return preds

@router.post('/preds', status_code = status.HTTP_201_CREATED)
def create_pred(scores:Scores):
    #Obtenemos el dataset y lo transformamos en un Dataframe con su respectivo target
    boston = datasets.load_boston()
    boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)
    boston_df["PRICE"] = boston.target
    #Obtenemos las variables con las que se entrenará el modelo
    X_bos = boston_df[['RM', 'LSTAT']]
    Y_bos = boston_df['PRICE']
    #Entrenamos el modelo de preddicción con las variables
    X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X_bos, Y_bos, test_size = TEST_SIZE, random_state = RANDOM_STATE)
    #Construimos el modelo de regresion lineal
    reg_1 = LinearRegression()
    reg_1.fit(X_train_1, Y_train_1)
    #Calculamos error cuadratico medio y la relación entre la variable dependiente e independiente
    y_pred_1 = reg_1.predict(X_test_1)
    scores.id = len(preds) + 1
    scores.rms = (np.sqrt(mean_squared_error(Y_test_1, y_pred_1)))
    scores.r2 = round(reg_1.score(X_test_1, Y_test_1), 2)
    preds.append(scores)
    return {"Modelo agregado correctamente"}

@router.delete('/preds/{pred_id}', status_code = status.HTTP_202_ACCEPTED)
def delete_pred(pred_id:int):
    for pred_item in preds:
        if pred_item.id == pred_id:
            preds.remove(pred_item)
            return{"Borrado correctamente"}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail = 'ID no encontrado')
    



    