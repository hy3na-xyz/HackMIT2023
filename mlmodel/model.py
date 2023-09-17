import numpy as np
import pandas as pd 
import os
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import mne 
import pickle
import ast
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Flatten
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K





def data_preprocess(path):
    df = pd.read_csv(path, skiprows=[0])

    columns_to_keep = ['Timestamp', 'EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 
                   'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4', 
                   'PM.Stress.Scaled', 'PM.Stress.Raw', 'PM.Stress.Min', 'PM.Stress.Max']

    df.drop(columns=df.columns.difference(columns_to_keep), inplace=True)
    df.fillna(0)
    
    x = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']
    eeg = df[x]
    
    y = ['PM.Stress.Scaled', 'PM.Stress.Raw', 'PM.Stress.Min', 'PM.Stress.Max']
    stress = df[y]
    
    stress.dropna(inplace=True)
    
    n = len(stress)
    filtered_eeg = eeg.head(n)
    
    # Separate the feature matrix (X) and target variable (y)
    X = filtered_eeg[['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']]
    y = stress['PM.Stress.Raw']
    
    return X, y 


def predict_target(X, trained_model):
    predicted_target = trained_model.predict(X)
    return predicted_target[0] 
    

df = pd.read_csv('/Users/akhil/Downloads/HackMIT2023/mlmodel/df.csv')
x = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']
eeg = df[x]

y = ['PM.Stress.Scaled', 'PM.Stress.Raw', 'PM.Stress.Min', 'PM.Stress.Max']
stress = df[y]

X = pd.read
data = X
scaler = MinMaxScaler(feature_range=(-1,1))
scaler = MinMaxScaler()
scaler.fit(data)
normalized = scaler.fit_transform(data)

X_train,X_Test,Y_Train, Y_Test = train_test_split(normalized, y)


data_preprocess("path of csv")
predict_target(X)