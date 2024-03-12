import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
class DataProcessor:
def __init__(self, file_path):
self.file_path = file_path
self.df = pd.read_csv(file_path)
self.numeric_columns = None
self.categorical_columns = None
self.df_scaled = None
def summarize_data(self):
numeric_summary = self.df[self.numeric_columns].describe()
print("Numeric Columns Statistical Summary:")
print(numeric_summary)
for column in self.categorical_columns:
print(f"Frequency of {column} Column:")
print(self.df[column].value_counts())
def encode_categorical_data(self):
label_encoder = LabelEncoder()
for column in self.categorical_columns:
self.df[column] = label_encoder.fit_transform(self.df[column])
def scale_numeric_data(self):
scaler = StandardScaler()
self.df_scaled = self.df.copy()
self.df_scaled[self.numeric_columns] =
scaler.fit_transform(self.df[self.numeric_columns])
def set_column_types(self, numeric_cols, categorical_cols):
self.numeric_columns = numeric_cols
self.categorical_columns = categorical_cols
class ModelTrainer:
def __init__(self, data):
self.data = data
self.models = {}
def train_kmeans(self, params, pca_data):
kmeans = KMeans(**params)
kmeans.fit(pca_data)
self.models['kmeans'] = kmeans
def train_gmm(self, params, data):
gmm = GaussianMixture(**params)
gmm.fit(data)
self.models['gmm'] = gmm
def train_xgb(self, params, X_train, y_train):
xgb = XGBRegressor(**params)
xgb.fit(X_train, y_train)
self.models['xgb'] = xgb
def train_mlp(self, params, X_train, y_train):
mlp = MLPRegressor(**params)
mlp.fit(X_train, y_train)
self.models['mlp'] = mlp
def evaluate_model(self, model_name, X_test, y_test):
if model_name in self.models:
model = self.models[model_name]
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"{model_name} - Mean Squared Error: {mse}")
else:
print(f"Model {model_name} not found.")