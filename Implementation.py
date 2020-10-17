# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train_dataset = pd.read_csv('Training.csv')
test_dataset = pd.read_csv('Testing.csv')

train_dataset.isnull().sum()

# Slicing and Dicing the dataset to separate features from predictions
X = train_dataset.iloc[:, 0:132].values
y = train_dataset.iloc[:, -1].values

# Dimensionality Reduction for removing redundancies
minimised_dataset = train_dataset.groupby(train_dataset['prognosis']).max()
minimised_dataset

# Encoding String values to integer constants
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
