import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

file = open("../../source_file/csci_data/SR-ARE-train/names_onehots.pickle","rb")
data = pickle.load(file)
print(data)
