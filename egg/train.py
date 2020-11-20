import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# /home/kenny/tf/SR-ARE-train
file = open("../../source_file/csci_data/SR-ARE-train/names_onehots.pickle","rb")
data = pickle.load(file)

# we want to sep the onehot and names
onehot = data["onehots"]

file = open("../../source_file/csci_data/SR-ARE-train/names_labels.txt","r")
content = np.loadtxt(file,delimiter=",",dtype="str")

toxic = []
for i in range(len(content)):
    toxic.append( int(content[i][1]) )
print(toxic)
