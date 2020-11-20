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

toxic = [] # list
for i in range(len(content)):
    toxic.append( int(content[i][1]) )

# making model
model = keras.Sequential(
    [
        layers.Flatten(input_shape = (70,325) ),
        layers.Dense(128, activation="relu" ),
        layers.Dense(32, activation="relu" ),
        layers.Dense(2, activation="softmax" ),
    ]
)
#    for loss function ( need to del)

optimizers = tf.keras.optimizers.Adam(learning_rate=0.01) # adjust lr
model.compile(optimizer = optimizers,
            loss = 'sparse_categorical_crossentropy',
            metrics=['accuracy'])
