import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dataloader(path,label_exist=True):
    pickle_in = path + "names_onehots.pickle"
    # load data from file
    with open (pickle_in,"rb") as f:
        obj = pickle.load(f)
    feature,name = obj['onehots'],obj['names']
    if label_exist:
        label_in = path + "names_labels.txt"
        label_data = np.loadtxt(label_in,dtype=str,delimiter=',')
        toxic_label = np.array(label_data[:,1],dtype=int) # get {0,1} form name
        return feature,name,toxic_label
    else:
        return feature,name

def seperate_sample(feature,label):
    pos_fea = []
    neg_fea = []
    for i,ele in enumerate(label):
        if ele == 1:
            pos_fea.append(feature[i])
        else:
            neg_fea.append(feature[i])
    pos_fea = np.array(pos_fea)
    pos_lab = np.zeros(len(pos_fea)) + 1
    neg_fea = np.array(neg_fea)
    neg_lab = np.zeros(len(neg_fea))
    return pos_fea,pos_lab,neg_fea,neg_lab

def create_model():
    # Define Sequential model with 3 layers
    model = keras.Sequential(
        [
            layers.Flatten(input_shape = (70,325),name="input"),
            #layers.Conv2D(2, 3, activation='relu', input_shape=(70,325)),
            layers.Dense(128, activation="relu", name="layer1"),
            layers.Dense(32, activation="relu",name="layer2"),
            layers.Dense(2, activation="softmax", name="output"),
        ]
    )
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses for loss function

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                #loss = 'binary_crossentropy',
                metrics=['accuracy'])
    return model
