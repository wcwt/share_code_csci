import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

file = open("/home/kenny/tf/SR-ARE-test/names_onehots.pickle","rb")
data = pickle.load(file)

onehot = data["onehots"]

file = open("/home/kenny/tf/SR-ARE-test/names_labels.txt","r")
content = np.loadtxt(file,delimiter=",",dtype="str")

toxic = [] # list
for i in range(len(content)):
    toxic.append( int(content[i][1]) )
toxic = np.array(toxic)

neg_index = []
pos_index = []

for i in range(len(toxic)):
    if toxic[i] == 0:
        neg_index.append(i)
    else:
        pos_index.append(i)

train = []
label = []
for ele in pos_index:
    train.append(onehot[ele])



model = keras.Sequential(
    [
        layers.Flatten(input_shape = (70,325) ),
        layers.Dense(256, activation="relu" ),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu" ),
        layers.Dense(64, activation="relu" ),
        layers.Dense(32, activation="relu" ),
        layers.Dense(2, activation="softmax" ),
    ]
)

optimizers = tf.keras.optimizers.Adam(learning_rate=0.0145)
model.compile(optimizer = optimizers,
            loss = 'sparse_categorical_crossentropy',
            metrics=['accuracy'])

# train model
model.fit(onehot,toxic,batch_size=20,epochs=5,shuffle=True) # batch_size ==> 20% data

model.save("model/model.save")
loss,acc = model.evaluate(onehot,toxic)
print(loss,acc)
