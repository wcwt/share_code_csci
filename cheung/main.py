import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

train_path = "../../source_file/csci_data/SR-ARE-train/"
test_path = "../../source_file/csci_data/SR-ARE-test/"

with open(train_path + "names_onehots.pickle","rb") as f:
    train_data = pickle.load(f)

train_structure,train_name = train_data['onehots'],train_data['names']

label_data = np.loadtxt(train_path+"names_labels.txt",dtype=str,delimiter=',')
train_label = np.array(label_data[:,1],dtype=int)

model = keras.Sequential(
    [
        layers.Flatten(input_shape = (70,325),name="input"),
        layers.Dense(128, activation="relu", name="layer1"),
        layers.Dense(32, activation="relu",name="layer2"),
        layers.Dense(2, activation="softmax", name="output"),
    ]
)

model.compile(optimizer = 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.fit(train_structure,train_label,epochs=2,shuffle=True,batch_size=10)

loss,acc = model.evaluate(train_structure,train_label)
print(f"loss = {loss}, accuracy = {acc}")


model.save('saved_model/my_model')

np.argmax(predict[3])
