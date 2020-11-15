import pickle
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


train_path = "../../source_file/csci_data/SR-ARE-train/"

with open (train_path + "names_onehots.pickle","rb") as f:
    train_data = pickle.load(f)

test_path = "../../source_file/csci_data/SR-ARE-test/"
with open (test_path + "names_onehots.pickle","rb") as f:
    test_data = pickle.load(f)

train_structure,train_name = train_data['onehots'],train_data['names']

label_data = np.loadtxt(train_path + "names_labels.txt",dtype=str,delimiter=',')

train_label = np.array(label_data[:,1],dtype=int)

pos_struct = []
neg_struct = []

for i in range( len(train_label) ):
    if train_label[i] == 1:
        pos_struct.append(train_structure[i])
    if train_label[i] == 0:
        neg_struct.append(train_structure[i])

pos_struct = np.array(pos_struct)
pos_label = np.zeros(len(pos_struct)) + 1
neg_struct = np.array(neg_struct)
neg_label = np.zeros(len(neg_struct)) + 0

# pos structure append 5x
for i in range(5):
    pos_struct = np.append(pos_struct,pos_struct,axis=0)
    pos_label = np.append(pos_label,pos_label,axis=0)

train_structure = np.append(pos_struct,neg_struct,axis=0)
train_label = np.append(pos_label,neg_label,axis=0)

# Number of layers, activation function.
model = keras.Sequential(
    [
            layers.Flatten(input_shape = (70,325)),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(2, activation="softmax"),
    ]
)

model.compile(optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Change epochs (the number of modelling)(increases),
# batch_size = percentage of drawing the sample.
model.fit(train_structure,train_label,epochs=5,shuffle=True,batch_size=20)

loss,acc = model.evaluate(train_structure,train_label)

print(f"loss = {loss}, accuracy = {acc}")

model.save('saved_model/kalong_model')
