import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# /home/kenny/tf/SR-ARE-test
# ../SR-ARE-score/
file = open("../../source_file/csci_data/SR-ARE-test/names_onehots.pickle","rb")
data = pickle.load(file)

# we want to sep the onehot and names
onehot = data["onehots"]

model = tf.keras.models.load_model('my_model')
predict = model.predict(onehot)


file = open("labels.txt","w+")
for i in range( len(predict) ):
    file.write(f"{np.argmax(predict[i])}\n")
file.close()
