import numpy as np
import pickle
import function as f
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

file_out = "labels.txt"
test_path = "../source_file/csci_data/SR-ARE-test/"
#test_path = "../SR-ARE-score/"

model = f.create_model()
model.load_weights('modle/modle.w')
model.summary()

test_feature,test_name = f.dataloader(test_path,label_exist=False)

predict = model.predict(test_feature)

with open (file_out,"w+") as f:
    for d in predict:
        f.write(f"{np.argmax(d)}\n")
