import tensorflow as tf
import numpy as np
import pickle
import os

# model = tf.keras.models.load_model("csci3230_mymodel")
checkpoint_path = "my_model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (70, 325)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.load_weights(latest)

test_x = pickle.load(open("../SR-ARE-score/names_onehots.pickle", "rb"))
# tf_label = open("./data/SR-ARE-test/names_labels.txt", "r")
# tf_label = open("./data/SR-ARE-score/names_smiles.txt", "r")

# test_labels = []
# # for i in tf_label:
# #     test_labels.append(i.split(",")[1].strip())
# for i in tf_label:
#     test_labels.append(i.split(",")[0].strip())

predictions = model.predict(np.array(test_x['onehots']))

f = open("labels.txt", "w")
j = 0
for i in predictions:
    f.write(str(np.argmax(i)))
    if j == len(predictions)-1:
        pass
    else:
        f.write("\n")
    j += 1

f.close()
# test_loss, test_acc = model.evaluate(np.array(test_x['onehots']), np.array(test_labels), verbose = 1)
# print("Test loss:", test_loss)
# print('Test accuracy:', test_acc)
