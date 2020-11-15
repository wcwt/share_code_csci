import numpy as np
import pickle
import function as f
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

train_path = "../source_file/csci_data/SR-ARE-train/"
test_path = "../source_file/csci_data/SR-ARE-test/"


# test label and predice label
def test(test,predict):
    p = []
    for pred in predict:    p.append(np.argmax(pred))
    x = range(len(test))
    fig, axs = plt.subplots(2)
    fig.suptitle('Performance (dots are error points)')
    axs[0].scatter(x, test,s=0.5,label="training label")
    axs[1].scatter(x, p,s=0.5,label="Predict label")
    axs[0].legend()
    axs[1].legend()
    plt.show()


def train(train_feature,train_label,test_feature,test_label):
    model = f.create_model()

    model.fit(train_feature,train_label,epochs=8,shuffle=True,batch_size=10)

    test_loss,test_acc = model.evaluate(train_feature,train_label,verbose=10)
    print(f"loss = {test_loss}, acc = {test_acc}")

    if not(os.path.isdir('modle')):
        model.save_weights('modle/modle.w')
    return model

def main():
    train_feature,train_name,train_toxic_label = f.dataloader(train_path)
    test_feature,test_name,test_toxic_label = f.dataloader(test_path)
    p_fea,p_lab, n_fea,n_lab = f.seperate_sample(train_feature,train_toxic_label)
    pt_fea,pt_lab, nt_fea,nt_lab = f.seperate_sample(test_feature,test_toxic_label)

    feature  = np.append(p_fea,n_fea[:len(p_fea)],axis=0)
    label = np.append(p_lab,n_lab[:len(p_fea)],axis=0)


    model = train(feature,label,test_feature,test_toxic_label)
    predict = model.predict(feature)
    test(label,predict)

if __name__ == "__main__":
    main()
