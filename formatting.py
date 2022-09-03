import numpy as np

from keras.utils import to_categorical
import pandas as pd


def csvToArray(filePth):
    csvDataFrame = pd.read_csv(filePth,header= None)
    csvNumpyArray=csvDataFrame.to_numpy()
    y_train_unflatten,x_train= np.split(csvNumpyArray,[1],axis=1)
    x_train = x_train.astype(np.float64)
    y_train=y_train_unflatten.flatten()
    # make sure array is of type float64
    x_train = x_train.astype(np.float64)
    return x_train,y_train

def intToOnehot(labelArray):
    oneHotList=[]
    for i in range(len(labelArray)):
        oneHotList.append(to_categorical(labelArray[i]-1, num_classes=10))
    return np.array(oneHotList)

def main():
    x_train,y_train = csvToArray("validate.csv")
    print(x_train.shape)
    print(y_train.shape)
    y_train_onehot = intToOnehot(y_train)
    print(y_train_onehot)
if __name__ == '__main__':
    main()