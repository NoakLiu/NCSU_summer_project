import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model

df1 = pd.read_csv('FB.csv')
df2 = pd.read_csv('TSLA.csv')
data1 = np.array(df1['Close'])
data2 = np.array(df2['Close'])

k = int(input('train or not:train/1;not/0'))

def do_lstm(dataframe,stockname, forward, cell):
    dataset = dataframe['Close'].values.reshape(-1, 1)
    #normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = 1115+forward

    trainlist = dataset[:train_size]
    testlist = dataset[train_size:]
    testlist1 = dataset

    def create_dataset(dataset, forward):
        dataX, dataY = [], []
        for i in range(len(dataset) - forward):
            a = dataset[i:(i + forward - 1)]
            dataX.append(a)
            dataY.append(dataset[i + forward - 1])
        return numpy.array(dataX), numpy.array(dataY)

    trainX, trainY = create_dataset(trainlist, forward)
    testX, testY = create_dataset(testlist, forward)
    test1X, test1Y = create_dataset(testlist1, forward)
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    test1X = numpy.reshape(test1X, (test1X.shape[0], test1X.shape[1], 1))

    # LSTM+Fully connect
    model = Sequential()
    model.add(LSTM(cell, activation='relu', input_shape=(None, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    if (k == 1):
        model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
        model.save(os.path.join("Model"+stockname + ".h5"))
    else:
        model = load_model("Model%s.h5"%stockname)

    trainPredict = model.predict(trainX)  # to 2019/12/31
    testPredict = model.predict(testX)  # after 2020
    test1Predict = model.predict(test1X)  # whole period

    # inverse normalize
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)
    test1Predict = scaler.inverse_transform(test1Predict)
    test1Y = scaler.inverse_transform(test1Y)

    plt.figure(stockname)

    plt.subplot(221)
    plt.title('train')
    plt.plot(trainY, 'b', label='train')
    plt.plot(trainPredict,c='r',label='prediction')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()

    plt.subplot(222)
    plt.title('test')
    plt.plot(testY, 'b', label='test')
    plt.plot(testPredict, 'r', label='prediction')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()

    plt.subplot(212)
    plt.title(stockname+' whole')
    plt.plot(test1Y,'b', label='truth')
    plt.plot(test1Predict, 'r', label='prediction')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()

    plt.show()

do_lstm(df1,'FB',forward=2,cell=3)
do_lstm(df2,'TSLA',forward=2,cell=5)