# Original Code from: https://medium.com/intel-student-ambassadors/music-generation-using-lstms-in-keras-9ded32835a8f

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, LeakyReLU, Conv1D, Flatten
from keras.models import Sequential
from scipy.io.wavfile import read, write

path_vn = os.getcwd() + '/Audio/VN/'
rate, music1 = read(path_vn + 'v1.wav')
rate, music2 = read(path_vn + 'v2.wav')
rate, music3 = read(path_vn + 'v3.wav')
rate, music4 = read(path_vn + 'v4.wav')
rate, music5 = read(path_vn + 'v5.wav')
rate, music6 = read(path_vn + 'v6.wav')
rate, music7 = read(path_vn + 'v7.wav')
rate, music8 = read(path_vn + 'v8.wav')
rate, music9 = read(path_vn + 'v9.wav')
rate, music10 = read(path_vn + 'v10.wav')
rate, music11 = read(path_vn + 'v11.wav')
rate, music12 = read(path_vn + 'v12.wav')

dataStart = 0
dataEnd = 40000
music1 = pd.DataFrame(music1[dataStart:dataEnd, :])
music2 = pd.DataFrame(music2[dataStart:dataEnd, :])
music3 = pd.DataFrame(music3[dataStart:dataEnd, :])
music4 = pd.DataFrame(music4[dataStart:dataEnd, :])
music5 = pd.DataFrame(music5[dataStart:dataEnd, :])
music6 = pd.DataFrame(music6[dataStart:dataEnd, :])
music7 = pd.DataFrame(music7[dataStart:dataEnd, :])
music8 = pd.DataFrame(music8[dataStart:dataEnd, :])
music9 = pd.DataFrame(music9[dataStart:dataEnd, :])
music10 = pd.DataFrame(music10[dataStart:dataEnd, :])
music11 = pd.DataFrame(music11[dataStart:dataEnd, :])
music12 = pd.DataFrame(music12[dataStart:dataEnd, :])


# function to create data by shifting the music data
def create_dataset(df, look_back, train=True):
    dataX1, dataX2, dataY1, dataY2 = [], [], [], []
    for i in range(len(df) - look_back - 1):
        dataX1.append(df.iloc[i: i + look_back, 0].values)
        dataX2.append(df.iloc[i: i + look_back, 1].values)
        if train:
            dataY1.append(df.iloc[i + look_back, 0])
            dataY2.append(df.iloc[i + look_back, 1])
    if train:
        return np.array(dataX1), np.array(dataX2), np.array(dataY1), np.array(dataY2)
    else:
        return np.array(dataX1), np.array(dataX2)


trainStart = dataStart
trainEnd = 20000
training = pd.concat([music1.iloc[trainStart:trainEnd, :], music2.iloc[trainStart:trainEnd, :],
                      music3.iloc[trainStart:trainEnd, :], music4.iloc[trainStart:trainEnd, :],
                      music5.iloc[trainStart:trainEnd, :], music6.iloc[trainStart:trainEnd, :],
                      music7.iloc[trainStart:trainEnd, :], music8.iloc[trainStart:trainEnd, :],
                      music9.iloc[trainStart:trainEnd, :], music10.iloc[trainStart:trainEnd, :],
                      music11.iloc[trainStart:trainEnd, :], music12.iloc[trainStart:trainEnd, :]],
                     axis=0)

testStart = trainEnd + 1
testEnd = dataEnd
testing = pd.concat([music1.iloc[testStart:testEnd, :], music2.iloc[testStart:testEnd, :],
                     music3.iloc[testStart:testEnd, :], music4.iloc[testStart:testEnd, :],
                     music5.iloc[testStart:testEnd, :], music6.iloc[testStart:testEnd, :],
                     music7.iloc[testStart:testEnd, :], music8.iloc[testStart:testEnd, :],
                     music9.iloc[testStart:testEnd, :], music10.iloc[testStart:testEnd, :],
                     music11.iloc[testStart:testEnd, :], music12.iloc[testStart:testEnd, :]],
                    axis=0)

shape = 100
iters = 50
batch = 50

# Create training dataset
X1, X2, Y1, Y2 = create_dataset(training, look_back=shape, train=True)

# Create testing dataset
test1, test2 = create_dataset(testing, look_back=shape, train=False)

X1 = X1.reshape((-1, 1, shape))
X2 = X2.reshape((-1, 1, shape))
test1 = test1.reshape((-1, 1, shape))
test2 = test2.reshape((-1, 1, shape))

# LSTM Model for channel 1 of the music data
rnn1 = Sequential()
rnn1.add(LSTM(units=100, activation='linear', input_shape=(None, shape)))
rnn1.add(LeakyReLU())
rnn1.add(Dense(units=50, activation='linear'))
rnn1.add(LeakyReLU())
rnn1.add(Dense(units=25, activation='linear'))
rnn1.add(LeakyReLU())
rnn1.add(Dense(units=12, activation='linear'))
rnn1.add(LeakyReLU())
rnn1.add(Dense(units=1, activation='linear'))
rnn1.add(LeakyReLU())
rnn1.compile(optimizer='adam', loss='mean_squared_error')


# LSTM Model for channel 2 of the music data
rnn2 = rnn1
rnn1.fit(X1, Y1, epochs=iters, batch_size=batch)
rnn2.fit(X2, Y2, epochs=iters, batch_size=batch)

# making predictions for channel 1 and channel 2
pred_rnn1 = rnn1.predict(test1)
pred_rnn2 = rnn2.predict(test2)

# saving the LSTM predicitons in wav format
write('pred_rnn.wav', rate, pd.concat([pd.DataFrame(pred_rnn1.astype('int16')),
                                             pd.DataFrame(pred_rnn2.astype('int16'))],
                                            axis=1).values)
# saving the original music in wav format
write('original.wav', rate, testing.values)
