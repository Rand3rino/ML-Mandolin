import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
print(tf.test.is_gpu_available())
print(tf.test.gpu_device_name())
# import os
# import pandas as pd
# # import librosa
#
# import numpy as np
# # import pydub
#
# from keras.layers import Dense, LSTM, LeakyReLU
# from keras.models import Sequential#, load_model
# from scipy.io.wavfile import read, write
# path_vn = os.getcwd() + '/Audio/VN/'
# rate, music1 = read(path_vn + 'v1.wav')
# rate, music2 = read(path_vn + 'v2.wav')
#
# music1 = pd.DataFrame(music1[0:400000, :])
# music2 = pd.DataFrame(music2[0:400000, :])
# print(rate)
# print(music1)
# print(music2)
#
#
# # function to create data by shifting the music data
# def create_dataset(df, look_back, train=True):
#     dataX1, dataX2, dataY1, dataY2 = [], [], [], []
#     for i in range(len(df)-look_back-1):
#         dataX1.append(df.iloc[i : i + look_back, 0].values)
#         dataX2.append(df.iloc[i : i + look_back, 1].values)
#         if train:
#             dataY1.append(df.iloc[i + look_back, 0])
#             dataY2.append(df.iloc[i + look_back, 1])
#     if train:
#         return np.array(dataX1), np.array(dataX2), np.array(dataY1), np.array(dataY2)
#     else:
#         return np.array(dataX1), np.array(dataX2)
#
#
# # Create training dataset
# X1, X2, y1, y2 = create_dataset(pd.concat([music1.iloc[0:160000, :],music2.iloc[0:160000, :]], axis=0), look_back=10, train=True)
#
# # Create testing dataset
# test1, test2 = create_dataset(pd.concat([music1.iloc[160001 : 400000, :],music2.iloc[160001 : 400000, :]], axis=0), look_back=10, train=False)
#
# X1 = X1.reshape((-1, 1, 10))
# X2 = X2.reshape((-1, 1, 10))
# test1 = test1.reshape((-1, 1, 10))
# test2 = test2.reshape((-1, 1, 10))
#
# # LSTM Model for channel 1 of the music data
# rnn1 = Sequential()
# rnn1.add(LSTM(units=100, activation='linear', input_shape=(None, 10)))
# rnn1.add(LeakyReLU())
# rnn1.add(Dense(units=50, activation='linear'))
# rnn1.add(LeakyReLU())
# rnn1.add(Dense(units=25, activation='linear'))
# rnn1.add(LeakyReLU())
# rnn1.add(Dense(units=12, activation='linear'))
# rnn1.add(LeakyReLU())
# rnn1.add(Dense(units=1, activation='linear'))
# rnn1.add(LeakyReLU())
# rnn1.compile(optimizer='adam', loss='mean_squared_error')
# rnn1.fit(X1, y1, epochs=20, batch_size=100)
#
# # LSTM Model for channel 2 of the music data
# rnn2 = Sequential()
# rnn2.add(LSTM(units=100, activation='linear', input_shape=(None, 10
#                                                            )))
# rnn2.add(LeakyReLU())
# rnn2.add(Dense(units=50, activation='linear'))
# rnn2.add(LeakyReLU())
# rnn2.add(Dense(units=25, activation='linear'))
# rnn2.add(LeakyReLU())
# rnn2.add(Dense(units=12, activation='linear'))
# rnn2.add(LeakyReLU())
# rnn2.add(Dense(units=1, activation='linear'))
# rnn2.add(LeakyReLU())
# rnn2.compile(optimizer='adam', loss='mean_squared_error')
# rnn2.fit(X1, y1, epochs=20, batch_size=100)
#
# # making predictions for channel 1 and channel 2
# pred_rnn1 = rnn1.predict(test1)
# pred_rnn2 = rnn2.predict(test2)
#
# # saving the LSTM predicitons in wav format
# write('pred_rnn.wav', rate, pd.concat([pd.DataFrame(pred_rnn1.astype('int16')), pd.DataFrame(pred_rnn2.astype('int16'))], axis=1).values)
# # saving the original music in wav format
# write('original.wav', rate, pd.concat([music1.iloc[160001: 400000, :], music2.iloc[160001: 400000, :]], axis=0).values)
# # def data_preprocessing():
# #     # Converts .WAV files into data and returns dataframe to main
# #
# # # Get directories to two music folders
# # path_not = os.getcwd() + '/Audio/Not VN'
# # path_vn = os.getcwd() + '/Audio/VN'
# # print("HERE1")
# # # Find the total number of files in each
# # total_not = len(os.listdir(path_not))
# # total_vn = len(os.listdir(path_vn))
# #
# # # Convert .WAV files to dataframes
# # dataframe = pd.DataFrame(columns=['Classifier', 'Audio'])
# # songs = []
# # for file in os.listdir(path_not):
# #     data, sampling_rate = librosa.load(path_not + '/' + file)
# #
# #     songs.append(data)
# #     print("HERE2")
# # for file in os.listdir(path_vn):
# #     data, sampling_rate = librosa.load(path_vn + '/' + file)
# #     songs.append(data)
# #     print("HERE3")
# # classification = [0] * total_not + [1] * total_vn
# #
# # dataframe.Classifier = classification
# # dataframe.Audio = songs
# # print("HERE4")
# #     return dataframe
# #
# #
# # dataset = data_preprocessing()
# #
# # dataset.to_csv(os.getcwd() + '/AudioData.csv', index=False)