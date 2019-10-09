import os
import pandas as pd
import librosa


def data_preprocessing():
    # Converts .WAV files into data and returns dataframe to main

    # Get directories to two music folders
    path_not = os.getcwd() + '/Audio/Not VN'
    path_vn = os.getcwd() + '/Audio/VN'

    # Find the total number of files in each
    total_not = len(os.listdir(path_not))
    total_vn = len(os.listdir(path_vn))

    # Convert .WAV files to dataframes
    dataframe = pd.DataFrame(columns=['Classifier', 'Audio'])
    songs = []
    for file in os.listdir(path_not):
        data, sampling_rate = librosa.load(path_not + '/' + file)
        songs.append(data)

    for file in os.listdir(path_vn):
        data, sampling_rate = librosa.load(path_vn + '/' + file)
        songs.append(data)

    classification = [0] * total_not + [1] * total_vn

    dataframe.Classifier = classification
    dataframe.Audio = songs
    return dataframe


dataset = data_preprocessing()

dataset.to_csv(os.getcwd() + '/AudioData.csv', index=False)