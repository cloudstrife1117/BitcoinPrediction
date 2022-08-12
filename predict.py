import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from TransformerModel import VisionTransformer
from Metrics import MSE, RMSE


def commanum2float(x):
    return float(x.replace(',', ''))


def unit2numeric(x):
    multiplier = 1
    unit = x[-1]
    numeric = x[:-1]
    numeric = commanum2float(numeric)
    if unit == 'K':
        multiplier = 1000
    elif unit == 'M':
        multiplier = 1000000
    elif unit == 'B':
        multiplier = 1000000000
    numeric = numeric * multiplier
    return numeric


def main():
    df = pd.read_csv("./data/BTCBinanceHistoricalData_New.csv")
    df = df[::-1].reset_index(drop=True)
    df['Price'] = df['Price'].map(commanum2float)
    max_price = df['Price'].max()
    min_price = df['Price'].min()
    df['Open'] = df['Open'].map(commanum2float)
    df['High'] = df['High'].map(commanum2float)
    df['Low'] = df['Low'].map(commanum2float)
    df['Vol.'] = df['Vol.'].replace({'-': '0.0'})
    df['Vol.'] = df['Vol.'].map(unit2numeric)
    scaler = MinMaxScaler()
    target_columns = ['Price', 'Open', 'High', 'Low', 'Vol.']
    df_scaled = pd.DataFrame(scaler.fit_transform(df[target_columns]), columns=target_columns)
    df_target = df_scaled['Price']
    df_scaled = df_scaled.to_numpy(dtype=np.float32)
    df_target = df_target.to_numpy(dtype=np.float32)

    time_series_generator = TimeseriesGenerator(df_scaled, df_target, length=7, sampling_rate=1, batch_size=1)
    time_series_generator_len = len(time_series_generator)
    time_series_data_x = []
    time_series_data_y = []

    for i in range(time_series_generator_len):
        x, y = time_series_generator[i]
        time_series_data_x.append(x[0])
        time_series_data_y.append(y[0])

    split_index = int(time_series_generator_len * 0.8)
    split_index2 = int(time_series_generator_len * 0.9)
    time_series_data_x = np.array(time_series_data_x)

    # Directory path to save model weights
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    model_path = ROOT_PATH + "\Models\ViT_7_32_Test"

    # Define variables and hyperparameters
    batch_size = 1  # Size of training batch

    ViT = tf.keras.models.load_model(model_path, custom_objects={'MSE':MSE, 'RMSE':RMSE})
    next_day_x = np.array([df_scaled[-7:]])
    next_day_y = ViT.predict(next_day_x, batch_size=batch_size)
    print("Estimate Tomorrow End:", next_day_y[0][0] * (max_price-min_price) + min_price)
    next_day_x = np.array([df_scaled[-8:-1]])
    next_day_y = ViT.predict(next_day_x, batch_size=batch_size)
    print("Estimate Today End:", next_day_y[0][0] * (max_price - min_price) + min_price)
    next_day_x = np.array([df_scaled[-9:-2]])
    next_day_y = ViT.predict(next_day_x, batch_size=batch_size)
    print("Estimate Yesterday End:", next_day_y[0][0] * (max_price - min_price) + min_price)

    y_pred = ViT.predict(time_series_data_x, batch_size=batch_size)
    y_pred = [x for xs in y_pred for x in xs]
    plt.plot(range(1300, time_series_generator_len), time_series_data_y[1300:])
    #plt.plot(range(0, split_index), y_pred[:split_index])
    #plt.plot(range(split_index, split_index2), y_pred[split_index:split_index2])
    #plt.plot(range(split_index2, time_series_generator_len), y_pred[split_index2:])
    plt.plot(range(1300, split_index2), y_pred[1300:split_index2])
    plt.plot(range(split_index2, time_series_generator_len), y_pred[split_index2:])
    plt.show()


if __name__ == '__main__':
    main()
