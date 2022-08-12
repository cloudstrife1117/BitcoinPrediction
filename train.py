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
    df = pd.read_csv("./data/BTCBinanceHistoricalData.csv")
    df = df[::-1].reset_index(drop=True)
    df['Price'] = df['Price'].map(commanum2float)
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

    split_index = int(time_series_generator_len*0.8)
    split_index2 = int(time_series_generator_len * 0.9)
    X_train = np.array(time_series_data_x[:split_index])
    y_train = np.array(time_series_data_y[:split_index])
    X_val = np.array(time_series_data_x[split_index:split_index2])
    y_val = np.array(time_series_data_y[split_index:split_index2])

    # Define variables and hyperparameters
    patch_num = 7  # Number of total patches
    proj_dim = 32  # The dimension size for each patches to project to  # 32
    batch_size = 1  # Size of training batch

    # Directory path to save model weights
    model_name = "ViT_" + str(patch_num) + "_" + str(proj_dim)
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    save_path = ROOT_PATH + "/Models/" + model_name

    ViT = VisionTransformer(input_shape=(7, 5),
                            batch_size=batch_size,
                            patch_num=patch_num,
                            proj_dim=proj_dim,
                            num_heads=8,  # 8
                            stack_num=3,  # 6
                            dropout=0.1
                            )

    ViT.summary()

    ViT.train(X_train=X_train,
              X_val=X_val,
              y_train=y_train,
              y_val=y_val,
              optimizer='adamW',
              lr=0.001,  # 0.001
              loss=MSE,
              metrics=[RMSE],
              epochs=20,
              lr_decay=None,
              decay_rate=0.0000199,  # 0.0000199
              weight_decay=0.00001,  # 0.00001
              save_model=True,
              save_path=save_path,
              monitor='val_loss',
              mode='min'
              )


if __name__ == '__main__':
    main()