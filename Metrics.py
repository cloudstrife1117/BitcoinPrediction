"""
Module of metrics or loss functions
"""
import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf


def MSE(y_true, y_pred):
    """
    Function to calculate the mean squared error between the true and the predicted data

    Parameters
    ----------
    y_true : array, tensor
      The tensor of the true data
    y_pred : array, tensor
      The tensor of the predicted data

    Returns
    -------
    mse : float32 tensor
      The mean squared error value
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mse = tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)))

    return mse


def RMSE(y_true, y_pred):
    """
    Function to calculate the root mean squared error between the true and the predicted data

    Parameters
    ----------
    y_true : array, tensor
      The tensor of the true data
    y_pred : array, tensor
      The tensor of the predicted data

    Returns
    -------
    rmse : float32 tensor
      The root mean squared error value
    """
    rmse = tf.sqrt(MSE(y_true, y_pred))

    return rmse
