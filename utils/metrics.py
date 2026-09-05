# metrics functions
import numpy as np


def calculate_mae(actual, predicted):

    return np.mean(
        np.abs(actual - predicted)
    )



def calculate_rmse(actual, predicted):

    return np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )



def calculate_mape(actual, predicted):

    return np.mean(
        np.abs(
            (actual - predicted) / actual
        )
    ) * 100



def directional_accuracy(
        actual,
        predicted
):

    actual_direction = np.diff(actual) > 0

    predicted_direction = np.diff(predicted) > 0


    return np.mean(
        actual_direction == predicted_direction
    ) * 100