import sys
import os
import json
import numpy as np
import pandas as pd


sys.path.append(".")


from kronos.inference import KronosInference



def load_data(path):

    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    )

    df = df.reset_index(
        drop=True
    )

    return df



def create_test_sample(
    df,
    input_size=600,
    pred_size=120
):

    start = len(df) - input_size - pred_size


    input_data = df.iloc[
        start:
        start + input_size
    ].copy()


    actual_future = df.iloc[
        start + input_size:
        start + input_size + pred_size
    ].copy()


    return (
        input_data,
        actual_future
    )



def calculate_metrics(
    actual,
    predicted
):

    actual_close = (
        actual["close"]
        .values
    )


    predicted_close = (
        predicted["close"]
        .values
    )


    mae = np.mean(
        np.abs(
            actual_close -
            predicted_close
        )
    )


    rmse = np.sqrt(
        np.mean(
            (
                actual_close -
                predicted_close
            ) ** 2
        )
    )


    mape = np.mean(
        np.abs(
            (
                actual_close -
                predicted_close
            )
            /
            actual_close
        )
    ) * 100


    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }



def save_results(
    prediction,
    actual,
    metrics
):

    os.makedirs(
        "outputs",
        exist_ok=True
    )


    prediction.to_csv(
        "outputs/prediction.csv",
        index=True
    )


    actual.to_csv(
        "outputs/actual.csv",
        index=True
    )


    with open(
        "outputs/metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )



if __name__ == "__main__":


    print(
        "Loading dataset..."
    )


    df = load_data(
        "data/XAUUSD_clean.csv"
    )


    input_data, actual_future = (
        create_test_sample(df)
    )


    print(
        "Input shape:",
        input_data.shape
    )


    print(
        "Actual future shape:",
        actual_future.shape
    )


    print(
        "Loading Kronos..."
    )


    kronos = KronosInference(
        model_path=
        "NeoQuasar/Kronos-small",

        tokenizer_path=
        "NeoQuasar/Kronos-Tokenizer-base"
    )


    kronos.load_model()


    print(
        "Running prediction..."
    )


    prediction = kronos.predict(
        input_data,
        pred_len=120
    )


    print(
        "Prediction completed"
    )


    metrics = calculate_metrics(
        actual_future,
        prediction
    )


    print("\nMetrics:")
    print(metrics)


    save_results(
        prediction,
        actual_future,
        metrics
    )


    print(
        "\nResults saved in outputs/"
    )


def directional_accuracy(
    actual,
    predicted
):

    actual_direction = (
        actual["close"]
        .diff()
        .values[1:]
    )


    predicted_direction = (
        predicted["close"]
        .diff()
        .values[1:]
    )


    correct = (
        (actual_direction > 0)
        ==
        (predicted_direction > 0)
    )


    accuracy = (
        np.mean(correct)
        * 100
    )


    return float(accuracy)