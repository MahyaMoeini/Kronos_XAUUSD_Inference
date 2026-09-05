import os
import itertools
import time
import pandas as pd
import numpy as np
import sys


sys.path.append(".")


from kronos.inference import KronosInference



PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "experiments"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "context_horizon_grid.csv"
)



# ======================================================
# Experiment Space
# ======================================================


INPUT_SIZES = [

    100,
    150,
    200,
    250,
    300,

    350,
    400,
    450,
    500,
    600,

    700,
    800,
    900,
    1000,

    1200,
    1400,
    1600,
    1800,
    2000,

    2500,
    3000

]



PRED_SIZES = [

    5,
    10,
    15,
    20,
    30,

    40,
    50,
    60,

    75,
    90,
    105,
    120,

    135,
    150,
    180,

    210,
    240,
    270,
    300

]



STEP = 500

WINDOWS = 100




# ======================================================
# Load Dataset
# ======================================================


def load_data():

    path = os.path.join(
        PROJECT_ROOT,
        "data",
        "XAUUSD_clean.csv"
    )


    df = pd.read_csv(
        path
    )


    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )


    df = df.sort_values(
        "timestamp"
    )


    df = df.reset_index(
        drop=True
    )


    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]


    df = df[required]


    return df




# ======================================================
# Metrics
# ======================================================


def calculate_metrics(
    actual,
    prediction
):


    actual_close = (
        actual["close"]
        .values
    )


    predicted_close = (
        prediction["close"]
        .values
    )



    if len(actual_close) != len(predicted_close):

        raise ValueError(
            f"Length mismatch actual={len(actual_close)} predicted={len(predicted_close)}"
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



    actual_direction = (
        np.diff(actual_close) > 0
    )


    predicted_direction = (
        np.diff(predicted_close) > 0
    )



    directional_accuracy = (

        actual_direction ==
        predicted_direction

    ).mean() * 100



    return {

        "MAE": float(mae),

        "RMSE": float(rmse),

        "MAPE": float(mape),

        "Directional_Accuracy": float(
            directional_accuracy
        )

    }




# ======================================================
# Single Experiment
# ======================================================


def run_experiment(
    kronos,
    df,
    input_size,
    pred_size
):


    results = []


    # Change predictor only once
    kronos.create_predictor(
        input_size
    )



    start = (

        len(df)

        -

        pred_size

        -

        WINDOWS * STEP

        -

        input_size

    )



    if start < 0:

        raise ValueError(
            "Not enough data for this configuration"
        )



    for i in range(WINDOWS):


        end_index = (

            start

            +

            i * STEP

            +

            input_size

        )



        if (
            end_index + pred_size
            >
            len(df)
        ):

            continue



        x = df.iloc[
            end_index-input_size:
            end_index
        ].copy()



        y = df.iloc[
            end_index:
            end_index+pred_size
        ].copy()



        prediction = kronos.predict(

            x,

            pred_len=pred_size

        )



        if len(prediction) != pred_size:

            print(
                "Skipping invalid prediction length"
            )

            continue



        m = calculate_metrics(

            y,

            prediction

        )


        results.append(m)



    if len(results) == 0:

        raise ValueError(
            "No valid windows"
        )



    return {

        "input_size":
        input_size,


        "pred_size":
        pred_size,


        "step":
        STEP,


        "windows":
        WINDOWS,


        "MAE":
        np.mean(
            [
                r["MAE"]
                for r in results
            ]
        ),


        "RMSE":
        np.mean(
            [
                r["RMSE"]
                for r in results
            ]
        ),


        "MAPE":
        np.mean(
            [
                r["MAPE"]
                for r in results
            ]
        ),


        "Directional_Accuracy":
        np.mean(
            [
                r["Directional_Accuracy"]
                for r in results
            ]
        )

    }




# ======================================================
# Main
# ======================================================


if __name__ == "__main__":


    print(
        "Loading data..."
    )


    df = load_data()


    print(
        "Dataset:",
        df.shape
    )



    print(
        "Loading Kronos once..."
    )


    kronos = KronosInference(

        model_path=
        "NeoQuasar/Kronos-small",

        tokenizer_path=
        "NeoQuasar/Kronos-Tokenizer-base"

    )


    # ONLY ONE TIME

    kronos.load_model()



    # Create valid experiments

    experiments = [

        (inp, pred)

        for inp, pred in itertools.product(

            INPUT_SIZES,

            PRED_SIZES

        )

        if inp >= pred * 2

    ]



    print(
        "Valid experiments:",
        len(experiments)
    )



    # Resume support

    if os.path.exists(
        RESULT_FILE
    ):

        old = pd.read_csv(
            RESULT_FILE
        )


        completed = len(old)


        results = old.to_dict(
            "records"
        )


        print(
            "Resuming from:",
            completed
        )


    else:

        completed = 0

        results = []




    for idx, (inp, pred) in enumerate(

        experiments[completed:],

        start=completed + 1

    ):


        print(
            "\n=============================="
        )


        print(
            f"Experiment {idx}/{len(experiments)}"
        )


        print(
            "Context:",
            inp,

            "Forecast:",
            pred
        )



        start_time = time.time()



        try:


            result = run_experiment(

                kronos,

                df,

                inp,

                pred

            )



            result["time_sec"] = (

                time.time()
                -
                start_time

            )



            results.append(
                result
            )



            pd.DataFrame(
                results
            ).to_csv(

                RESULT_FILE,

                index=False

            )



            print(
                result
            )



        except Exception as e:


            print(
                "FAILED:",
                e
            )


            results.append({

                "input_size": inp,

                "pred_size": pred,

                "error": str(e)

            })



            pd.DataFrame(
                results
            ).to_csv(

                RESULT_FILE,

                index=False

            )



    print(
        "\n================================"
    )


    print(
        "FINISHED"
    )


    print(
        RESULT_FILE
    )