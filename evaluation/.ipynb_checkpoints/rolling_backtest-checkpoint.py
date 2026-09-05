import os
import json
import numpy as np
import pandas as pd
import sys


sys.path.append(".")


from kronos.inference import KronosInference



# ==================================================
# PROJECT PATH
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ==================================================
# BEST CONFIGURATION
# Change only here
# ==================================================

INPUT_SIZE = 250

PRED_SIZE = 5

NUMBER_OF_WINDOWS = 100

STEP = 500



# ==================================================
# OUTPUT
# ==================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "backtest"
)


WINDOW_DIR = os.path.join(
    OUTPUT_DIR,
    "windows"
)


os.makedirs(
    WINDOW_DIR,
    exist_ok=True
)



# ==================================================
# DATA LOADER
# ==================================================

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



# ==================================================
# METRICS
# ==================================================

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



    # Direction accuracy

    if len(actual_close) > 1:


        actual_direction = (
            np.diff(actual_close) > 0
        )


        predicted_direction = (
            np.diff(predicted_close) > 0
        )


        direction_accuracy = (
            np.mean(
                actual_direction ==
                predicted_direction
            )
            *
            100
        )

    else:

        direction_accuracy = 0



    return {

        "MAE":
            float(mae),

        "RMSE":
            float(rmse),

        "MAPE":
            float(mape),

        "Directional_Accuracy":
            float(direction_accuracy)

    }




# ==================================================
# CREATE WINDOW
# ==================================================

def create_window(
    df,
    end_index
):


    input_data = df.iloc[
        end_index-INPUT_SIZE:end_index
    ].copy()



    actual_future = df.iloc[
        end_index:end_index+PRED_SIZE
    ].copy()



    return (
        input_data,
        actual_future
    )




# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":


    print(
        "Loading dataset..."
    )


    df = load_data(

        os.path.join(
            PROJECT_ROOT,
            "data",
            "XAUUSD_clean.csv"
        )

    )


    print(
        "Dataset:",
        df.shape
    )



    print(
        "\nLoading Kronos model..."
    )


    kronos = KronosInference(

        model_path=
        "NeoQuasar/Kronos-small",


        tokenizer_path=
        "NeoQuasar/Kronos-Tokenizer-base"

    )


    kronos.load_model()



    results = []



    start = (
        len(df)
        -
        PRED_SIZE
        -
        NUMBER_OF_WINDOWS * STEP
        -
        INPUT_SIZE
    )



    print(
        "\nStarting Backtest..."
    )



    for i in range(NUMBER_OF_WINDOWS):


        print(
            f"\nWindow {i+1}/{NUMBER_OF_WINDOWS}"
        )



        end_index = (

            start

            +

            i * STEP

            +

            INPUT_SIZE

        )



        x, actual = create_window(
            df,
            end_index
        )



        if len(actual) < PRED_SIZE:


            print(
                "Skipped incomplete window"
            )

            continue



        prediction = kronos.predict(

            x,

            pred_len=PRED_SIZE

        )



        metrics = calculate_metrics(

            actual,

            prediction

        )



        metrics["window"] = i + 1


        metrics["date"] = (

            actual["timestamp"]
            .iloc[0]
            .strftime(
                "%Y-%m-%d %H:%M"
            )

        )



        print(
            metrics
        )



        results.append(
            metrics
        )



        actual.to_csv(

            os.path.join(

                WINDOW_DIR,

                f"window_{i+1}_actual.csv"

            ),

            index=False

        )



        prediction.to_csv(

            os.path.join(

                WINDOW_DIR,

                f"window_{i+1}_prediction.csv"

            ),

            index=False

        )



        pd.DataFrame(
            results
        ).to_csv(

            os.path.join(

                OUTPUT_DIR,

                "results_partial.csv"

            ),

            index=False

        )




    # ==================================================
    # FINAL SAVE
    # ==================================================


    results_df = pd.DataFrame(
        results
    )


    results_df.to_csv(

        os.path.join(

            OUTPUT_DIR,

            "results.csv"

        ),

        index=False

    )



    summary = {


        "configuration":

        {

            "input_size":
                INPUT_SIZE,


            "prediction_size":
                PRED_SIZE,


            "windows":
                NUMBER_OF_WINDOWS,


            "step":
                STEP

        },


        "Number_of_windows":

            len(results_df),


        "Average_MAE":

            float(
                results_df["MAE"].mean()
            ),


        "Average_RMSE":

            float(
                results_df["RMSE"].mean()
            ),


        "Average_MAPE":

            float(
                results_df["MAPE"].mean()
            ),


        "Average_Directional_Accuracy":

            float(
                results_df["Directional_Accuracy"].mean()
            )


    }



    with open(

        os.path.join(

            OUTPUT_DIR,

            "summary.json"

        ),

        "w"

    ) as f:


        json.dump(

            summary,

            f,

            indent=4

        )



    print(
        "\n===================="
    )

    print(
        "FINAL RESULT"
    )

    print(
        "===================="
    )


    print(
        json.dumps(
            summary,
            indent=4
        )
    )