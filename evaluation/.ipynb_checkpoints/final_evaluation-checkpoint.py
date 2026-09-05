import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ==================================================
# Select one backtest window for visualization
# ==================================================

WINDOW = 15


INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "backtest",
    "windows"
)


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "final"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



prediction_file = os.path.join(
    INPUT_DIR,
    f"window_{WINDOW}_prediction.csv"
)


actual_file = os.path.join(
    INPUT_DIR,
    f"window_{WINDOW}_actual.csv"
)



if not os.path.exists(prediction_file):
    raise FileNotFoundError(
        prediction_file
    )


if not os.path.exists(actual_file):
    raise FileNotFoundError(
        actual_file
    )



# ==================================================
# LOAD
# ==================================================

prediction = pd.read_csv(
    prediction_file
)


actual = pd.read_csv(
    actual_file
)


actual["timestamp"] = pd.to_datetime(
    actual["timestamp"]
)


if "timestamp" in prediction.columns:

    prediction["timestamp"] = pd.to_datetime(
        prediction["timestamp"]
    )



# ==================================================
# Merge Results
# ==================================================

result = pd.DataFrame()


result["timestamp"] = actual["timestamp"]


for col in [
    "open",
    "high",
    "low",
    "close"
]:

    result[f"actual_{col}"] = actual[col]


    result[f"predicted_{col}"] = (
        prediction[col].values
    )



# Error

result["close_error"] = (
    result["predicted_close"]
    -
    result["actual_close"]
)


result["absolute_error"] = (
    result["close_error"]
    .abs()
)



result.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "prediction_results.csv"
    ),
    index=False
)



# ==================================================
# Metrics
# ==================================================

actual_close = (
    result["actual_close"]
)


pred_close = (
    result["predicted_close"]
)



mae = np.mean(
    np.abs(
        actual_close -
        pred_close
    )
)



rmse = np.sqrt(
    np.mean(
        (
            actual_close -
            pred_close
        ) ** 2
    )
)



mape = np.mean(
    np.abs(
        (
            actual_close -
            pred_close
        )
        /
        actual_close
    )
) * 100



# Direction accuracy

actual_direction = (
    actual_close.diff()
    .dropna()
    > 0
)


pred_direction = (
    pred_close.diff()
    .dropna()
    > 0
)



direction_accuracy = (
    actual_direction.values ==
    pred_direction.values
).mean() * 100



metrics = {


    "window":

        WINDOW,


    "MAE":

        float(mae),


    "RMSE":

        float(rmse),


    "MAPE":

        float(mape),


    "Directional Accuracy":

        float(direction_accuracy)

}



with open(
    os.path.join(
        OUTPUT_DIR,
        "metrics.json"
    ),
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )



print(
    json.dumps(
        metrics,
        indent=4
    )
)



# ==================================================
# Plot 1
# Actual vs Prediction
# ==================================================

plt.figure(
    figsize=(12,5)
)


plt.plot(
    result["timestamp"],
    result["actual_close"],
    label="Actual Close"
)


plt.plot(
    result["timestamp"],
    result["predicted_close"],
    label="Kronos Prediction"
)



plt.title(
    f"Kronos XAUUSD Prediction - Window {WINDOW}"
)


plt.xlabel(
    "Time"
)


plt.ylabel(
    "Price"
)


plt.legend()
plt.grid()
plt.tight_layout()



plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "forecast.png"
    ),
    dpi=300
)



plt.close()



# ==================================================
# Plot 2
# Error
# ==================================================

plt.figure(
    figsize=(12,4)
)


plt.plot(
    result["timestamp"],
    result["close_error"]
)


plt.axhline(
    0
)



plt.title(
    "Prediction Error"
)



plt.xlabel(
    "Time"
)


plt.ylabel(
    "Error"
)


plt.grid()
plt.tight_layout()



plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "error.png"
    ),
    dpi=300
)


plt.close()



# ==================================================
# Plot 3
# Direction
# ==================================================

plt.figure(
    figsize=(12,4)
)


plt.plot(
    actual_direction.values,
    label="Actual"
)


plt.plot(
    pred_direction.values,
    label="Prediction"
)


plt.title(
    "Market Direction Prediction"
)


plt.yticks(
    [0,1],
    [
        "Down",
        "Up"
    ]
)


plt.legend()
plt.grid()
plt.tight_layout()



plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "direction.png"
    ),
    dpi=300
)


plt.close()



print(
    "\nFinal evaluation completed."
)