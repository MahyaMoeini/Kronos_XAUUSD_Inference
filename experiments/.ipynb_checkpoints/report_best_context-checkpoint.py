import os
import json
import pandas as pd


# ==================================================
# Paths
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "experiments",
    "context_horizon_grid.csv"
)


OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "experiments",
    "best_context_report.json"
)


# ==================================================
# Load CSV
# ==================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"CSV not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(INPUT_FILE)


print(
    f"Loaded {len(df)} experiments"
)


# ==================================================
# Normalize metrics
# ==================================================

def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            [1] * len(series),
            index=series.index
        )

    return (
        series - minimum
    ) / (
        maximum - minimum
    )


# Error metrics
df["RMSE_norm"] = normalize(
    df["RMSE"]
)

df["MAE_norm"] = normalize(
    df["MAE"]
)

df["MAPE_norm"] = normalize(
    df["MAPE"]
)


# Direction metric
df["direction_norm"] = normalize(
    df["Directional_Accuracy"]
)


# ==================================================
# Ranking score
#
# Time execution ignored
#
# RMSE                 40%
# Direction Accuracy   40%
# MAE                  20%
#
# ==================================================

df["score"] = (

    0.40 *
    (1 - df["RMSE_norm"])

    +

    0.40 *
    df["direction_norm"]

    +

    0.20 *
    (1 - df["MAE_norm"])

)


# ==================================================
# Sort best experiments
# ==================================================

ranking = (
    df
    .sort_values(
        by="score",
        ascending=False
    )
)


top10 = ranking.head(10)


best = top10.iloc[0]


# ==================================================
# Create JSON report
# ==================================================

report = {

    "experiment":

        "context_horizon_grid",


    "ranking_method":

    {
        "RMSE_weight": 0.40,
        "Directional_Accuracy_weight": 0.40,
        "MAE_weight": 0.20,
        "time_sec_weight": 0
    },


    "total_tests":

        int(len(df)),


    "best_configuration":

    {

        "input_size":

            int(best["input_size"]),


        "prediction_size":

            int(best["pred_size"]),


        "step":

            int(best["step"]),


        "windows":

            int(best["windows"]),


        "metrics":

        {

            "MAE":

                float(best["MAE"]),


            "RMSE":

                float(best["RMSE"]),


            "MAPE":

                float(best["MAPE"]),


            "Directional_Accuracy":

                float(best["Directional_Accuracy"]),


            "time_sec":

                float(best["time_sec"])

        },


        "final_score":

            float(best["score"])

    },


    "top_10_configurations":

    top10[
        [
            "input_size",
            "pred_size",
            "step",
            "windows",
            "MAE",
            "RMSE",
            "MAPE",
            "Directional_Accuracy",
            "time_sec",
            "score"
        ]
    ]
    .round(6)
    .to_dict(
        orient="records"
    )

}


# ==================================================
# Save JSON
# ==================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4,
        ensure_ascii=False
    )


# ==================================================
# Console output
# ==================================================

print("\n================================")
print("BEST CONFIGURATION")
print("================================")


print(
    json.dumps(
        report["best_configuration"],
        indent=4,
        ensure_ascii=False
    )
)


print("\nJSON saved:")
print(OUTPUT_FILE)