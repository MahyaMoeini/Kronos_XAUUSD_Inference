import pandas as pd



def load_data(path):
    """
    Load XAUUSD candle data.
    
    Expected columns:
    timestamp, open, high, low, close, volume
    """

    df = pd.read_csv(path)

    return df




def clean_data(
    df,
    start_date="2022-01-01"
):
    """
    Clean and prepare XAUUSD data.

    Steps:
    - Remove duplicates
    - Convert timestamp
    - Sort chronologically
    - Filter date range
    - Remove missing values
    - Remove unused columns
    - Keep only OHLCV features
    """

    # Remove duplicate rows

    df = df.drop_duplicates()



    # Timestamp processing

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df = df.sort_values(
            "timestamp"
        )


        df = df[
            df["timestamp"] >= start_date
        ]



    # Remove missing values

    df = df.dropna()



    # Remove unnecessary columns

    if "timeframe" in df.columns:

        df = df.drop(
            columns=["timeframe"]
        )



    # Remove engineered features

    if "amount" in df.columns:

        df = df.drop(
            columns=["amount"]
        )



    # Keep only Kronos input features

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]


    df = df[
        [
            col
            for col in required_columns
            if col in df.columns
        ]
    ]



    # Reset index

    df = df.reset_index(
        drop=True
    )


    return df




def create_windows(
    df,
    input_size=600,
    prediction_size=120
):
    """
    Create input and target windows.

    Input:
        Past candles

    Target:
        Future candles
    """

    windows = []


    total = len(df)



    for i in range(
        total - input_size - prediction_size
    ):

        past = df.iloc[
            i:
            i + input_size
        ]


        future = df.iloc[
            i + input_size:
            i + input_size + prediction_size
        ]



        windows.append(
            {
                "input": past,
                "target": future
            }
        )



    return windows




def inspect_data(path):

    df = pd.read_csv(
        path
    )


    print(
        "Shape:"
    )

    print(
        df.shape
    )


    print(
        "\nColumns:"
    )

    print(
        df.columns.tolist()
    )


    print(
        "\nFirst rows:"
    )

    print(
        df.head()
    )


    print(
        "\nMissing values:"
    )

    print(
        df.isnull().sum()
    )




if __name__ == "__main__":


    input_path = (
        "data/XAUUSD_M5.csv"
    )


    output_path = (
        "data/XAUUSD_clean.csv"
    )



    print(
        "Loading data..."
    )


    df = load_data(
        input_path
    )



    print(
        "Cleaning data..."
    )


    clean_df = clean_data(
        df
    )



    clean_df.to_csv(
        output_path,
        index=False
    )



    print(
        "Saved:"
    )


    print(
        output_path
    )


    print(
        "Final shape:"
    )


    print(
        clean_df.shape
    )


    print(
        "\nColumns:"
    )


    print(
        clean_df.columns.tolist()
    )