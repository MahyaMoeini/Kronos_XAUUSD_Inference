# plotting functions
import matplotlib.pyplot as plt



def plot_prediction(
        actual,
        predicted,
        save_path
):

    plt.figure(figsize=(12,5))

    plt.plot(
        actual,
        label="Actual"
    )

    plt.plot(
        predicted,
        label="Predicted"
    )

    plt.legend()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



def plot_error(
        errors,
        save_path
):

    plt.figure(figsize=(12,5))

    plt.plot(errors)

    plt.title(
        "Prediction Error"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()