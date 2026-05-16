import matplotlib.pyplot as plt
import seaborn as sns


def residual_plot(y_true, y_pred):

    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        x=y_pred,
        y=residuals
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.xlabel("Predicted Values")

    plt.ylabel("Residuals")

    plt.title("Residual Analysis")

    plt.show()


def prediction_plot(y_true, y_pred):

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        x=y_true,
        y=y_pred
    )

    plt.xlabel("Actual Values")

    plt.ylabel("Predicted Values")

    plt.title("Actual vs Predicted")

    plt.show()
