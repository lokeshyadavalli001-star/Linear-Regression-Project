import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COLUMN
)

from src.feature_engineering import (
    engineer_features
)

from src.preprocessing import (
    build_preprocessor
)

from src.evaluation import (
    evaluate_model
)

from src.visualization import (
    residual_plot,
    prediction_plot
)


def train_pipeline(df: pd.DataFrame):

    # Feature Engineering

    df = engineer_features(df)

    # Features and Target

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    # Feature Types

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns

    # Preprocessing

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    # Complete Pipeline

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                Ridge()
            )
        ]
    )

    # Hyperparameter Grid

    param_grid = {
        "model__alpha": [
            0.01,
            0.1,
            1.0,
            10.0,
            100.0
        ]
    }

    # Train Test Split

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
    )

    # Grid Search Cross Validation

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="r2",
        cv=5,
        n_jobs=-1
    )

    print("\nTraining Model...\n")

    grid_search.fit(
        X_train,
        y_train
    )

    # Best Model

    best_model = (
        grid_search.best_estimator_
    )

    print(
        "Best Hyperparameters:"
    )

    print(
        grid_search.best_params_
    )

    # Cross Validation Scores

    cv_scores = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )

    print("\nCross Validation Scores")

    print(cv_scores)

    print(
        f"\nAverage CV Score:"
        f" {cv_scores.mean():.4f}"
    )

    # Predictions

    y_pred = best_model.predict(
        X_test
    )

    # Evaluation

    metrics = evaluate_model(
        y_test,
        y_pred
    )

    # Visualization

    residual_plot(
        y_test,
        y_pred
    )

    prediction_plot(
        y_test,
        y_pred
    )

    return best_model, metrics
