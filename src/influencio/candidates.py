from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


CLASSIFICATION_CANDIDATES = {
    "LogisticRegression": (
        LogisticRegression(solver="liblinear", max_iter=1000),
        {"predictor__C": [0.01, 0.1, 1, 10, 100], "predictor__penalty": ["l1", "l2"]},
    ),
    "RandomForestClassifier": (
        RandomForestClassifier(),
        {
            "predictor__n_estimators": [50, 100, 200],
            "predictor__max_depth": [None, 3, 5, 10],
            "predictor__min_samples_split": [2, 5, 10],
        },
    ),
    "GradientBoostingClassifier": (
        GradientBoostingClassifier(),
        {
            "predictor__n_estimators": [100, 200],
            "predictor__learning_rate": [0.01, 0.1, 0.2],
            "predictor__max_depth": [3, 5, 7],
        },
    ),
    "SVC": (
        SVC(probability=True),
        {
            "predictor__C": [0.01, 0.1, 1, 10],
            "predictor__kernel": ["linear", "rbf"],
            "predictor__gamma": ["scale", "auto"],
        },
    ),
    "KNeighborsClassifier": (
        KNeighborsClassifier(),
        {
            "predictor__n_neighbors": [3, 5, 7, 9],
            "predictor__weights": ["uniform", "distance"],
            "predictor__p": [1, 2],
        },
    ),
}

REGRESSION_CANDIDATES = {
    "LinearRegression": (LinearRegression(), {}),
    "Ridge": (Ridge(), {"predictor__alpha": [0.01, 0.1, 1.0, 10.0]}),
    "Lasso": (Lasso(), {"predictor__alpha": [0.01, 0.1, 1.0, 10.0]}),
    "RandomForestRegressor": (
        RandomForestRegressor(),
        {
            "predictor__n_estimators": [50, 100, 200],
            "predictor__max_depth": [None, 5, 10],
            "predictor__min_samples_split": [2, 5, 10],
        },
    ),
    "GradientBoostingRegressor": (
        GradientBoostingRegressor(),
        {
            "predictor__n_estimators": [100, 200],
            "predictor__learning_rate": [0.01, 0.1],
            "predictor__max_depth": [3, 5],
        },
    ),
}
