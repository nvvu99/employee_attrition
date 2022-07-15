# Utilities
import pickle
import argparse

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# GridSearchCV
from sklearn.model_selection import GridSearchCV

# PCA
from sklearn.decomposition import PCA

# imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# xgboost
from xgboost import XGBRFClassifier


def read_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def write_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


fn = "ibm_hr/{}.pkl"
MODELS = {
    "logistic_regression": ("Logistic Regression", LogisticRegression()),
    "decision_tree": ("Decision Tree", DecisionTreeClassifier()),
    "random_forest": ("Random Forest", RandomForestClassifier()),
    "xgboost": ("XGBoost", XGBRFClassifier()),
}

SAMPLINGS = {
    "over_sampling": RandomOverSampler(),
    "under_sampling": RandomUnderSampler(),
}

pca = PCA()
X_std = read_pickle(fn.format("X_std"))
X_norm = read_pickle(fn.format("X_norm"))
y_raw = read_pickle(fn.format("y_raw"))
DATA = {"std": (X_std, y_raw), "norm": (X_norm, y_raw)}
n_features = X_std.shape[1]
SAMPLING_METHODS = ("normal", "under", "over", "combined")


def parse_arguments():
    parser = argparse.ArgumentParser(description="IBM HR")
    parser.add_argument("mode", type=str, default="train", help="Working mode")
    return parser.parse_args()


def evaluate(data_type="std", sampling_method="normal"):
    data = DATA.get(data_type)
    if data:
        X, y = data
        base_steps = [("pca", pca)]
        base_param_grid = {
            "pca__n_components": list(range(1, n_features + 1)),
        }
        if sampling_method == "combined":
            base_steps.append(("over", RandomOverSampler()))
            base_steps.append(("under", RandomUnderSampler()))
        else:
            sampling = SAMPLINGS.get(sampling_method)
            if sampling:
                base_steps.append(("sampling", sampling))
        print(f"\n\nSampling method: {sampling_method}\n")
        eval_result = dict()

        for k, v in MODELS.items():
            # Define steps
            steps = base_steps.copy()
            steps.append((k, v[1]))

            # Define pipeline
            pipe = Pipeline(steps=steps)

            # Define grid parameters
            param_grid = base_param_grid.copy()
            if len(v) == 3:
                param_grid = {**param_grid, **v[2]}

            # GridSearch
            search = GridSearchCV(pipe, param_grid, n_jobs=-1)
            search.fit(X, y)

            # Result
            print(f"{v[0]}: {search.best_score_}")
            eval_result[k] = search.best_score_

            # Save data
            write_pickle(
                search.best_estimator_, fn.format(f"{data_type}_{sampling_method}_{k}")
            )
        write_pickle(eval_result, fn.format(f"rs_{data_type}_{sampling_method}"))


if __name__ == "__main__":
    args = parse_arguments()
    if args.mode == "train":
        evaluate()
    elif args.mode == "eval":
        for d in DATA.keys():
            for s in SAMPLING_METHODS:
                evaluate(d, s)
    else:
        print("Invalid mode!")
