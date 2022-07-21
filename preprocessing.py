import pickle
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.compose import ColumnTransformer

preprocessed_data_path = "preprocessed/{}.pkl"


def write_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def outlierTreat(x):
    """
    Using interquartile range
    """
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1

    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    return x.clip(lower, upper)


if __name__ == "__main__":
    df = pd.read_csv("ibm_hr.csv")
    df = df.drop(columns=["StandardHours", "EmployeeCount", "Over18", "EmployeeNumber"])

    # Outlier removal
    features_with_outlier = [
        "MonthlyIncome",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "YearsAtCompany",
        "YearsSinceLastPromotion",
        "YearsInCurrentRole",
        "YearsWithCurrManager",
    ]
    df.loc[:, features_with_outlier] = df.loc[:, features_with_outlier].apply(
        outlierTreat
    )

    encoder = LabelEncoder()
    df["Attrition"] = encoder.fit_transform(df["Attrition"])

    categorical_features = [
        column for column in df.columns if df[column].dtype == object
    ]
    numerical_features = [
        column
        for column in df.columns
        if df[column].dtype != object and column != "Attrition"
    ]

    # Raw data
    X_raw = df.drop(columns=["Attrition"])
    Y_raw = df["Attrition"]
    write_pickle(X_raw, preprocessed_data_path.format("X_df"))
    write_pickle(Y_raw, preprocessed_data_path.format("Y_raw"))

    # Z-score normalization
    z_score_norm_ct = ColumnTransformer(
        [
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )
    X_z_score = z_score_norm_ct.fit_transform(X_raw)
    write_pickle(X_z_score, preprocessed_data_path.format("X_z_score_norm"))
    write_pickle(z_score_norm_ct, preprocessed_data_path.format("z_score_norm_ct"))

    # Min-max normalizer
    min_max_norm_ct = ColumnTransformer(
        [
            ("num", MinMaxScaler(), numerical_features),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )
    X_min_max_norm = min_max_norm_ct.fit_transform(X_raw)
    write_pickle(X_min_max_norm, preprocessed_data_path.format("X_min_max_norm"))
    write_pickle(min_max_norm_ct, preprocessed_data_path.format("min_max_norm_ct"))
