import streamlit as st
import pandas as pd
import pickle
from uuid import uuid4
from random import randint


max_row = 1470
preprocessed_data_files_path = "preprocessed/{}.pkl"
result_files_path = "result/{}.pkl"
MODEL_NAMES = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "Decision Tree": "decision_tree",
    "XGBoost": "xgboost",
}
DEMO_TYPES = ("From Origin Data", "From New Input")
ORIGIN_CHOICES = ("Range", "Text Input")


def read_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_model(normalization_method="z_score_norm"):
    return {
        "logistic_regression": read_pickle(
            result_files_path.format(
                f"{normalization_method}_normal_logistic_regression"
            )
        ),
        "decision_tree": read_pickle(
            result_files_path.format(f"{normalization_method}_normal_decision_tree")
        ),
        "random_forest": read_pickle(
            result_files_path.format(f"{normalization_method}_normal_random_forest")
        ),
        "xgboost": read_pickle(
            result_files_path.format(f"{normalization_method}_normal_xgboost")
        ),
    }


def load_transformer(normalization_method="z_score_norm"):
    return read_pickle(
        preprocessed_data_files_path.format(f"{normalization_method}_ct")
    )


def load_data(normalization_method="z_score_norm"):
    X = read_pickle(preprocessed_data_files_path.format(f"X_{normalization_method}"))
    y = read_pickle(preprocessed_data_files_path.format("Y_raw"))
    return X, y


def convert_attrition(x):
    return tuple(map(lambda x: "Yes" if x else "No", x))


def convert_accuracy(x):
    return tuple(map(str, x))


def convert_text_input(x):
    try:
        return int(x)
    except:
        return None


@st.cache
def get_unique_value(series):
    return tuple(series.unique())


@st.cache
def get_min_max_value(series):
    return series.min(), series.max()


def precision(r1, r2):
    a = tuple(i[0] == i[1] for i in zip(r1, r2))
    b = len(tuple(filter(lambda x: x, a)))
    c = len(a)
    return a, b / c


class App:
    def __init__(self, normalization_method="z_score_norm"):
        self.models = load_model(normalization_method)
        self.transformer = load_transformer(normalization_method)
        self.X, self.y = load_data(normalization_method)
        self.df = read_pickle(preprocessed_data_files_path.format("X_df"))

    def get_model(self, algo):
        model_name = MODEL_NAMES.get(algo)
        return self.models.get(model_name)

    def predict(self, data, model):
        X_test = self.transformer.transform(data)
        y_pred = model.predict(X_test)
        return y_pred

    def predict_range_origin(self, min_id, max_id, model):
        t_max_id = max_id + 1
        X_test = self.X[min_id:t_max_id]
        y_test = self.y[min_id:t_max_id]
        y_pred = model.predict(X_test)
        a, b = precision(y_test, y_pred)
        rs = pd.DataFrame(
            {
                "ID": tuple(range(min_id, t_max_id)),
                "Base": convert_attrition(y_test),
                "Prediction": convert_attrition(y_pred),
                "Accurate": convert_accuracy(a),
            }
        )
        return rs, b

    def predict_text_origin(self, text, model):
        a = map(convert_text_input, text.split(","))
        ids = tuple(filter(lambda x: x, a))
        X_test = tuple(self.X[i] for i in ids)
        y_test = tuple(self.y[i] for i in ids)
        y_pred = model.predict(X_test)
        a, b = precision(y_test, y_pred)
        rs = pd.DataFrame(
            {
                "ID": ids,
                "Base": convert_attrition(y_test),
                "Prediction": convert_attrition(y_pred),
                "Accurate": convert_accuracy(a),
            }
        )
        return rs, b


def main():
    st.set_page_config(layout="wide")
    st.title("Employee Attrition")
    app = App()
    demo_type = st.sidebar.radio("Demo Type", DEMO_TYPES)
    algo = st.sidebar.radio("Select model", tuple(MODEL_NAMES))
    model = app.get_model(algo)
    if demo_type == DEMO_TYPES[1]:
        columns = tuple(app.df.columns)
        input_dict = dict()
        with st.form("form_1"):
            for r in range(5):
                st_col = st.columns(6)
                for c in range(6):
                    with st_col[c]:
                        col = columns[r * 6 + c]
                        series = app.df[col]
                        if series.dtype == object:
                            values = get_unique_value(series)
                            rand_index = randint(0, len(values) - 1)
                            input_dict[col] = st.selectbox(
                                col, values, key=str(uuid4()), index=rand_index
                            )
                        else:
                            min_v, max_v = get_min_max_value(series)
                            input_dict[col] = st.number_input(
                                col,
                                min_value=0,
                                step=1,
                                value=randint(min_v, max_v),
                                key=str(uuid4()),
                            )
            if st.form_submit_button("Submit"):
                input_df = pd.DataFrame(input_dict, index=[0])
                st.write("Input data")
                st.dataframe(input_df)
                rs = app.predict(input_df, model)
                st.write(f"Predicted Attrition: {convert_attrition(rs)[0]}")

        return

    choices = st.radio("ID selection method", ORIGIN_CHOICES)
    if choices == ORIGIN_CHOICES[0]:
        with st.form("form_2"):
            col1, col2 = st.columns(2)
            with col1:
                min_id = st.number_input(
                    "Enter min id", min_value=0, max_value=max_row, step=1
                )
            with col2:
                max_id = st.number_input(
                    "Enter max id", min_value=0, max_value=max_row, step=1
                )
            if st.form_submit_button("Submit"):
                df, p = app.predict_range_origin(min_id, max_id, model)
                st.table(df)
                st.write(f"Precision: {p}")

        return

    with st.form("form_3"):
        ids_input = st.text_input(
            "Enter list of ids, seperated by a comma",
            help="For e.g: 1,2,3,4,5,6,7,8,9,10",
        )
        if st.form_submit_button("Submit"):
            df, p = app.predict_text_origin(ids_input, model)
            st.table(df)
            st.write(f"Precision: {p}")


if __name__ == "__main__":
    main()
