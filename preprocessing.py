import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

fn = 'ibm_hr/{}.pkl'


def write_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':
    df = pd.read_csv('ibm_hr.csv')
    df = df.drop(columns=[
        'StandardHours',
        'EmployeeCount',
        'Over18',
        'EmployeeNumber'
    ])
    encoder = LabelEncoder()
    df['Attrition'] = encoder.fit_transform(df['Attrition'])

    categorical_features = []
    numerical_features = []
    for column in df.columns:
        if df[column].dtype == object:
            categorical_features.append(column)
        else:
            numerical_features.append(column)
    numerical_features.remove('Attrition')

    X_raw = df.drop(columns=['Attrition'])
    y_raw = df['Attrition']
    std_ct = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
    X_std = std_ct.fit_transform(X_raw)
    norm_ct = ColumnTransformer([
        ('num', Normalizer(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
    X_norm = norm_ct.fit_transform(X_raw)

    # Save data
    write_pickle(X_raw, fn.format('X_df'))
    write_pickle(y_raw, fn.format('y_raw'))
    write_pickle(X_std, fn.format('X_std'))
    write_pickle(X_norm, fn.format('X_norm'))
    write_pickle(std_ct, fn.format('std_ct'))
    write_pickle(norm_ct, fn.format('norm_ct'))
