import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os

def preprocess_data(df: pd.DataFrame):
    """Melakukan tahapan preprocessing (replace 0, handle outlier, scaling, SMOTE)"""
    # Ganti nilai nol dengan median
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zero:
        median_value = df[col].median()
        df[col] = df[col].replace(0, median_value)

    # Tangani outlier
    for col in df.columns[:-1]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper, upper,
                    np.where(df[col] < lower, lower, df[col]))

    # Pisahkan fitur dan label
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Standarisasi
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Balancing data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    return X_train, X_test, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir="preprocessing/pima_diabetes_processed"):
    """Menyimpan hasil preprocessing ke folder output"""
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_path = os.path.join(output_dir, "train_diabetes_processed.csv")
    test_path = os.path.join(output_dir, "test_diabetes_processed.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
