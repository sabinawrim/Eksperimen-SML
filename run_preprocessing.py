from preprocessing.automate_sabina import preprocess_data, save_preprocessed_data
import os
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, "diabetes.csv")   
    output_dir = os.path.join(base_dir, "preprocessing/pima_diabetes_processed")

    # Load data
    df = pd.read_csv(raw_path)

    # Jalankan preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Simpan hasil
    save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir)

if __name__ == "__main__":
    main()
