import numpy as np
import pandas as pd
import kagglehub
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data():
    print("Download dataset...")
    path = kagglehub.dataset_download("nikhil7280/weather-type-classification")
    print(f"Dataset disimpan pada: {path}")
    
    csv_file = os.path.join(path, "weather_classification_data.csv")
    df = pd.read_csv(csv_file)
    print(f"Memuat dataset. Shape: {df.shape}")

    df_numerik = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    print(f"Kolom numerik: {df_numerik}")

    def remove_outliers_iqr(df, columns):
        cleaned_df = df.copy()
        for col in columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            before = cleaned_df.shape[0]
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower) & (cleaned_df[col] <= upper)]
            after = cleaned_df.shape[0]
            print(f"Menghapus {before - after} outliers dari '{col}'")
        return cleaned_df

    print("Menghapus outliers menggunakan metode IQR...")
    cleaned_df = remove_outliers_iqr(df, df_numerik)
    print(f"Shape setelah menghapus outlier: {cleaned_df.shape}")

    columns_to_drop = ['Season', 'Visibility (km)', 'Location']
    preprocessed_df = cleaned_df.drop(columns=columns_to_drop)
    print(f"Drop kolom: {columns_to_drop}")
    print(f"Kolom setelah melakukan dropping: {list(preprocessed_df.columns)}")

    X = preprocessed_df.drop(columns=['Weather Type'])
    y = preprocessed_df['Weather Type']
    print(f"Features shape (X): {X.shape}")
    print(f"Target shape (y): {y.shape}")

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    print(f"Categorical features for encoding: {categorical_features}")
    print(f"Numerical features for scaling: {numerical_features}")

    X_encoded = pd.get_dummies(X, columns=categorical_features, prefix=categorical_features)
    print(f"Shape setelah encoding: {X_encoded.shape}")

    scaler = StandardScaler()
    X_scaled = X_encoded.copy()
    X_scaled[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])
    print(f"Shape setelah normalisasi: {X_scaled.shape}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Target variable encoded. Shape: {y_encoded.shape}")
    print(f"Target classes: {label_encoder.classes_}")

    final_processed_df = X_scaled.copy()
    final_processed_df['Weather_Type_Encoded'] = y_encoded
    print(f"Final shape: {final_processed_df.shape}")

    return X_scaled, y_encoded, scaler, label_encoder, final_processed_df


def save_preprocessed_data(df):
    file_path = os.path.join(os.getcwd(), 'final_weather_data.csv')
    df.to_csv(file_path, index=False)
    print(f"Final dataset disimpan ke {file_path}")


if __name__ == "__main__":
    X, y, fitted_scaler, fitted_label_encoder, final_df = load_and_preprocess_data()
    
    save_preprocessed_data(final_df)
    joblib.dump(fitted_scaler, 'scaler.pkl')
    joblib.dump(fitted_label_encoder, 'label_encoder.pkl')

    print(f"Final features shape: {X.shape}")
    print(f"Final target shape: {y.shape}")
