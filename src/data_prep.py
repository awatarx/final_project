import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    # Define file paths relative to the script execution (assuming root folder)
    input_path = os.path.join('../data', 'heart.csv')
    output_path = os.path.join('../data', 'heart_cleaned.csv')

    print("Loading data...")
    try:
        df = pd.read_csv(input_path)
        df.info()  # Print info to check for missing values and data types
        df.rename(columns={
                            'chest_pain_type': 'cp',
                            'resting_bp': 'trestbps',
                            'cholestoral': 'chol',
                            'fasting_blood_sugar': 'fbs',
                            'max_hr': 'thalach',
                            'num_major_vessels': 'ca'
                            }, inplace=True)
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}. Make sure you run this from the project root.")
        return

    # Define columns based on UCI dataset specs
    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    print("Encoding categorical variables...")
    # Convert categorical variables into dummy variables (One-Hot Encoding)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    print("Scaling continuous features...")
    # Scale continuous features
    scaler = StandardScaler()
    df_encoded[continuous_cols] = scaler.fit_transform(df_encoded[continuous_cols])

    # Save the cleaned dataset for the next phase
    df_encoded.to_csv(output_path, index=False)
    print(f"Data preprocessing complete! Cleaned dataset saved to: {output_path}")

if __name__ == "__main__":
    preprocess_data()