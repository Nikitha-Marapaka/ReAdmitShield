import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib
import os
import traceback

try:
    print("🔹 Starting training process...")

    # Load dataset
    data_path = os.path.abspath("data/simulated_ehr.csv")
    print(f"📂 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded dataset with {len(df)} records")

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    print("🔠 Encoded categorical variables.")

    # Define features and target
    X = df.drop('readmitted_30', axis=1)
    y = df['readmitted_30']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    print("🤖 Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("✅ Model training complete.")

    # Save model and encoders
    model_dir = os.path.abspath("models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "readmit_model.pkl")
    encoder_path = os.path.join(model_dir, "label_encoders.pkl")

    joblib.dump(model, model_path)
    joblib.dump(label_encoders, encoder_path)

    print(f"💾 Model saved at: {model_path}")
    print(f"💾 Label encoders saved at: {encoder_path}")
    print("🎉 Training completed successfully!")

except Exception as e:
    print("❌ ERROR during training:")
    traceback.print_exc()

