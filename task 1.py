
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import random

# ---------- Set Random Seed ----------
RSEED = 42
np.random.seed(RSEED)
random.seed(RSEED)

# ---------- File Path ----------
DATA_FILE = r"C:\Users\Hp\Downloads\train.csv"  

# ---------- Load and Prepare Data ----------
def load_and_prepare(path):
    assert os.path.exists(path), f"File not found: {path}"
    df = pd.read_csv(path)

    # Use features requested: GrLivArea, BedroomAbvGr, FullBath, SalePrice
    cols = ["GrLivArea", "BedroomAbvGr", "FullBath", "SalePrice"]
    df = df[cols].copy()

    # Drop rows with missing values in selected columns
    df = df.dropna()
    return df

# ---------- Train Model ----------
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# ---------- Main Function ----------
def main():
    df = load_and_prepare(DATA_FILE)
    print("✅ Loaded rows:", len(df))

    X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]
    y = df["SalePrice"]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RSEED
    )

    # Train model
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n📊 Mean Squared Error: {mse:.2f}")
    print(f"📈 R-squared: {r2:.4f}")

    # Display coefficients
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })
    print("\nModel Coefficients:")
    print(coef_df)

    # Save model
    joblib.dump(model, "task1_linear_regression_model.joblib")
    print("\n💾 Model saved as task1_linear_regression_model.joblib")

    # ---------- Visualization ----------
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--', linewidth=2
    )
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("Actual vs Predicted — House Prices")
    plt.tight_layout()
    plt.savefig("task1_actual_vs_pred.png", dpi=150)
    plt.show()

# ---------- Run ----------
if __name__ == "__main__":
    main()

