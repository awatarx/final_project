import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_evaluate():
    data_path = os.path.join('../data', 'heart_cleaned.csv')
    image_dir = os.path.join('../paper', 'images')
    
    # Ensure the image directory exists
    os.makedirs(image_dir, exist_ok=True)

    print("Loading cleaned data...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Run src/data_prep.py first.")
        return

    # Split into features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # Train/Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
    }

    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        # Generate and save Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted 0', 'Predicted 1'], 
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'{name} - Confusion Matrix')
        plt.tight_layout()
        
        # Save plot to paper/images/
        filename = name.lower().replace(" ", "_") + "_cm.png"
        save_path = os.path.join(image_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"  -> Confusion Matrix saved to: {save_path}")

if __name__ == "__main__":
    train_and_evaluate()