import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def train_classification_models(csv_file_path, target_column, test_size=0.2, random_state=27):

    # Load the dataset
    print(f"Loading dataset from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)

    # Display basic information about the dataset
    print("\nDataset Information:")
    print(f"Shape: {df.shape}")
    print("\nFeature Preview:")
    print(df.head())

    # Extract features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Handle categorical features if any
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"\nEncoding {len(categorical_cols)} categorical features...")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    # Feature scaling (important for Gaussian NB)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    print(f"\nSplit dataset into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples")

    # Initialize different Naive Bayes classifiers
    models = {
        'Gaussian NB': GaussianNB(),
        'SVM': SVC(random_state=random_state),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state)
    }

    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        try:
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {accuracy:.4f}")

            # Generate classification report
            class_report = classification_report(y_test, y_pred)
            print(f"\n{name} Classification Report:")
            print(class_report)

            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': cm,
                'predictions': y_pred
            }

            # Save feature importance for Random Forest
            if name == 'Random Forest':
                feature_names = X.columns if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                print("\nTop 10 Important Features:")
                print(feature_importance.head(10))
                results[name]['feature_importance'] = feature_importance

        except Exception as e:
            print(f"Error training {name}: {str(e)}")

    # Identify the best model based on accuracy
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.4f}")

    return results, scaler


def plot_confusion_matrices(results, save_path="confusion_matrices.png"):
    """Plot confusion matrices for all models"""
    plt.figure(figsize=(15, 5))

    for i, (name, result) in enumerate(results.items(), 1):
        plt.subplot(1, len(results), i)
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

    plt.tight_layout()
    # Save the figure instead of showing it
    plt.savefig(save_path)
    print(f"Confusion matrices saved to {save_path}")
    plt.close()


def plot_model_comparison(results, save_path="model_comparison.png"):
    model_names = list(results.keys())
    accuracies = [result['accuracy'] for result in results.values()]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color='skyblue')

    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f'{acc:.4f}',
                 ha='center', va='bottom')

    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, max(accuracies) + 0.1)  # Add some space above bars
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Model comparison saved to {save_path}")
    plt.close()


def plot_feature_importance(feature_importance_df, save_path="feature_importance.png", top_n=10):
    """
    Plot top N important features from Random Forest

    Parameters:
    feature_importance_df: DataFrame with feature importance values
    save_path: Path to save the plot image file
    top_n: Number of top features to display
    """
    top_features = feature_importance_df.head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importance (Random Forest)')
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Feature importance plot saved to {save_path}")
    plt.close()

def predict_with_best_model(model, new_data, scaler=None):
    """Make predictions with the best model"""
    if scaler is not None:
        new_data = scaler.transform(new_data)
    return model.predict(new_data)





csv_file_path = "total.csv"
target_column = "Burn_Classification"

# Train the models
results, scaler = train_classification_models(csv_file_path, target_column)

# Plot confusion matrices
if results:
    plot_confusion_matrices(results)
    plot_model_comparison(results)

    # Plot feature importance if Random Forest was trained
    if 'Random Forest' in results and 'feature_importance' in results['Random Forest']:
        plot_feature_importance(
            results['Random Forest']['feature_importance'],
           "feature_importance.png"
        )

    # Extract the best model
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = results[best_model_name]['model']

    print(f"\nYou can now use the best model ({best_model_name}) for predictions!")
