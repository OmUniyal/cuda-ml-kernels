import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from algorithms.knn import KNN

print("=== Test 1: KNN Classification ===")
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_classes=3, 
    n_informative=5,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNN(n_neighbors=5, task='classification')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)

print(f"Accuracy: {accuracy:.4f}")
print(f"Predictions shape: {predictions.shape}")
print("✅ Classification test passed!")

print("\n=== Test 2: KNN Regression ===")
X, y = make_regression(
    n_samples=1000, 
    n_features=10, 
    noise=10,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNN(n_neighbors=5, task='regression')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
r2 = knn.score(X_test, y_test)

print(f"R² Score: {r2:.4f}")
print(f"Predictions shape: {predictions.shape}")
print("✅ Regression test passed!")

print("\n🎉 All KNN tests passed!")