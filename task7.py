# Task 7: Support Vector Machines (SVM)
# Objective: Use SVM to classify data using different kernels and visualize the decision boundaries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Step 1: Load Dataset
df = pd.read_csv("C:/Users/jay30/OneDrive/Documents/myprojects/python/Iris.csv")

# Step 2: Filter for Binary Classification (Setosa = 0, Versicolor = 1)
df = df[df['Species'].isin(['Iris-setosa', 'Iris-versicolor'])]

# Step 3: Select 2 features for 2D visualization
X = df[['SepalLengthCm', 'SepalWidthCm']].values
y = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1}).values

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train SVM Classifiers
svm_linear = SVC(kernel='linear', C=1)
svm_rbf = SVC(kernel='rbf', C=1, gamma=0.5)

svm_linear.fit(X_train_scaled, y_train)
svm_rbf.fit(X_train_scaled, y_train)

# Step 7: Plot Decision Boundaries
def plot_decision_boundary(model, X, y, title):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel('SepalLengthCm (scaled)')
    plt.ylabel('SepalWidthCm (scaled)')
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_decision_boundary(svm_linear, X_train_scaled, y_train, "SVM - Linear Kernel (Training Data)")
plot_decision_boundary(svm_rbf, X_train_scaled, y_train, "SVM - RBF Kernel (Training Data)")

# Step 8: Cross-Validation Accuracy
linear_cv = cross_val_score(svm_linear, X_train_scaled, y_train, cv=5)
rbf_cv = cross_val_score(svm_rbf, X_train_scaled, y_train, cv=5)

print(f"Linear SVM Accuracy (CV): {linear_cv.mean():.2f}")
print(f"RBF SVM Accuracy (CV): {rbf_cv.mean():.2f}")
