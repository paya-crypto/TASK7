# Task 7: Support Vector Machines (SVM)

## 📌 Objective
Use Support Vector Machines (SVM) for **linear** and **non-linear** classification on a real-world dataset. 
This task demonstrates how different SVM kernels perform in classifying data and how decision boundaries are visualized using 2D feature spaces.

---

## 🛠️ Tools Used
- **Scikit-learn**: for model building and evaluation  
- **NumPy**: for numerical operations  
- **Matplotlib**: for plotting decision boundaries  

---

## 📁 Dataset
**Name**: Iris Dataset (`Iris.csv`)  
**Source**: UCI Machine Learning Repository / Preloaded in Scikit-learn  
**Preprocessing**:
- Selected only two classes: `Iris-setosa` and `Iris-versicolor` for **binary classification**
- Selected two features: `SepalLengthCm` and `SepalWidthCm` for **2D visualization**

---

## ✅ Steps Followed

1. **Load and prepare a dataset** for binary classification  
2. **Train SVM models** using:
   - Linear kernel
   - RBF (non-linear) kernel  
3. **Visualize decision boundaries** using 2D plots  
4. **Tune hyperparameters** like `C` and `gamma`  
5. **Evaluate model performance** using **5-fold cross-validation**

---

## 📊 Results

### 🔷 SVM with Linear Kernel
- Produced a straight decision boundary
- Performed well on linearly separable features

### 🔷 SVM with RBF Kernel
- Captured non-linear patterns
- Produced a flexible decision boundary

### 🔁 Cross-Validation Scores
- **Linear Kernel**: **0.99**
- **RBF Kernel**: **0.99**
  
---

## 📈 Visualizations

- **Figure 1**: Decision boundary for **Linear Kernel**
- **Figure 2**: Decision boundary for **RBF Kernel**

Each plot shows how SVM separates the two classes in feature space.

---

## 🧠 Learnings

- SVMs with linear kernels are ideal for linearly separable data.
- RBF kernels adapt to more complex boundaries but may overfit if not tuned properly.
- Visualizing SVM boundaries in 2D helps understand model behavior intuitively.
- Cross-validation is essential for reliable model evaluation.

---
