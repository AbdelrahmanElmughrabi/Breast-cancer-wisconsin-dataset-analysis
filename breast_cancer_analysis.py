
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# A)
data = pd.read_csv('wdbc.data', header=None)
data.columns = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
data = data.drop(['ID'], axis=1)
data['Diagnosis'] = data['Diagnosis'].map({'B': 0, 'M': 1})

X = data.iloc[:, 1:].values  
y = data['Diagnosis'].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Visualize class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Diagnosis', data=data)
plt.title('Class Distribution')
plt.xlabel('Diagnosis (0 = Benign, 1 = Malignant)')
plt.ylabel('Count')
plt.show()

# Feature correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(data.iloc[:, 1:].corr(), cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.show()

# Calculating the prior probabilities
prior_benign = (y_train == 0).mean()
prior_malignant = (y_train == 1).mean()

print(f"Prior probability of benign: {prior_benign}")
print(f"Prior probability of malignant: {prior_malignant}")

# B) Estimating the mean and covariance for each class
reg_const = 1e-6 

X_train_benign = X_train[y_train == 0]
X_train_malignant = X_train[y_train == 1]

mean_benign = np.mean(X_train_benign, axis=0)
cov_benign = np.cov(X_train_benign, rowvar=False) + reg_const * np.eye(X_train_benign.shape[1])

mean_malignant = np.mean(X_train_malignant, axis=0)
cov_malignant = np.cov(X_train_malignant, rowvar=False) + reg_const * np.eye(X_train_malignant.shape[1])

# Classifying using MAP rule
y_pred = []
for x in X_test:
    prob_benign = multivariate_normal.pdf(x, mean=mean_benign, cov=cov_benign, allow_singular=True) * prior_benign
    prob_malignant = multivariate_normal.pdf(x, mean=mean_malignant, cov=cov_malignant, allow_singular=True) * prior_malignant
    y_pred.append(1 if prob_malignant > prob_benign else 0)

# Determining the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix for original model:")
print(conf_matrix)

# Visualizing the original confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Benign', 'Predicted Malignant'], 
            yticklabels=['Actual Benign', 'Actual Malignant'])
plt.title('Confusion Matrix (Original Model)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#Calculating Type I and Type II Errors
type_I_error_count = conf_matrix[0, 1]
type_II_error_count = conf_matrix[1, 0]
print(f"Type I error count (false alarm): {type_I_error_count}")
print(f"Type II error count (miss): {type_II_error_count}")

#Reducing Type II Errors by adjusting the threshold
threshold = 0.9
y_pred_adjusted = []
for x in X_test:
    prob_benign = multivariate_normal.pdf(x, mean=mean_benign, cov=cov_benign, allow_singular=True) * prior_benign
    prob_malignant = multivariate_normal.pdf(x, mean=mean_malignant, cov=cov_malignant, allow_singular=True) * prior_malignant
    y_pred_adjusted.append(1 if prob_malignant > threshold * prob_benign else 0)

#Evaluate the adjusted model
conf_matrix_adjusted = confusion_matrix(y_test, y_pred_adjusted)
print("Confusion matrix for adjusted model:")
print(conf_matrix_adjusted)

# Visualizing the adjusted confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_adjusted, annot=True, fmt='d', cmap='Oranges', xticklabels=['Predicted Benign', 'Predicted Malignant'], 
            yticklabels=['Actual Benign', 'Actual Malignant'])
plt.title('Confusion Matrix (Adjusted Model)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


selected_feature = 1
plt.figure(figsize=(10, 6))
sns.kdeplot(X_train_benign[:, selected_feature], label='Benign', fill=True)
sns.kdeplot(X_train_malignant[:, selected_feature], label='Malignant', fill=True)
plt.title(f'Probability Distribution for Feature {selected_feature}')
plt.xlabel(f'Feature {selected_feature} Value')
plt.ylabel('Density')
plt.legend()
plt.show()
