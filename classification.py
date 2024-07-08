import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
data = pd.read_csv(r'C:\Users\pc\Desktop\Machine learning\magic04.csv')
print(data)

#  Balance the dataset
gamma_indices = data[data.iloc[:, -1] == 'g'].index
hadron_indices = data[data.iloc[:, -1] == 'h'].index
num_extra_gamma = len(gamma_indices) - len(hadron_indices)
balanced_gamma_indices = np.random.choice(gamma_indices, size=len(hadron_indices), replace=True)

balanced_indices = np.concatenate([balanced_gamma_indices, hadron_indices])
balanced_data = data.iloc[balanced_indices]

X = balanced_data.iloc[:, :-1]
y = balanced_data.iloc[:, -1]

#spilit data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

'''#make random rangefrom 1 to 50 for k value to try many values to get best k that achive best  measurements(accuracy,percision,...etc)
# make an empty list for each one to save the values in'''
k_values = list(range(1,50))
accuracies = []
precisions = []
recalls = []
f1_scores = []
confusion_matrices = []

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)

    # Make predictions on the validation data
    y_pred = knn_classifier.predict(X_val)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='g')
    recall = recall_score(y_test, y_pred, pos_label='g')
    f1 = f1_score(y_test, y_pred, pos_label='g')
    confusion = confusion_matrix(y_test, y_pred)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    confusion_matrices.append(confusion)
# Perform cross-validation for different values of k
cv_scores = []
for k in k_values:
    knn_classifier.n_neighbors = k
    scores = cross_val_score(knn_classifier, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
print("Cross-Validation Scores:")
for k, score in zip(k_values, cv_scores):
    print(f"K = {k}: {score}")

# Plot cross-validation scores for different k values
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, label='Cross-Validation Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Cross-Validation Accuracy vs. Number of Neighbors')
plt.legend()
plt.grid(True)
plt.xticks(k_values)
plt.show()

# Plot performance metrics for different k values
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, label='Accuracy')
plt.plot(k_values, precisions, label='Precision')
plt.plot(k_values, recalls, label='Recall')
plt.plot(k_values, f1_scores, label='F1-score')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Performance Metric')
plt.title('Performance Metrics vs. Number of Neighbors')
plt.legend()
plt.grid(True)
plt.xticks(k_values)
plt.show()