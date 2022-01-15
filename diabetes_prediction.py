'''

# Pima Indian Diabetes Prediction

The aims of this project is to predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Logistic Regression is used as the prediction method.

Variables included in the dataset : 

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age
* Outcome

Dataset source : https://www.kaggle.com/uciml/pima-indians-diabetes-database

'''

# 1. Import Libraries
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix

plt.style.use('fivethirtyeight')

# 2. Load the dataset
df = pd.read_csv("dataset/diabetes.csv")
print(df.head(10))

# 3. Generate descriptive statistics of the dataset
print('\nData Description :\n', df.describe())

# 4. Show the class distribution and the skewness in the dataset
class_dist = df.groupby('Outcome').size()
print('Class Distribution : \n{}\n'.format(class_dist))
skew = df.skew()
print('Skew : \n{}\n'.format(skew))

# 5. Calculate the Pearson's correlation of each variables
print(df.corr(method='pearson'))

# 6. Show the histogram plot of each variables
df.hist(figsize=(20,10));
plt.show()

# 7. Show the density plot of each variables
df.plot(kind='density', subplots=True, layout=(3,3), figsize=(20,10));
plt.show()\

# 8. Split the dependent and independent variables in the dataset
x = df.iloc[:, 0:8]
y = df.iloc[:, 8]

# 9. Scales the independent variables
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 10. Split the dataset into the training and the testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 11. Search the best n number of neighbors for the kNN model
knn_result = []
for n in range (2,9):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_train, y_train)
    knn_result.append(knn.score(x_train, y_train))

best_n = knn_result.index(max(knn_result)) + 2

print('The best n value : {} with accuracy : {}'.format(best_n, max(knn_result)))

# 12. Visualize the accuracy of kNN values with k 2-8
plt.figure(figsize=(20,6))
plt.plot([i for i in range(2, 9)], knn_result)
plt.xlabel('k Neighbors')
plt.ylabel('Accuracy Score')
plt.title('k-Nearest Neighbors')
plt.show()

# 13. Create the kNN and logistic regression model
knn = KNeighborsClassifier(n_neighbors=3)
knn_fit = knn.fit(x_train, y_train)

log = LogisticRegression(max_iter=1000)
log_fit = log.fit(x_train, y_train.values.ravel())

# 14. Predict the training and the testing set
knn_train_pred = knn_fit.predict(x_train)
knn_test_pred = knn_fit.predict(x_test)
log_train_pred = log_fit.predict(x_train)
log_test_pred = log_fit.predict(x_test)

print('kNN :')
print('Model accuracy on the training set : {}'.format(accuracy_score(y_train, knn_train_pred)))
print('Model accuracy on the testing set : {}'.format(accuracy_score(y_test, knn_test_pred)))
print('\nLogistic Regression :')
print('Model accuracy on the training set : {}'.format(accuracy_score(y_train, log_train_pred)))
print('Model accuracy on the testing set : {}'.format(accuracy_score(y_test, log_test_pred)))

# 15. Show the classification report of kNN and logistic regression model
print('Classification Report of kNN Classifier: \n\n {}'.format(classification_report(y_test, knn_test_pred)))
print('Classification Report of Logistic Regression: \n\n {}'.format(classification_report(y_test, log_test_pred)))

# 16. Visualize the confusion matrix of the logistic regression model
fig=plt.figure(figsize=(20,8))
fig.suptitle('Confusion Matrix', fontsize=20)
ax1 = plt.subplot(1,2,1)
ax1.set_title('Training Set')
plot_confusion_matrix(knn, x_train, y_train, cmap=plt.cm.Blues, ax=ax1)

ax2 = plt.subplot(1,2,2)
ax2.set_title('Testing Set')
plot_confusion_matrix(knn, x_test, y_test, cmap=plt.cm.Blues, ax=ax2)
plt.show()