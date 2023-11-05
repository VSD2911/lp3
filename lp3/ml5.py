# Diabetes classification using KNN
import pandas as pd
import numpy as np

data = pd.read_csv("diabetes.csv")
data.head()

data.isnull().any()

data.describe().T

"""Glucose,
BloodPressure,
SkinThickness,
Insulin,
BMI
<br>
columns have values 0 which does not ake sense , hence are missing values
"""

data_copy = data.copy(deep = True)
data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
data_copy.isnull().sum()

"""To fill these Nan values the data distribution needs to be understood"""

p = data.hist(figsize = (20,20))

data_copy['Glucose'].fillna(data_copy['Glucose'].mean(), inplace = True)
data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].mean(), inplace = True)
data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median(), inplace = True)
data_copy['Insulin'].fillna(data_copy['Insulin'].median(), inplace = True)
data_copy['BMI'].fillna(data_copy['BMI'].median(), inplace = True)

p = data_copy.hist(figsize = (20,20))

import missingno as msno
p = msno.bar(data)

p=data.Outcome.value_counts().plot(kind="bar")

"""The above graph shows that the data is biased towards datapoints having outcome value as 0 where it means that diabetes was not present actually. The number of non-diabetics is almost twice the number of diabetic patients"""

import seaborn as sns
p=sns.pairplot(data_copy, hue = 'Outcome')

import matplotlib.pyplot as plt
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(data.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap

plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(data_copy.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(data_copy.drop(["Outcome"], axis =1),),columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])

X.head()

y =data_copy.Outcome

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42, stratify=y)

from sklearn.neighbors import KNeighborsClassifier

train_scores = []
test_scores = []

for i in range(1,15):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

max_test_score =max(test_scores)

test_score_index = [i for i, v in enumerate(test_scores) if v== max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_score_index))))

plt.figure(figsize=(12,5))
p = sns.lineplot(x=range(1, 15), y=train_scores, marker='*', label='Train Score', markers=True)
p = sns.lineplot(x=range(1, 15), y=test_scores, marker='o', label='Test Score', markers=True)

# K=11
#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(11)

knn.fit(X_train,y_train)
knn.score(X_test,y_test)

from mlxtend.plotting import plot_decision_regions
value = 20000
width  =20000

plot_decision_regions(X.values, y.values, clf = knn, legend =2,filler_feature_values={2: value, 3: value, 4: value, 5: value, 6: value, 7: value},
                      filler_feature_ranges={2: width, 3: width, 4: width, 5: width, 6: width, 7: width},
                      X_highlight=X_test.values)
plt.title("KNN with diabetes data")
plt.show()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
y_pred = knn.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_pred)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

def model_evaluation(y_test, y_pred, model_name):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta = 2.0)

    results = pd.DataFrame([[model_name, acc, prec, rec, f1, f2]],
                       columns = ["Model", "Accuracy", "Precision", "Recall",
                                 "F1 SCore", "F2 Score"])
    results = results.sort_values(["Precision", "Recall", "F2 Score"], ascending = False)
    return results



model_evaluation(y_test, y_pred, "KNN")

# Alternate way
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import auc, roc_auc_score, roc_curve

y_pred_proba = knn.predict_proba(X_test)[:,-1]
fpr, tpr, threshold = roc_curve(y_test, y_pred_proba)

classifier_roc_auc = roc_auc_score(y_test, y_pred_proba)
plt.plot([0,1],[0,1], label = "---")

plt.plot(fpr, tpr, label ='KNN (area = %0.2f)' % classifier_roc_auc)
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title('Knn(n_neighbors=11) ROC curve')
plt.legend(loc="lower right", fontsize = "medium")
plt.xticks(rotation=0, horizontalalignment="center")
plt.yticks(rotation=0, horizontalalignment="right")
plt.show()

#Hyper parameters tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters_grid = {"n_neighbors": np.arange(0,50)}
knn= KNeighborsClassifier()
knn_GSV = GridSearchCV(knn, param_grid=parameters_grid, cv = 5)
knn_GSV.fit(X, y)

print("Best Params" ,knn_GSV.best_params_)
print("Best score" ,knn_GSV.best_score_)





"""
Data normalization, also known as feature scaling, is a preprocessing technique used to transform data into a common scale or range. Normalizing data is essential, especially in machine learning algorithms that are sensitive to the magnitude of features. Here are some common methods of data normalization:

1. **Min-Max Scaling (Normalization):**
   - Also known as min-max normalization.
   - Scales features to a specific range, usually between 0 and 1.
   - The formula for min-max scaling is: 
     - [X_normalized = {X - X_{min}}/{X_{max} - X_{min}}]
   - It preserves the relationships between data points.
   - Suitable for algorithms that rely on feature values within a specific range, like neural networks.

2. **Z-Score Standardization (Standardization):**
   - Also known as standardization or z-score normalization.
   - Scales features to have a mean (\(\mu\)) of 0 and a standard deviation (\(\sigma\)) of 1.
   - The formula for standardization is:
     - [X_standardized = {X - mu}{sigma}]
   - It centers the data around the mean and scales it by the standard deviation.
   - Useful for algorithms that assume normally distributed data, like many statistical methods and PCA (Principal Component Analysis).

3. **Robust Scaling:**
   - Similar to min-max scaling but more resistant to outliers.
   - It uses the interquartile range (IQR) instead of the minimum and maximum values.
   - The formula for robust scaling is:
     - \[X_scaled = \frac{X - Q_1}{Q_3 - Q_1}\]
     - Where \(Q_1\) is the 25th percentile, and \(Q_3\) is the 75th percentile of the data.
   - Useful when dealing with data containing outliers.

4. **Log Transformation:**
   - Applies a logarithmic function to the data.
   - Useful when the data has a skewed distribution and you want to make it more symmetrical.
   - Often used with data containing exponential growth or decay.

5. **Box-Cox Transformation:**
   - A family of power transformations that includes the logarithm as a special case.
   - It optimizes the transformation based on the data to make it as close to a normal distribution as possible.
   - Applicable to data with varying degrees of non-linearity.

6. **Quantile Transformation:**
   - Maps data to a uniform or normal distribution.
   - Each feature's values are replaced with their respective quantiles.
   - Useful for converting data to a form suitable for machine learning models that make distributional assumptions.

7. **Unit Vector Transformation (Vector Normalization):**
   - Scales each feature to have a Euclidean length (L2 norm) of 1.
   - Useful for algorithms sensitive to the magnitude of vectors, like K-nearest neighbors.

The choice of data normalization method depends on the specific characteristics of your data and the requirements of your machine learning algorithm. It's important to consider the distribution of your data, the presence of outliers, and the assumptions of the models you're using when selecting a normalization technique.
"""
"""
The Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC) are important tools for evaluating the performance of binary classification models. They help assess how well a model can distinguish between two classes (e.g., positive and negative) by analyzing the trade-off between sensitivity and specificity.

Here's a detailed explanation of the ROC AUC curve:

1. **Binary Classification Background**: In binary classification, you typically have two classes, often referred to as "positive" and "negative." The goal is to develop a model that can classify data points into one of these two categories.

2. **ROC Curve (Receiver Operating Characteristic Curve)**: The ROC curve is a graphical representation of the performance of a binary classifier at various classification thresholds. It plots two important metrics:
   
   - **Sensitivity (True Positive Rate, TPR)**: Sensitivity is the fraction of actual positive cases that the model correctly identifies as positive. It is calculated as TPR = TP / (TP + FN), where TP is True Positives and FN is False Negatives. Sensitivity measures the model's ability to correctly detect positive cases.

   - **1 - Specificity (False Positive Rate, FPR)**: Specificity is the fraction of actual negative cases that the model incorrectly classifies as positive. It is calculated as FPR = FP / (FP + TN), where FP is False Positives and TN is True Negatives. 1 - Specificity measures the model's ability to avoid classifying negative cases as positive.

3. **Threshold Variation**: The ROC curve is constructed by varying the classification threshold used by the model. At each threshold, it calculates the TPR and FPR, resulting in a point on the ROC curve. By varying the threshold, you can observe how the model's performance changes.

4. **AUC (Area Under the Curve)**: The AUC is a single scalar value that summarizes the entire ROC curve. It quantifies the overall performance of a binary classification model. The AUC represents the probability that the classifier will rank a randomly chosen positive instance higher than a randomly chosen negative instance. An AUC of 0.5 indicates that the model performs no better than random guessing, while an AUC of 1.0 signifies a perfect classifier.

   - An AUC of 0.5: The model has no discriminative ability, equivalent to random guessing.
   - An AUC less than 0.5: The model's performance is worse than random guessing.
   - An AUC between 0.5 and 1.0: The model is better than random guessing, with a higher AUC indicating better performance.

5. **Interpreting ROC AUC**: A model with a higher AUC is generally better at distinguishing between the two classes. It means that as you vary the classification threshold, the model is more likely to assign higher probabilities to true positives than to true negatives.

6. **Choosing the Threshold**: The ROC curve does not provide an optimal threshold directly. Depending on the specific use case, you may select a threshold that balances sensitivity and specificity or meets specific requirements.

In summary, the ROC AUC curve provides a visual and quantitative way to assess the performance of binary classification models by considering their ability to discriminate between positive and negative cases across various classification thresholds. A higher AUC indicates better model performance.
"""
"""
K-Nearest Neighbors (KNN) is a simple and versatile classification and regression machine learning algorithm. It is used for both classification and regression tasks. The key idea behind KNN is to predict the class or value of a data point based on the majority class or average value of its neighboring data points.

Here's how KNN works:

1. **Training Phase:** In the training phase, KNN doesn't learn a specific model. Instead, it memorizes the entire training dataset.

2. **Prediction Phase (Classification):** When you want to classify a new data point, KNN looks at the K-nearest neighbors (data points with the most similar features) from the training dataset. It uses a distance metric (typically Euclidean distance, Manhattan distance, etc.) to measure the similarity between data points.

   - If it's a classification task (predicting a class label), KNN counts the number of neighbors in each class. The class with the majority of neighbors becomes the predicted class for the new data point.

3. **Prediction Phase (Regression):** In regression tasks, KNN predicts a continuous value. Instead of counting class labels, KNN calculates the average (or weighted average) of the target values of the K-nearest neighbors.

   - For example, if you want to predict a house's price, KNN will average the prices of the K-nearest neighboring houses.

4. **Choosing K:** The choice of the value K is crucial in KNN. It's typically an odd number to avoid ties, but the optimal K depends on your dataset and problem. A smaller K (e.g., 1 or 3) makes the model more sensitive to noise, while a larger K provides a smoother decision boundary.

KNN's strengths and weaknesses:

**Strengths:**
- KNN is simple and easy to implement.
- It can be used for both classification and regression.
- It doesn't assume a particular form for the decision boundary.
- It can be effective for multi-class classification problems.

**Weaknesses:**
- KNN can be computationally expensive, especially for large datasets.
- It's sensitive to the choice of the distance metric and the value of K.
- The curse of dimensionality: KNN becomes less effective as the number of dimensions (features) increases.
- It doesn't handle imbalanced datasets well, where one class has significantly more instances than the others.

KNN is often used for its simplicity and can serve as a baseline model for classification and regression tasks. However, to make the most of it, you need to carefully choose the right distance metric, K value, and handle issues like scaling features and handling missing data, if applicable.
"""
