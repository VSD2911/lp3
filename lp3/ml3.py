import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import io

"""## Read the Dataset"""

df=pd.read_csv("Churn_Modelling.csv")
df.head()

"""## 2. Drop the Columns which are unique for all users"""

df=df.drop(['RowNumber','CustomerId','Surname'],axis=1)
df.head()

df.isna().any()
df.isna().sum()

"""## BiVariate Analysis


"""

print(df.shape)
df.info()

df.describe()

"""#### Before performing Bivariate analysis, Lets bring all the features to the same range"""

## Scale the data
scaler=StandardScaler()
## Extract only the Numerical Columns to perform Bivariate Analysis
subset=df.drop(['Geography','Gender','HasCrCard','IsActiveMember'],axis=1)
scaled=scaler.fit_transform(subset)
scaled_df=pd.DataFrame(scaled,columns=subset.columns)
sns.pairplot(scaled_df,diag_kind='kde')

sns.heatmap(scaled_df.corr(),annot=True,cmap='rainbow')

"""### From the above plots, We can see that there is no significant Linear relationship between the features"""

## Categorical Features vs Target Variable
sns.countplot(x='Geography',data=df,hue='Exited')
plt.show()
sns.countplot(x='Gender',data=df,hue='Exited')
plt.show()
sns.countplot(x='HasCrCard',data=df,hue='Exited')
plt.show()
sns.countplot(x='IsActiveMember',data=df,hue='Exited')
plt.show()

"""### Analysing the Numerical Features relationship with the Target variable. Here 'Exited' is the Target Feature."""

subset = subset.drop('Exited', axis=1)
for i in subset.columns:
  sns.boxplot(x=df['Exited'], y=df[i], hue=df['Gender'])
  plt.show()

"""## Insights from Bivariate Plots


1. The Avg Credit Score seem to be almost the same for Active and Churned customers
2. Young People seem to stick to the bank compared to older people
3. The Average Bank Balance is high for Churned Customers
4. The churning rate is high with German Customers
5. The Churning rate is high among the Non-Active Members

### 4. Distinguish the Target and Feature Set and divide the dataset into Training and Test sets
"""

X=df.drop('Exited',axis=1)
y=df.pop('Exited')

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=5)
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.10,random_state=5)
print("X_train size is {}".format(X_train.shape[0]))
print("X_val size is {}".format(X_val.shape[0]))
print("X_test size is {}".format(X_test.shape[0]))

## Standardising the train, Val and Test data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
num_cols=['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']
num_subset=scaler.fit_transform(X_train[num_cols])
X_train_num_df=pd.DataFrame(num_subset,columns=num_cols)
X_train_num_df['Geography']=list(X_train['Geography'])
X_train_num_df['Gender']=list(X_train['Gender'])
X_train_num_df['HasCrCard']=list(X_train['HasCrCard'])
X_train_num_df['IsActiveMember']=list(X_train['IsActiveMember'])
X_train_num_df.head()
## Standardise the Validation data
num_subset=scaler.fit_transform(X_val[num_cols])
X_val_num_df=pd.DataFrame(num_subset,columns=num_cols)
X_val_num_df['Geography']=list(X_val['Geography'])
X_val_num_df['Gender']=list(X_val['Gender'])
X_val_num_df['HasCrCard']=list(X_val['HasCrCard'])
X_val_num_df['IsActiveMember']=list(X_val['IsActiveMember'])
## Standardise the Test data
num_subset=scaler.fit_transform(X_test[num_cols])
X_test_num_df=pd.DataFrame(num_subset,columns=num_cols)
X_test_num_df['Geography']=list(X_test['Geography'])
X_test_num_df['Gender']=list(X_test['Gender'])
X_test_num_df['HasCrCard']=list(X_test['HasCrCard'])
X_test_num_df['IsActiveMember']=list(X_test['IsActiveMember'])

## Convert the categorical features to numerical
X_train_num_df=pd.get_dummies(X_train_num_df,columns=['Geography','Gender'])
X_test_num_df=pd.get_dummies(X_test_num_df,columns=['Geography','Gender'])
X_val_num_df=pd.get_dummies(X_val_num_df,columns=['Geography','Gender'])
X_train_num_df.head()

"""### Initialise and build the Model"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(7,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

import tensorflow as tf
optimizer=tf.keras.optimizers.Adam(0.01)
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

model.fit(X_train_num_df,y_train,epochs=100,batch_size=10,verbose=1)

"""## Predict the Results using 0.5 threshold

"""

y_pred_val=model.predict(X_val_num_df)
y_pred_val[y_pred_val>0.5]=1
y_pred_val[y_pred_val <0.5]=0

y_pred_val=y_pred_val.tolist()
X_compare_val=X_val.copy()
X_compare_val['y_actual']=y_val
X_compare_val['y_pred']=y_pred_val
X_compare_val.head(10)

"""## Confusion Matrix of the Validation set"""

from sklearn.metrics import confusion_matrix
cm_val=confusion_matrix(y_val,y_pred_val)
cm_val

# Extract TP, TN, FP, FN from the confusion matrix
TP = cm_val[1, 1]  # True Positives
TN = cm_val[0, 0]  # True Negatives
FP = cm_val[0, 1]  # False Positives
FN = cm_val[1, 0]  # False Negatives

# Calculate accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

print("Accuracy:", accuracy)

loss1,accuracy1=model.evaluate(X_train_num_df,y_train,verbose=False)
loss2,accuracy2=model.evaluate(X_val_num_df,y_val,verbose=False)
print("Train Loss {}".format(loss1))
print("Train Accuracy {}".format(accuracy1))
print("Val Loss {}".format(loss2))
print("Val Accuracy {}".format(accuracy2))

"""### Since our Training Accuracy and Validation Accuracy are pretty close, we can conclude that our model generalises well. So, lets apply the model on the Test set and make predictions and evaluate the model against the Test."""

from sklearn import metrics
y_pred_test=model.predict(X_test_num_df)
y_pred_test[y_pred_test>0.5]=1
y_pred_test[y_pred_test <0.5]=0
cm_test=metrics.confusion_matrix(y_test,y_pred_test)
cm_test
print("Test Confusion Matrix")

cm_test

loss3,accuracy3=model.evaluate(X_test_num_df,y_test,verbose=False)
print("Test Accuracy is {}".format(accuracy3))
print("Test loss is {}".format(loss3))





"""
A neural network is a machine learning model inspired by the structure and function of the human brain. It consists of interconnected nodes (artificial neurons) organized in layers to process and learn patterns from data. Neural networks are a fundamental component of deep learning, capable of solving complex tasks, such as image and speech recognition, natural language processing, and much more. Here's an explanation of key components and concepts related to neural networks:

1. **Neurons (Nodes):**
   - Neurons are the basic building blocks of a neural network. They receive input, apply a transformation, and produce an output.

2. **Layers:**
   - Neural networks consist of layers, including an input layer, one or more hidden layers, and an output layer. Input and output layers serve for data ingestion and output prediction, respectively, while hidden layers process and transform information.

3. **Weights and Biases:**
   - Each connection between neurons has an associated weight. The weighted sum of inputs plus a bias is passed through an activation function to produce an output for the next layer.

4. **Activation Functions:**
   - Activation functions introduce non-linearity into the model. Common activation functions include ReLU, sigmoid, and tanh, and they determine if a neuron should "fire" or not.

5. **Feedforward and Backpropagation:**
   - Feedforward is the process of passing data through the network to make predictions. During training, backpropagation is used to adjust weights and biases to minimize the difference between predicted and actual outputs.

6. **Loss Function:**
   - A loss function measures how well the network's predictions match the actual target values. Training aims to minimize this loss.

7. **Optimization Algorithms:**
   - Optimization algorithms like stochastic gradient descent (SGD), Adam, and RMSprop are used to update weights and biases during training to minimize the loss function.

8. **Deep Learning:**
   - Neural networks with multiple hidden layers are often referred to as deep neural networks (or deep learning models). These models can learn hierarchical features from data.

9. **Overfitting:**
   - Overfitting occurs when a neural network learns the training data too well and performs poorly on unseen data. Regularization techniques are used to combat overfitting.

10. **Convolutional Neural Networks (CNNs):**
    - CNNs are specialized neural networks designed for processing grid-like data, such as images and videos. They employ convolutional and pooling layers to capture local patterns.

11. **Recurrent Neural Networks (RNNs):**
    - RNNs are suitable for sequence data. They maintain a hidden state that captures temporal dependencies, making them suitable for tasks like natural language processing and time series prediction.

12. **Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU):**
    - LSTM and GRU are specialized RNN architectures that address the vanishing gradient problem and better capture long-range dependencies.

Neural networks have demonstrated remarkable performance in various fields, enabling breakthroughs in computer vision, speech recognition, natural language understanding, and more. Their adaptability and ability to learn intricate patterns from data have made them a vital part of modern machine learning and artificial intelligence.
"""
"""
Activation functions are an essential component of neural networks and are used to introduce non-linearity into the model. They help neural networks learn complex patterns and relationships in data. Here are some common activation functions used in neural networks:

1. **Sigmoid (Logistic) Function:**
   - Formula: σ(x) = 1 / (1 + e^(-x))
   - Range: (0, 1)
   - Use: Used in the output layer for binary classification problems. It squashes the input into the (0, 1) range, representing probabilities.

2. **Hyperbolic Tangent (Tanh) Function:**
   - Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
   - Range: (-1, 1)
   - Use: Commonly used in hidden layers of neural networks. It maps input values to the (-1, 1) range, which helps mitigate the vanishing gradient problem.

3. **Rectified Linear Unit (ReLU):**
   - Formula: f(x) = max(0, x)
   - Range: [0, ∞)
   - Use: One of the most widely used activation functions. It is computationally efficient and helps the network learn sparse representations. However, it may suffer from the "dying ReLU" problem when neurons become inactive.

4. **Leaky ReLU:**
   - Formula: f(x) = x if x > 0, else f(x) = αx (α is a small positive constant)
   - Range: (-∞, ∞)
   - Use: An improvement over ReLU, Leaky ReLU allows a small gradient when the unit is not active, addressing the dying ReLU problem.

5. **Parametric ReLU (PReLU):**
   - Formula: f(x) = x if x > 0, else f(x) = αx (α is a learnable parameter)
   - Range: (-∞, ∞)
   - Use: Similar to Leaky ReLU, but the slope of the negative part is learned during training.

6. **Exponential Linear Unit (ELU):**
   - Formula: f(x) = x if x > 0, else f(x) = α(e^x - 1) (α is a positive constant)
   - Range: (-α, ∞)
   - Use: ELU has the benefits of ReLU and avoids the dying ReLU problem. It introduces smoothness for negative values.

7. **Scaled Exponential Linear Unit (SELU):**
   - Formula: f(x) = λx if x > 0, else f(x) = λα(e^x - 1) (α and λ are derived constants)
   - Range: (-αλ, ∞)
   - Use: SELU is designed to overcome the vanishing/exploding gradient problem and is particularly useful in deep neural networks.

8. **Softmax Function:**
   - Formula: σ(x)_i = e^(x_i) / Σ(e^(x_j)) for all i
   - Range: (0, 1)
   - Use: Typically used in the output layer for multiclass classification problems. It normalizes the input values to produce a probability distribution over multiple classes.

These activation functions serve different purposes and are chosen based on the problem at hand and the network's architecture. The choice of the right activation function can significantly impact the learning and performance of a neural network.
