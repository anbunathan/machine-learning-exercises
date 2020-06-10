from __future__ import print_function
import os
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data
# Import the data using the file path
filepath = 'knn/Orange_Telecom_Churn_Data.csv'
data = pd.read_csv(filepath)
print("data = ",data.head(1).T)
# Remove extraneous columns
data.drop(['state', 'area_code', 'phone_number'], axis=1, inplace=True)
print("data.columns = ",data.columns)

# Scale the data
lb = LabelBinarizer()
for col in ['intl_plan', 'voice_mail_plan', 'churned']:
    data[col] = lb.fit_transform(data[col])
warnings.filterwarnings('ignore', module='sklearn')
msc = MinMaxScaler()
data = pd.DataFrame(msc.fit_transform(data),  # this is an np.array, not a dataframe.
                    columns=data.columns)

# Separate the feature columns (everything except churned) from the label (churned)
# Get a list of all the columns that don't contain the label
x_cols = [x for x in data.columns if x != 'churned']
# Split the data into two dataframes
X_data = data[x_cols]
y_data = data['churned']

# Fit a K-nearest neighbors model with a value of k=3 to this data and predict the outcome on the same data
knn = KNeighborsClassifier(n_neighbors=3)
knn = knn.fit(X_data, y_data)
y_pred = knn.predict(X_data)

# Define a function to calculate the % of values that were correctly predicted
def accuracy(real, predict):
    return sum(y_data == y_pred) / float(real.shape[0])

# calculate the accuracy of this K-nearest neighbors model on the data
print("accuracy with n_neighbors set to 3 = ",accuracy(y_data, y_pred))

# Fit the K-nearest neighbors model again with n_neighbors=3 but this time use distance for the weights
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn = knn.fit(X_data, y_data)
y_pred = knn.predict(X_data)
print("accuracy with n_neighbors set to 5 and weights set to distance = ",accuracy(y_data, y_pred))

# Fit another K-nearest neighbors model with uniform weights but set the power parameter for the Minkowski distance metric to be 1 (p=1) i.e. Manhattan Distance
knn = KNeighborsClassifier(n_neighbors=5, p=1)
knn = knn.fit(X_data, y_data)
y_pred = knn.predict(X_data)
print("accuracy with n_neighbors set to 5 and p set to 1 = ",accuracy(y_data, y_pred))

# Fit the K-nearest neighbors model with different values of k and plot
score_list = list()
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn = knn.fit(X_data, y_data)
    y_pred = knn.predict(X_data)
    score = accuracy(y_data, y_pred)
    score_list.append((k, score))
score_df = pd.DataFrame(score_list, columns=['k', 'accuracy'])
print("score_df = ",score_df)

#Plot the accuracy
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('dark')
ax = score_df.set_index('k').plot()
ax.set(xlabel='k', ylabel='accuracy')
ax.set_xticks(range(1, 21))
plt.show()