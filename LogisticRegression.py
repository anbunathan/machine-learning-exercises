from __future__ import print_function
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# Load the data, scale, encode
filepath = 'knn/Human_Activity_Recognition_Using_Smartphones_Data.csv'
data = pd.read_csv(filepath, sep=',')
print("data.dtypes.value_counts() = ",data.dtypes.value_counts())
print("data.dtypes.tail() = ",data.dtypes.tail())
print("data.iloc[:, :-1].min().value_counts() = ",data.iloc[:, :-1].min().value_counts())
print("data.iloc[:, :-1].max().value_counts() = ",data.iloc[:, :-1].max().value_counts())
print("data.Activity.value_counts() = ",data.Activity.value_counts())
le = LabelEncoder()
data['Activity'] = le.fit_transform(data.Activity)
data['Activity'].sample(5)

# Identify the variables that are most correlated
# Calculate the correlation values
feature_cols = data.columns[:-1]
corr_values = data[feature_cols].corr()
# Simplify by emptying all the data below the diagonal
tril_index = np.tril_indices_from(corr_values)
# Make the unused values NaNs
for coord in zip(*tril_index):
    corr_values.iloc[coord[0], coord[1]] = np.NaN
# Stack the data and convert to a data frame
corr_values = (corr_values
               .stack()
               .to_frame()
               .reset_index()
               .rename(columns={'level_0': 'feature1',
                                'level_1': 'feature2',
                                0: 'correlation'}))
# Get the absolute values for sorting
corr_values['abs_correlation'] = corr_values.correlation.abs()
sns.set_context('talk')
sns.set_style('white')
sns.set_palette('dark')
ax = corr_values.abs_correlation.hist(bins=50)
ax.set(xlabel='Absolute Correlation', ylabel='Frequency')
plt.show()
# The most highly correlated values
corr_values.sort_values('correlation', ascending=False).query('abs_correlation>0.8')

# Use StratifiedShuffleSplit to split the data into test and train
# Get the split indexes
strat_shuf_split = StratifiedShuffleSplit(n_splits=1,
                                          test_size=0.3,
                                          random_state=42)
train_idx, test_idx = next(strat_shuf_split.split(data[feature_cols], data.Activity))
# Create the dataframes
X_train = data.loc[train_idx, feature_cols]
y_train = data.loc[train_idx, 'Activity']
X_test  = data.loc[test_idx, feature_cols]
y_test  = data.loc[test_idx, 'Activity']
print("y_train.value_counts = ",y_train.value_counts(normalize=True))
print("y_test.value_counts = ",y_test.value_counts(normalize=True))

# Fit a logistic regression model, without/with L1 and L2 regularization
# Standard logistic regression
lr = LogisticRegression().fit(X_train, y_train)
# L1 regularized logistic regression
lr_l1 = LogisticRegressionCV(Cs=10, cv=4, penalty='l1', solver='liblinear').fit(X_train, y_train)
# L2 regularized logistic regression
lr_l2 = LogisticRegressionCV(Cs=10, cv=4, penalty='l2').fit(X_train, y_train)

# Compare the magnitudes of the coefficients for LR, L1, L2 models
# Combine all the coefficients into a dataframe
coefficients = list()
coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]
for lab,mod in zip(coeff_labels, coeff_models):
    coeffs = mod.coef_
    coeff_label = pd.MultiIndex(levels=[[lab], [0,1,2,3,4,5]],
                                 labels=[[0,0,0,0,0,0], [0,1,2,3,4,5]])
    coefficients.append(pd.DataFrame(coeffs.T, columns=coeff_label))
coefficients = pd.concat(coefficients, axis=1)
coefficients.sample(10)

# Prepare six separate plots for each of the multi-class coefficients
fig, axList = plt.subplots(nrows=3, ncols=2)
axList = axList.flatten()
fig.set_size_inches(10, 10)
for ax in enumerate(axList):
    loc = ax[0]
    ax = ax[1]
    data = coefficients.xs(loc, level=1, axis=1)
    data.plot(marker='o', ls='', ms=2.0, ax=ax, legend=False)
    if ax is axList[0]:
        ax.legend(loc=4)
    ax.set(title='Coefficient Set ' + str(loc))
plt.tight_layout()
plt.show()

# Predict the class and the probability for each
y_pred = list()
y_prob = list()
coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]
for lab, mod in zip(coeff_labels, coeff_models):
    y_pred.append(pd.Series(mod.predict(X_test), name=lab))
    y_prob.append(pd.Series(mod.predict_proba(X_test).max(axis=1), name=lab))
y_pred = pd.concat(y_pred, axis=1)
y_prob = pd.concat(y_prob, axis=1)
print("y_pred.head() = ",y_pred.head())

# For each model, calculate the error metrics such as accuracy, precision, recall, fscore, confusion matrix
metrics = list()
cm = dict()
for lab in coeff_labels:
    # Preciision, recall, f-score from the multi-class support function
    precision, recall, fscore, _ = score(y_test, y_pred[lab], average='weighted')
    # The usual way to calculate accuracy
    accuracy = accuracy_score(y_test, y_pred[lab])
    # ROC-AUC scores can be calculated by binarizing the data
    auc = roc_auc_score(label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5]),
                        label_binarize(y_pred[lab], classes=[0, 1, 2, 3, 4, 5]),
                        average='weighted')
    # Last, the confusion matrix
    cm[lab] = confusion_matrix(y_test, y_pred[lab])
    metrics.append(pd.Series({'precision': precision, 'recall': recall,
                              'fscore': fscore, 'accuracy': accuracy,
                              'auc': auc},
                             name=lab))
metrics = pd.concat(metrics, axis=1)
print("metrics = ",metrics)

# Display or plot the confusion matrix for each model
fig, axList = plt.subplots(nrows=2, ncols=2)
axList = axList.flatten()
fig.set_size_inches(12, 10)
axList[-1].axis('off')
for ax, lab in zip(axList[:-1], coeff_labels):
    sns.heatmap(cm[lab], ax=ax, annot=True, fmt='d');
    ax.set(title=lab)
plt.tight_layout()
plt.show()

