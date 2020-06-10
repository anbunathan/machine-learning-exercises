from __future__ import print_function
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data and prepare
filepath = 'knn/Wholesale_Customers_Data.csv'
data = pd.read_csv(filepath, sep=',')
print("data.shape = ",data.shape)
print("data.head() = ",data.head())
data = data.drop(['Channel', 'Region'], axis=1)
print("data.dtypes = ",data.dtypes)
# Convert to floats
for col in data.columns:
    data[col] = data[col].astype(np.float)
# Preserve the original
data_orig = data.copy()

# Examine Correlation, skew. Perform transformations and scaling. Plot pairwise correlation plots
corr_mat = data.corr()
# Strip the diagonal for future examination
for x in range(corr_mat.shape[0]):
    corr_mat.iloc[x, x] = 0.0
print("corr_mat = ",corr_mat)
# Find most strongly correlated variable
print("corr_mat.abs().idxmax() = ",corr_mat.abs().idxmax())

log_columns = data.skew().sort_values(ascending=False)
log_columns = log_columns.loc[log_columns > 0.75]
print("log_columns = ",log_columns)

# The log transformations
for col in log_columns.index:
    data[col] = np.log1p(data[col])
print("data.head() after log transformations = ",data.head())

# use MinMaxScaler to scale the data
mms = MinMaxScaler()
for col in data.columns:
    data[col] = mms.fit_transform(data[[col]]).squeeze()
print("data.head() after MinMaxScaler = ",data.head())

# Plot pairwise plot
sns.set_context('notebook')
sns.set_palette('dark')
sns.set_style('white')
sns.pairplot(data)
plt.show()

# Create a pipeline to pre-process the data and compare with previous result
# The custom NumPy log transformer
log_transformer = FunctionTransformer(np.log1p)
# The pipeline
estimators = [('log1p', log_transformer), ('minmaxscale', MinMaxScaler())]
pipeline = Pipeline(estimators)
# Convert the original data
data_pipe = pipeline.fit_transform(data_orig)
print("check two arrays (pipeline, no pipeline) are equal = ",np.allclose(data_pipe, data))

# Perform PCA with n_components ranging from 1 to 5. Find and plot the explained variance and feature importances
pca_list = list()
feature_weight_list = list()
# Fit a range of PCA models
for n in range(1, 6):
    # Create and fit the model
    PCAmod = PCA(n_components=n)
    PCAmod.fit(data)
    # Store the model and variance
    pca_list.append(pd.Series({'n': n, 'model': PCAmod,
                               'var': PCAmod.explained_variance_ratio_.sum()}))
    # Calculate and store feature importances
    abs_feature_values = np.abs(PCAmod.components_).sum(axis=0)
    feature_weight_list.append(pd.DataFrame({'n': n,
                                             'features': data.columns,
                                             'values': abs_feature_values / abs_feature_values.sum()}))
pca_df = pd.concat(pca_list, axis=1).T.set_index('n')
print("pca_df with feature importances = ",pca_df)
# Create a table of feature importances for each data column
features_df = (pd.concat(feature_weight_list)
               .pivot(index='n', columns='features', values='values'))
print("features_df with feature importances = ",features_df)

# Create a plot of explained variances
sns.set_context('talk')
ax = pca_df['var'].plot(kind='bar')
ax.set(xlabel='Number of dimensions',
       ylabel='Percent explained variance',
       title='Explained Variance vs Dimensions')
plt.show()

# Create plot of feature importances
ax = features_df.plot(kind='bar')
ax.set(xlabel='Number of dimensions',
       ylabel='Relative importance',
       title='Feature importance vs Dimensions')
plt.show()

# Define a function for scorer
# Custom scorer--use negative rmse of inverse transform
def scorer(pcamodel, X, y=None):
    try:
        X_val = X.values
    except:
        X_val = X
    # Calculate and inverse transform the data
    data_inv = pcamodel.fit(X_val).transform(X_val)
    data_inv = pcamodel.inverse_transform(data_inv)
    # The error calculation
    mse = mean_squared_error(data_inv.ravel(), X_val.ravel())
    # Larger values are better for scorers, so take negative value
    return -1.0 * mse

# Define a function for GridSearchCVforPCA
def GridSearchCVforPCA(param_grid):
    if __name__ == '__main__':
        # The grid search
        kernelPCA = GridSearchCV(KernelPCA(kernel='rbf', fit_inverse_transform=True),
                             param_grid=param_grid,
                             scoring=scorer,
                             n_jobs=-1)
        return kernelPCA

# Fit a KernelPCA model with kernel='rbf'. use GridSearchCV to tune the parameters
if __name__ == '__main__':
    param_grid = {'gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
                  'n_components': [2, 3, 4]}
    kernelPCA = GridSearchCVforPCA(param_grid)
    kernelPCA = kernelPCA.fit(data)
    print("kernelPCA.best_estimator_ = ",kernelPCA.best_estimator_)

