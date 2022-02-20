import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

train_file = "mlgenetics_hw7_TRAIN.txt"
batch_file = "mlgenetics_hw7_TRAIN_batch.txt"

#reading data
print("reading data")
train_data = pd.read_csv(train_file,sep='\t',header=None)
batch_data = pd.read_csv(batch_file,sep='\t',header=None)
print("train data shape: ",train_data.shape)
print("batch data shape: ",batch_data.shape)

#normalize data to mean 0 variance 1
train_data_normalized = StandardScaler().fit_transform(train_data)

#compute PCA
pca = PCA()
principalComponents = pca.fit_transform(train_data_normalized)
print("principal components shape: ", principalComponents.shape)
#print(principalComponents)
print("proportion of variance explained by first 10 PC: ",sum(pca.explained_variance_ratio_[:10]))

#finding correlation with first 10 PC
corr = [pearsonr(batch_data[:][0],principalComponents[:,i])[0]**2 for i in range(10)]
#print(corr)
print("PC with max squared correlation: ",np.argmax(corr), max(corr))

#finding sparse 1000 sites based on absolution loading value
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
indices = sorted(np.argsort(abs(loadings[:,np.argmax(corr)]))[-1000:])

#setup sparse data and normalize
sparse_train_data = train_data.iloc[:,indices]
sparse_train_data_normalized = StandardScaler().fit_transform(sparse_train_data)
principalComponents = pca.fit_transform(sparse_train_data_normalized)

#finding correlation with first 10 sparse PC
sparse_corr = [pearsonr(batch_data[:][0],principalComponents[:,i])[0]**2 for i in range(10)]
print("sparse PC with max squared correlation: ",np.argmax(sparse_corr), max(sparse_corr))



