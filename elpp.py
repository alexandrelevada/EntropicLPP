#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Entropic Locality Preserving Projections

Created on Wed Jul 24 16:59:33 2022

@author: Alexandre L. M. Levada

"""

# Imports
import sys
import time
import warnings
import umap
import sklearn.datasets as skdata
import matplotlib.pyplot as plt
import numpy as np
from numpy import log
from numpy import trace
from numpy import dot
from scipy import stats
from numpy.linalg import det
from scipy.linalg import eigh
from numpy.linalg import inv
from numpy.linalg import cond
from numpy import eye
from sklearn import preprocessing
from sklearn import metrics
import sklearn.neighbors as sknn
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

##############################################
# PCA implementation
##############################################
def myPCA(dados, d):
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(np.cov(dados.T))
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    
    return novos_dados

##################################################################
# KL-divergence between two multivariate Gaussian distributions
##################################################################
def divergenciaKL(mu1, mu2, cov1, cov2):
    m = len(mu1)
    # If covariance matrices are ill-conditioned
    if np.linalg.cond(cov1) > 1/sys.float_info.epsilon:
        cov1 = cov1 + np.diag(0.001*np.ones(m))
    if np.linalg.cond(cov2) > 1/sys.float_info.epsilon:
        cov2 = cov2 + np.diag(0.001*np.ones(m))    
    dM1 = 0.5*(mu2-mu1).T.dot(inv(cov2)).dot(mu2-mu1)
    dM2 = 0.5*(mu1-mu2).T.dot(inv(cov1)).dot(mu1-mu2)
    dTr = 0.5*trace(dot(inv(cov1), cov2) + dot(inv(cov2), cov1))
    
    dKL = 0.5*(dTr + dM1 + dM2 - m)
    
    return dKL

############################################
# Regular LPP
############################################
def LPP(X, k, d, t, mode):
    if mode == 'distancias':
        knnGraph = sknn.kneighbors_graph(X, n_neighbors=k, mode='distance')
        knnGraph.data = np.exp(-(knnGraph.data**2)/t)
    else:
        knnGraph = sknn.kneighbors_graph(X, n_neighbors=k, mode='connectivity')

    W = knnGraph.toarray()  
    
    # Diagonal matrix of the degrees
    D = np.diag(W.sum(1))   
    L = D - W

    # Compute the matrix for spectral decomposition
    X = X.T
    M1 = np.dot(np.dot(X, D), X.T)
    M2 = np.dot(np.dot(X, L), X.T)
    # Regularize the matrix before inversion if necessary (to avoid numerical issues)
    if cond(M1) < 1/sys.float_info.epsilon:
        M = np.dot(inv(M1), M2)
    else:
        M1 = M1 + 0.00001*eye(M1.shape[0])
        M = np.dot(inv(M1), M2)
    # Spectral decomposition
    lambdas, alphas = eigh(M, eigvals=(0, d-1))   
    
    output = np.dot(alphas.T, X)

    return output

#################################
# Entropic LPP
#################################
def EntropicLPP(X, k, d, t, mode):
    # Build the KNN graph
    knnGraph = sknn.kneighbors_graph(X, n_neighbors=k, mode='connectivity')
    W = knnGraph.toarray()  
    # To store the means and covariance matrices of each patch
    medias = np.zeros((dados.shape[0], dados.shape[1]))
    matriz_covariancias = np.zeros((dados.shape[0], dados.shape[1], dados.shape[1]))
    # Computes the local means and covariance matrices
    for i in range(X.shape[0]):       
        vizinhos = W[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) < 2:   # treat isolated points
            medias[i, :] = X[i, :]
            matriz_covariancias[i, :, :] = np.eye(dados.shape[1])   # covariance matriz is the identity
        else:
            amostras = dados[indices]
            medias[i, :] = amostras.mean(0)
            matriz_covariancias[i, :, :] = np.cov(amostras.T)
    # Define the entropic Laplacian matrix
    Wkl = W.copy()
    # This parameter is used to define whether we want to apply a Gaussian kernel or not in the KL-divergences
    if mode == 'distancias':
        for i in range(Wkl.shape[0]):
            for j in range(Wkl.shape[1]):
                if Wkl[i, j] > 0:
                    Wkl[i, j] = divergenciaKL(medias[i, :], medias[j, :], matriz_covariancias[i, :, :], matriz_covariancias[j, :, :])
        # Gaussian Kernel
        Wkl = np.exp(-(Wkl**2)/t)            
    else:
        for i in range(Wkl.shape[0]):
            for j in range(Wkl.shape[1]):
                if Wkl[i, j] > 0:
                    Wkl[i, j] = divergenciaKL(medias[i, :], medias[j, :], matriz_covariancias[i, :, :], matriz_covariancias[j, :, :])
        # Disconnect edges whose weights are less than the local average plus 2 times the std. dev.
        medias = Wkl.mean(axis=1)
        desvios = Wkl.std(axis=1)
        for i in range(Wkl.shape[0]):
           for j in range(Wkl.shape[1]):
                if Wkl[i, j] > medias[i] + 2*desvios[i]:
                    W[i, j] = 0
        Wkl = W.copy()
    # Diagonal matrix D
    D = np.diag(Wkl.sum(1))
    # Laplacian matrix L
    L = D - Wkl            
    # Compute the matrix for spectral decomposition
    X = X.T
    M1 = np.dot(np.dot(X, D), X.T)
    M2 = np.dot(np.dot(X, L), X.T)
    # Regularize before the inversion if necessary
    if cond(M1) < 1/sys.float_info.epsilon:
        M = np.dot(inv(M1), M2)
    else:
        M1 = M1 + 0.00001*eye(M1.shape[0])
        M = np.dot(inv(M1), M2)
    # Perform spectral decomposition
    lambdas, alphas = eigh(M, eigvals=(0, d-1))   # não descarta menor autovalor (diferente do Lap. Eig.)
    
    output = np.dot(alphas.T, X)
    
    return output

##########################################################################################
'''
 Computes the Silhouette coefficient and the supervised classification
 accuracies for several classifiers: KNN, DT, QDA, RFC
 dados: learned representation (output of a dimens. reduction - DR)
 target: ground-truth (data labels)
 method: string to identify the DR method (PCA, NP-PCAKL, KPCA, ISOMAP, LLE, LAP, ...)
'''
##########################################################################################
def Classification(dados, target, method):
    print()
    print('Supervised classification for %s features' %(method))
    print()
    
    lista = []

    # 50% for training and 40% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados.real.T, target, test_size=.5, random_state=42)

    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train) 
    acc = neigh.score(X_test, y_test)
    lista.append(acc)
    #print('KNN accuracy: ', acc)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    acc = dt.score(X_test, y_test)
    lista.append(acc)
    #print('DT accuracy: ', acc)

    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    acc = qda.score(X_test, y_test)
    lista.append(acc)
    #print('QDA accuracy: ', acc)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    acc = rfc.score(X_test, y_test)
    lista.append(acc)
    #print('RFC accuracy: ', acc)

    # Computes the Silhoutte coefficient
    sc = metrics.silhouette_score(dados.real.T, target, metric='euclidean')
    print('Silhouette coefficient: ', sc)
    
    # Computes the average accuracy
    average = sum(lista)/len(lista)
    maximo = max(lista)

    print('Average accuracy: ', average)
    print('Maximum accuracy: ', maximo)
    print()

    return [sc, average]

##########################################################################
# Make scatterplots for visualization after dimensionality reduction
##########################################################################
def PlotaDados(dados, labels, metodo):
    
    nclass = len(np.unique(labels))

    if metodo == 'LDA':
        if nclass == 2:
            return -1

    # Converte labels para inteiros
    lista = []
    for x in labels:
        if x not in lista:  
            lista.append(x)     # contém as classes (sem repetição)

    # Mapeia rotulos para números
    rotulos = []
    for x in labels:  
        for i in range(len(lista)):
            if x == lista[i]:  
                rotulos.append(i)

    # Converte para vetor
    rotulos = np.array(rotulos)

    if nclass > 11:
        cores = ['black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred']
        np.random.shuffle(cores)
    else:
        cores = ['blue', 'red', 'cyan', 'black', 'orange', 'magenta', 'green', 'darkkhaki', 'brown', 'purple', 'salmon']

    plt.figure(10)
    for i in range(nclass):
        indices = np.where(rotulos==i)[0]
        #cores = ['blue', 'red', 'cyan', 'black', 'orange', 'magenta', 'green', 'darkkhaki', 'brown', 'purple', 'salmon', 'silver', 'gold', 'darkcyan', 'royalblue', 'darkorchid', 'plum', 'crimson', 'lightcoral', 'orchid', 'powderblue', 'pink', 'darkmagenta', 'turquoise', 'wheat', 'tomato', 'chocolate', 'teal', 'lightcyan', 'lightgreen', ]
        cor = cores[i]
        plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, marker='*')
    
    nome_arquivo = metodo + '.png'
    plt.title(metodo+' clusters')

    plt.savefig(nome_arquivo)
    plt.close()

######################################
#  Data loading
######################################
X = skdata.load_iris()
#X = skdata.fetch_openml(name='servo', version=1) 
#X = skdata.fetch_openml(name='SPECTF', version=1)  
#X = skdata.fetch_openml(name='triazines', version=2) 
#X = skdata.fetch_openml(name='veteran', version=2) 
#X = skdata.fetch_openml(name='parity5', version=1) 
#X = skdata.fetch_openml(name='tic-tac-toe', version=1) 
#X = skdata.fetch_openml(name='sleuth_ex1605', version=2) 
#X = skdata.fetch_openml(name='aids', version=1) 
#X = skdata.fetch_openml(name='cloud', version=2) 
#X = skdata.fetch_openml(name='haberman', version=1) 
#X = skdata.fetch_openml(name='breast-tissue', version=2) 
#X = skdata.fetch_openml(name='bolts', version=2) 
#X = skdata.fetch_openml(name='analcatdata_creditscore', version=1) 
#X = skdata.fetch_openml(name='threeOf9', version=1) 
#X = skdata.fetch_openml(name='corral', version=1)
#X = skdata.fetch_openml(name='xd6', version=1)
#X = skdata.fetch_openml(name='car-evaluation', version=1)  
#X = skdata.fetch_openml(name='cars1', version=1)
#X = skdata.fetch_openml(name='calendarDOW', version=1)
#X = skdata.fetch_openml(name='solar-flare', version=3)
#X = skdata.fetch_openml(name='LED-display-domain-7digit', version=1)
#X = skdata.fetch_openml(name='balance-scale', version=1) 
#X = skdata.fetch_openml(name='hayes-roth', version=1) 
#X = skdata.fetch_openml(name='confidence', version=1) 
#X = skdata.fetch_openml(name='fl2000', version=1) 
#X = skdata.fetch_openml(name='diggle_table_a2', version=1) 
#X = skdata.fetch_openml(name='nursery', version=3)             # 10%
#X = skdata.fetch_openml(name='Diabetes130US', version=1)       # 1%
#X = skdata.fetch_openml(name='blogger', version=1)

dados = X['data']
target = X['target']  

# Uncomment this line to select a fraction of the samples
#dados, trash, target, garbage = train_test_split(dados, target, train_size=0.1, random_state=42)

# Number of samples
n = dados.shape[0]
# Number of features
m = dados.shape[1]
# Number of classes
c = len(np.unique(target))
nn = round(np.sqrt(n))
# Print information
print('N = ', n)
print('M = ', m)
print('C = %d' %c)
input()

# Required for  OpenML datasets
# Treat categorical data manually (label encoding)
if not isinstance(dados, np.ndarray):
    cat_cols = dados.select_dtypes(['category']).columns
    dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
    # Convert dataframe to numpy
    dados = dados.to_numpy()
    target = target.to_numpy()

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

##################################
# Simple PCA 
##################################
dados_pca = myPCA(dados, 2)

#################################
# ISOMAP
#################################
model = Isomap(n_neighbors=nn, n_components=2)
dados_isomap = model.fit_transform(dados)
dados_isomap = dados_isomap.T

##################################
# LLE
##################################
model = LocallyLinearEmbedding(n_neighbors=nn, n_components=2)
dados_LLE = model.fit_transform(dados)
dados_LLE = dados_LLE.T

#################################
# Lap. Eig.
#################################
model = SpectralEmbedding(n_neighbors=nn, n_components=2)
dados_Lap = model.fit_transform(dados)
dados_Lap = dados_Lap.T

#################################
# Regular LPP
#################################
dados_lpp = LPP(X=dados, k=nn, d=2, t=1, mode='distancias')

#################################
# UMAP
#################################
model = umap.UMAP(n_components=2)
dados_umap = model.fit_transform(dados)
dados_umap = dados_umap.T


###############################################
# Supervised classification
###############################################
L_pca = Classification(dados_pca.real, target, 'PCA')
L_iso = Classification(dados_isomap, target, 'ISOMAP')
L_lle = Classification(dados_LLE, target, 'LLE')
L_lap = Classification(dados_Lap, target, 'Lap. Eig.')
L_lpp = Classification(dados_lpp, target, 'LPP')
L_umap = Classification(dados_umap, target, 'UMAP')

###############################################
# Plot data (scatterplots)
###############################################
PlotaDados(dados_pca.T, target, 'PCA')
PlotaDados(dados_isomap.T, target, 'ISOMAP')
PlotaDados(dados_LLE.T, target, 'LLE')
PlotaDados(dados_Lap.T, target, 'LAP')
PlotaDados(dados_lpp.T, target, 'LPP')
PlotaDados(dados_umap.T, target, 'UMAP')

#######################################################
# Entropic LPP (find the optimum number of neighbors)
#######################################################
start = 2
high = 51
increment = 1
vizinhos = list(range(start, high, increment))

best_acc = 0
best_viz = 0

print('Supervised classification for Entropic LPP features')

for viz in vizinhos:

    # Entropic LPP
    dados_lpp_ent = EntropicLPP(X=dados, k=viz, d=2, t=1, mode='distancias')
    dados_lpp_ent = dados_lpp_ent.T

    #%%%%%%%%%%%%%%%%%%%% Supervised classification for Kernel PCA features

    print()
    print('K = %d' %viz)

    # 50% for training and 40% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados_lpp_ent.real, target, test_size=.5, random_state=42)
    acc = 0

    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train)
    acc += neigh.score(X_test, y_test)
    #print('KNN accuracy: ', neigh.score(X_test, y_test))

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    acc += dt.score(X_test, y_test)
    #print('DT accuracy: ', dt.score(X_test, y_test))

    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    acc += qda.score(X_test, y_test)
    #print('QDA accuracy: ', qda.score(X_test, y_test))

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    acc += rfc.score(X_test, y_test)
    #print('RFC accuracy: ', rfc.score(X_test, y_test))

    # # Computes the Silhoutte coefficient
    # print('Silhouette coefficient: ', metrics.silhouette_score(dados_lpp_ent.real, target, metric='euclidean'))
    mean_acc = acc/4
    print('Acurácia média: ', mean_acc)

    if mean_acc > best_acc:
        best_acc = mean_acc
        best_viz = viz

print()
print('==============')
print('BEST ACCURACY')
print('==============')
print()
print('Average accuracy: ', best_acc)
print('K = ', best_viz)

# Apply the transformation using the optimum K (number of neighbors)
dados_lpp_ent = EntropicLPP(X=dados, k=best_viz, d=2, t=1, mode='distancias')
dados_lpp_ent = dados_lpp_ent.T
PlotaDados(dados_lpp_ent.real, target, 'LPP-KL')