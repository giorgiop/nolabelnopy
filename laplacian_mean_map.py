# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:53:38 2015

@author: giorgio
"""

from utils import *
#import warnings

#scientific libraries
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.utils.graph import graph_laplacian

#iterators
from itertools import product

#sklearn
from sklearn.cross_validation import _BaseKFold, KFold


def mean_operator(X, y):
    return np.dot(y, X) / X.shape[0]


#the index bag in X is used to assign the predictions y_pred to bags
def proportions_abs_err(X, pi, p, y_pred):

    err = 0    
    for j in range(pi.shape[0]):
        idx_bag = X.ix[j]
        err += p[j] * np.abs(pi[j] - np.mean(to_zero_one(y_pred[idx_bag])))
    return err
    

def logistic_loss(theta, X, mean_op, lam):
    m = X.shape[0]
    dot = np.dot(X, theta)
    ai = np.log(np.exp(dot) + np.exp(- dot))
    ai = ai if np.isfinite(ai).all else dot #prevent overflow, take the limit
    log_partition = sum(ai)
    mean_op_part = np.dot(theta, mean_op)
    reg = 0.5 * lam * np.dot(theta, theta)
    return log_partition / m - mean_op_part + reg
    

def logistic_loss_grad(theta, X, mean_op, lam):
    m = X.shape[0]
    dot = np.dot(X, theta)
    return np.sum(np.dot(np.tanh(dot), X)) / m - mean_op + lam * theta


def compute_bag_means(X):
    n = len(X.index.levels[0])
    return pd.DataFrame([np.mean(X.loc[i]) for i in range(n)])


def similarity_matrix(B, d):
    l = B.shape[0]
    A = np.zeros((l,l))
    for i in range(l):
        for j in range(i):
            A[i][j] = d(B.ix[i,:], B.ix[j,:])
    return A + np.transpose(A) #copy elems in the other triangular matrix
    
    
def laplacian_matrix(B, d, normed=False):
    A = similarity_matrix(B, d)
    #return np.diag(np.sum(A, axis=0)) - A
    return graph_laplacian(A, normed=normed)


class BaggedKFold(_BaseKFold):
    """Bagged-K-Folds cross validation iterator.
        Provides train/test indices to split data in train test sets. Split
        dataset into k consecutive folds (without shuffling).
        Each fold is then used a validation set once while the k - 1 remaining
        fold form the training set.
        In contrast with K fold crossvalidation, it get as input bag_id of and
        does train - test split consistently: the sampling is done uniformly
        in each bag separately.
        
        The purpose of BaggedKFold (similarly to StratifiedKFold) is to 
        (approximatively) preserve the proportions of samples for each class, 
        such that the proportions available for training are meaninful for each 
        train - test split.f
        
        Parameters
        ----------
        bag_id : int
            Total number of elements.
        n_folds : int, default=3
            Number of folds. Must be at least 2.
        shuffle : boolean, optional
            Whether to shuffle the data before splitting into batches.
        random_state : None, int or RandomState
            Pseudo-random number generator state used for random
            sampling. If None, use default numpy RNG for shuffling
    """
    
    def __init__(self, bag_id, n_folds=3, indices=None, shuffle=False,
                 random_state=None):
                         
        n = len(np.asarray(bag_id))
        super(BaggedKFold, self).__init__(
                n, n_folds, indices, shuffle, random_state)
            
        unique_bag, bag_inversed = np.unique(bag_id, return_inverse=True)
        bag_sizes = np.bincount(bag_inversed)

        min_bag_size = np.min(bag_sizes)
        if self.n_folds > min_bag_size:
            raise ValueError(("The least populated bag has only %d members, which is"
                            "less than n_folds=%d. Instances will be resampled."
                            .format(min_bag_size, self.n_folds)))
#       TO DOUBLE CHECK
#        if any(pi < self.n_folds/bag_sizes):
#            warnings.warn(("NOT GOOD FOR PROPORTIONS CONSISTENCY"
#                            .format(min_bag_size, self.n_folds)), Warning)
                        

        # don't want to use the same seed in each bag's shuffle
        if self.shuffle:
            rng = check_random_state(self.random_state)
        else:
            rng = self.random_state
            

        #create mapping from bag to bag_id vector (instances position)
        bag_to_x = np.array([[i for i, x in enumerate(bag_id) if x == b] for b in unique_bag])
        

        test_folds = np.zeros(self.n, dtype=np.int)
        for b in unique_bag:
            n_b = np.sum(bag_id == b)
            for i , (_, test_split) in enumerate(KFold(n_b, self.n_folds, 
                    shuffle=self.shuffle,random_state=rng)):
                        
                for j in test_split:
                    test_folds[bag_to_x[b][j]] = i

        self.test_folds = test_folds
        self.bag_id = bag_id
        
        
    def _iter_test_masks(self):
        for i in range(self.n_folds):
            yield self.test_folds == i
            

    def __repr__(self):
        return '%s.%s(bags id=%s, n_folds=%i, shuffle=%s, random_state=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.bag_id,
            self.n_folds,
            self.shuffle,
            self.random_state
        )


    def __len__(self):
        return self.n_folds


class LaplacianMeanMap:

    def __init__(self, Laplacian, B, gamma=1, weight='identity', epsilon=0.0):
        self.La = Laplacian
        self.B = B
        self.gamma = gamma
        self.weight = weight
        self.epsilon = epsilon
        self.lam = 0.0
        self.theta = 0.0

    def fit(self, X, pi):
        
        #problem dimension
        self.m = X.shape[0] #sample size
        self.n = len(pi) #numer of bags
        
        #estimate the bag frequencies \hat{p}_j
        p = [X.loc[i].shape[0] / self.m for i in range(self.n)]
        
        #set weight matrix
        D_w = np.identity(self.n)
        if self.weight == 'bag-size':
            D_w *= 1/np.array(p)
            
        #set the Laplacian
        L = np.identity(2*self.n) * self.epsilon + sp.linalg.block_diag(self.La, self.La)
        #print(L)        
        
        #set the proportions matrix
        Pi = np.transpose(np.hstack((np.diag(pi), np.diag(1-pi))))
        
        #Step1 - Least Squares
        Pi_dot_Dw = np.dot(Pi,D_w)
        B_pm = sp.linalg.solve((np.dot(Pi_dot_Dw, np.transpose(Pi)) + \
            self.gamma * L), np.dot(Pi_dot_Dw, self.B))
        
        #Step 2 - Mean Operator estimation
        b_plus = B_pm[:self.n]
        b_minus = B_pm[-self.n:]
        mean_op = np.sum(np.array([(p[j] * (pi[j] * b_plus[j] - \
            (1-pi[j]) * b_minus[j])) for j in range(self.n)]), axis=0)
        
        #Step 3 - Logistic regression
        theta_0 = np.empty(X.shape[1])
        theta_0.fill(0.0001)        

        res = sp.optimize.minimize(fun=logistic_loss, x0=theta_0, 
                                   args= (X, mean_op, self.lam), 
                                    jac= logistic_loss_grad, 
                                   method='L-BFGS-B')#, options={'gtol': 1e-6, 'disp': True})
        self.theta = res.x
        
        return self
        
    def predict(self, X):
        return np.sign(sp.special.expit(2 * np.dot(X.iloc[:], self.theta)) - .5)
        
        
        
class LaplacianMeanMapGridSearch:
    """
    Grid Search for laplacian mean map. This implementation is more efficient
    than one using GridSearchCV() on LaplacianMeanMap() as we can compute
    and store L for each of the CV folds --for big datasets computing L from
    is often the bottleneck
    """
                 
    def __init__(self, cv=5, alphas=[0], L_normed=[False],
                 gammas=[1], w_types=['identity'], epsilon=0,
                d=['rbf'], sigmas=[1]):
        self.cv = cv
        self.alphas = alphas
        self.L_normed = L_normed
        self.gammas = gammas
        self.w_types = w_types
        self.epsilon = epsilon
#       self.ds = d NOT USED SO FAR, it will list different similarity fun
        self.sigmas = sigmas
        self.theta = 0.0
#        self.dataframe_result = pd.DataFrame()
#        self.dataframe_result.columm = ['alpha', 'L_normed', 'gamma', 'D_w', 'eps', 
#                                        'd', 'sigma', 'accuracy']
                                        
    def compute_D_w(self, w_type='identity', p=None):

        n = self.n
        if p and n != len(p):
            raise ValueError("The vector p must be of size n, but %d != %d"
                            .format(n, len(p)))
        D_w = np.identity(n)
        if w_type == 'identity':
            return D_w  
        elif w_type == 'bag-size':
            return D_w * 1/np.array(p)
        elif w_type == 'inverse-bag-size':
            return D_w * np.array(p)


    def _fit(self, X, Pi, B, p, D_w, L, gamma, alpha):
    
        #Step1 - Least Squares
        Pi_dot_Dw = np.dot(Pi,D_w)
        B_pm = sp.linalg.solve((np.dot(Pi_dot_Dw, np.transpose(Pi)) + \
            gamma * L), np.dot(Pi_dot_Dw, B))
        
        #Step 2 - Mean Operator estimation
        pi = np.diag(Pi[:,:self.n]) #get the proportion in an array
        b_plus = B_pm[:self.n]
        b_minus = B_pm[-self.n:]
        mean_op = np.sum(np.array([(p[j] * (pi[j] * b_plus[j] - \
            (1-pi[j]) * b_minus[j])) for j in range(self.n)]), axis=0)
        
        #Step 3 - Logistic regression
        theta_0 = np.empty(X.shape[1])
        theta_0.fill(0.0001)
        

        res = sp.optimize.minimize(fun=logistic_loss, x0=theta_0, 
                                   args= (X, mean_op, alpha), 
                                    jac= logistic_loss_grad, 
                                   method='L-BFGS-B')#, options={'gtol': 1e-6, 'disp': True})
        
        self.theta = res.x
        return
        
        
    def fit(self, X, pi):
        
        #problem dimension
        self.m = X.shape[0] #sample size
        self.n = len(pi) #numer of bags
        
        #precompute all bag-wise means on train-validation splits
        B = list()
        #bag ids 
        bag_id = X.index.get_level_values('bag')
        
        #keep the same object for CV below
        random_state = 14
        folds = BaggedKFold(bag_id, n_folds=self.cv, random_state=random_state)
        
        for ix_train, ix_validation in folds:
            B.append(compute_bag_means(X.iloc[ix_train]))

        #estimate the bag frequencies
        p = [X.loc[i].shape[0] / self.m for i in range(self.n)]            

        #set the proportions matrix --this does not change as we do not know the labels
        Pi = np.transpose(np.hstack((np.diag(pi), np.diag(1-pi))))
                
        #grid search
        opt_sigma, opt_D_w, opt_gamma, opt_alpha = 1, 'identity', 1, 0
        opt_score = np.inf #we want to minimize this score

        
#       for d, sigma, L_normed in product(self.ds, self.sigmas, self.L_normed):   
        for sigma, L_normed in product(self.sigmas, self.L_normed):   

            La = [ laplacian_matrix(B[i], d, normed=L_normed) for i in range(self.cv)]
            #set the block Laplacian
            L = [ self.epsilon * np.identity(2*self.n) +
            sp.linalg.block_diag(La[i], La[i]) for i in range(self.cv)]
            
            for w_type in self.w_types:
                #set weight matrix
                D_w = self.compute_D_w(w_type=w_type)

                for gamma, alpha in product(self.gammas, self.alphas):
                
                    #cross validation of these params
                    score = 0
                    for v, (idx_train, idx_validation) in enumerate(folds):

                        self._fit(X.iloc[idx_train], Pi, B[v], p, D_w, L[v], gamma, alpha)
                        score += 1/self.cv * proportions_abs_err(X, pi, p, self.predict(X.iloc[idx_validation]))
                    
                    #print("score {} - sigma: {}, D_w: {}, gamma: {}, alpha: {}"
                    #        .format(score, sigma, w_type, gamma, alpha))
                    #check if the params improve solution
                    if np.mean(score) < opt_score:
                        opt_score = score
                        opt_sigma, opt_D_w, opt_gamma, opt_alpha = \
                        sigma, w_type, gamma, alpha
                        
                        
        #recompute L for the whole data, re-train with the opt params
        B = compute_bag_means(X)
        L_a = laplacian_matrix(B, d)      
        L = np.identity(2*self.n) * self.epsilon + sp.linalg.block_diag(L_a, L_a)
        
        self._fit(X, Pi, B, p, D_w, L, opt_gamma, opt_alpha)
        
        return self
    
    def predict(self, X):
        return np.sign(sp.special.expit(2 * np.dot(X.iloc[:], self.theta)) - .5)
