# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:49:52 2015

@author: giorgio
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

def to_zero_one(label):
    "binary label transformation {-1,+1} -> {0,1}"
    return ((label + 1)/2).astype(np.int)
        
def to_one_one(label):
    "binary label transformation {-1,+1} -> {0,1}"
    return (2*(label) - 1).astype(np.int)
    

    
def assign_bags(X, strategy='random', n=10, random_state=int(0)):
    """
    Simulate bags assigment with a fully labelled dataset
    As side effect, when the method is called with stragety == "feature",
    it also removed that feature from X
    """

    if strategy == 'random':
        #Random index for bags
        np.random.seed(random_state)
        bag_id = np.random.randint(0, n, X.shape[0])
    else:
        if not strategy in X.columns:
            raise NameError("unknownn strategy for bags creation")
        
        bag_id = pd.Categorical(X[strategy]).labels  
        #then, drop the features
        X = X.drop(strategy, axis=1)
        
    return X, bag_id
    
    
def create_bags(X, y, bag_id):
    """
    Create a hierarchical index for the bag on X,
    and compute the label proportions for each bag
    """
    
    X['bag'] = bag_id
    X.set_index(['bag', X.index], inplace=True)
    X.index.set_names(['bag', 'i'])
    X.sort_index(axis=0, inplace=True) #sort by bag and then i
    
    y.index = bag_id #order within bag does not matter
    prop = y.groupby(y.index).apply(lambda x: np.mean(x))
    
    return(X, prop)
    
    
def d(x1, x2, distance="rbf", sigma=1):
    
    if distance == "rbf":
        return np.exp(- sigma * d(x1, x2, distance="L2"))    
    elif distance == "L2":
        return np.sum((x1-x2) ** 2)
    elif distance == "L1":
        return np.sum(np.abs(x1-x2))
    elif distance == "dot":
        return np.inner(x1,x2)
        
    
def preprocess_adult(data_path, train_subsample=False, bag_strategy='random',
                     n_bag = 10):
    
    adult_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race',
               'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
               'native-country', 'income']


    #Read files
    train_csv = pd.read_csv(data_path + "adult.data.txt", header=None, names=adult_names)
    test_csv = pd.read_csv(data_path + "adult.test.txt", header=None, names=adult_names)
    train_csv = train_csv.drop(train_csv.tail(1).index) #last row is Nan
    test_csv = test_csv.drop(test_csv.tail(1).index)


    #Shuffle train
    np.random.seed(100)
    train_csv.index = np.random.permutation(train_csv.index)


    #Labels
    y_train = train_csv['income']
    y_test = test_csv['income']
    train_csv = train_csv.drop('income', 1)
    test_csv = test_csv.drop('income', 1)

    y_train[(y_train == " <=50K") | (y_train == " <=50K.")] = 0
    y_train[(y_train == " >50K") | (y_train == " >50K.")] = 1
    y_test[(y_test == " <=50K") | (y_test == " <=50K.")] = 0
    y_test[(y_test == " >50K") | (y_test == " >50K.")] = 1


    #If train_subsample is a integer, only take a slice of train
    if train_subsample != False:       
        train_csv = train_csv[1:train_subsample]
        y_train = y_train[1:train_subsample]
        

    if bag_strategy == 'random':
        train_csv, bag_id = assign_bags(train_csv, strategy=bag_strategy, n=n_bag)
    else: #otherwise, by feature
        train_csv, bag_id = assign_bags(train_csv, strategy=bag_strategy)
        test_csv = test_csv.drop(bag_strategy, axis=1)
        
    
    #Join train and test for data preprocessing
    train_test = train_csv.append(test_csv, ignore_index=True)
    train_size = train_csv.shape[0]
    del train_csv
    del test_csv
    
    
    ##
    ## Feature Engineering
    ##
    
    #categorical names    
    cat_col = [adult_names[i] for i in [1,3,5,6,7,8,9,13]]
    #continuous names
    cont_col = [adult_names[i] for i in [0,2,4,10,11,12]]


    #Exclude the feature used to make the bags, if any
    if bag_strategy != 'random':
        cat_col = list(set(cat_col) - set([bag_strategy]))
        cont_col = list(set(cont_col) - set([bag_strategy]))

    
    #categoricals
    for i in np.intersect1d(cat_col, train_test.columns.values):
        temp = pd.get_dummies(train_test[i])
        names = [i + " %s" % j for j in range(temp.shape[1])]
        train_test[names] = temp 
        del train_test[i]
    
    
    #Split train and test as before
    train = train_test[:train_size]
    test = train_test[train_size:]
    
    
    #numerical features - also include dates
    cont_col = np.intersect1d(cont_col, train_test.columns.values)
    scaler = preprocessing.StandardScaler()
    train[cont_col] = scaler.fit_transform(train[cont_col])
    test[cont_col] = scaler.transform(test[cont_col])
    
    del train_test
    
    ###
    #End feature engineering
    ###

    #Make bags: assign examples to bags
    #The map is given by a hierarchical index in train [bag, i]
    #Proportions pi are computed accordingly
    X_train, pi = create_bags(train, y_train, bag_id)
    
    #Cast to np.array of int
    y_train = np.array(y_train).astype(np.int)
    y_test = np.array(y_test).astype(np.int)    
    
    return X_train, pi, y_train, test, y_test
    
    