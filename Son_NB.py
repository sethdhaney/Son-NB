
# Random Forest Algorithm on Sonar Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from pylab import *

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#Calc Prob of Priors
def calc_prior(Y, C):
	return Y.count(C)/len(Y)

#PRE-Calc parameters for GAUSSIAN dist. vars
#Also assumes y in Y_RANGE
def pre_calc_ev_param(X, Y, mu, sig, y_Range):
	mu = list()
	sig = list()
	for C in y_Range:
		mu_t = list()
		sig_t = list()
		idx = find(Y==C)
		for i in range(0,X.shape[1]):
			mu_t.append(sum(X[idx,i])/len(idx))
			sig_t.append(sqrt((sum(X[idx,i]*X[idx,i])/len(idx)) - mu_t[-1]*mu_t[-1]))
		mu.append(mu_t)
		sig.append(sig_t)
	mu = np.array(mu)
	sig = np.array(sig)
	return mu, sig

#Predict using model
def predict(x_t, mu, sig, y_Range,priors):
	p = list()
	for C in y_Range:
		y = 1
		for j in range(0,shape(x_t)[0]): #Loop over features
			y = y * (1/sqrt(2*pi*sig[C,j]*sig[C,j])) \
				* exp(-(x_t[j] - mu[C,j]) \
				*(x_t[j] - mu[C,j]) \
				/(2*sig[C,j]*sig[C,j]))


		y = y*priors[C]
		p.append(y)
	return argmax(p), p[argmax(p)]

# LOAD CSV File
filename = 'sonar.all-data.csv'

dataset = list()
with open(filename, 'r') as file:
	csv_reader = reader(file)
	for row in csv_reader:
		if not row:
			continue
		dataset.append(row)

# Separate features(X) from Class(Y)
for i in range(0, len(dataset[0])-1):
	for row in dataset:                            
		row[i] = float(row[i].strip())
 
X = [row[0:-1] for row in dataset] 
X = np.array(X)
X_T = X.transpose()

Y = list()



for row in dataset:
	if row[-1] == 'R':
		Y.append(0)
	else:
		Y.append(1)

Y = np.array(Y)

### NAIVE BAYES ###
# Calc Prior = P(C), Evidence

y_Range = [0,1]

# Split Training and Test Set
p_Tr = 0.8
rnds = rand(len(Y))
iTr = find(rnds<p_Tr)
iTst = find(rnds>=p_Tr)
X_Tr = X[iTr,:]
Y_Tr = Y[iTr]
X_Tst = X[iTst,:]
Y_Tst = Y[iTst]

#Calc Params
mu = list()
sig = list()
mu, sig = pre_calc_ev_param(X_Tr, Y_Tr, mu, sig, y_Range)

#Predict on Training Set
pr = list()
priors = list()
for i in [0, 1]:
	priors.append(len(find(Y_Tr==i))/len(Y_Tr))

priors = array(priors)
for idx in iTr:
	pred, prob = predict(X[idx,:], mu, sig, [0, 1], priors)
	pr.append(pred)

pr = np.array(pr)
err_Tr = len(find(pr != Y_Tr))/len(Y_Tr)
print('Error on Training Set ', err_Tr)

#Predict on Test Set
pr = list()
for idx in iTst:
	pred, prob = predict(X[idx,:], mu, sig, [0,1], priors)
	pr.append(pred)

pr = np.array(pr)
err_Tst = len(find(pr != Y_Tst))/len(Y_Tst)
print('Error on Test Set', err_Tst)





