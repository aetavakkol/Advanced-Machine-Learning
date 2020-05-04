

#import numpy as np
##%matplotlib inline
#import matplotlib.pyplot as plt
##import sys
##sys.path.append("C:\\Users\\aettehad\\source\\repos\\intro-to-dl\\week1")
##import grading
#import os
#os.chdir(r"C:\Users\aettehad\Source\Repos\intro-to-dl\week1")
#path  = os.getcwd()

#with open('train.npy', 'rb') as fin:
#    X = np.load(fin)
    
#with open('target.npy', 'rb') as fin:
#    y = np.load(fin)

## Show X-Y-Class Plot
##plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=20)
##plt.show()

#def expand(X):
#    """
#    Adds quadratic features. 
#    This expansion allows your linear model to make non-linear separation.
    
#    For each sample (row in matrix), compute an expanded row:
#    [feature0, feature1, feature0^2, feature1^2, feature0*feature1, 1]
    
#    :param X: matrix of features, shape [n_samples,2]
#    :returns: expanded features of shape [n_samples,6]
#    """
#    X_expanded = np.zeros((X.shape[0], 6))
#    X_expanded[:,0] = X[:, 0]
#    X_expanded[:,1] = X[:, 1]
    
#    X_expanded[:,2] = X[:, 0]*X[:, 0]
#    X_expanded[:,3] = X[:, 1]*X[:, 1]
    
#    X_expanded[:,4] = X[:, 0]*X[:, 1]
#    X_expanded[:, 5] = 1
    
#    return X_expanded


#X_expanded = expand(X)


## simple test on random numbers

#dummy_X = np.array([
#        [0,0],
#        [1,0],
#        [2.61,-1.28],
#        [-0.59,2.1]
#    ])

## call your expand function
#dummy_expanded = expand(dummy_X)

## what it should have returned:   x0       x1       x0^2     x1^2     x0*x1    1
#dummy_expanded_ans = np.array([[ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  1.    ],
#                               [ 1.    ,  0.    ,  1.    ,  0.    ,  0.    ,  1.    ],
#                               [ 2.61  , -1.28  ,  6.8121,  1.6384, -3.3408,  1.    ],
#                               [-0.59  ,  2.1   ,  0.3481,  4.41  , -1.239 ,  1.    ]])

##tests
#assert isinstance(dummy_expanded,np.ndarray), "please make sure you return numpy array"
#assert dummy_expanded.shape == dummy_expanded_ans.shape, "please make sure your shape is correct"
#assert np.allclose(dummy_expanded,dummy_expanded_ans,1e-3), "Something's out of order with features"

#print("Seems legit!")



#def probability(X, w):
#    """
#    Given input features and weights
#    return predicted probabilities of y==1 given x, P(y=1|x), see description above
        
#    Don't forget to use expand(X) function (where necessary) in this and subsequent functions.
    
#    :param X: feature matrix X of shape [n_samples,6] (expanded)
#    :param w: weight vector w of shape [6] for each of the expanded features
#    :returns: an array of predicted probabilities in [0,1] interval.
#    """
    
#    X_expanded = expand(X)
#    Prob_X_expanded = np.zeros((X_expanded.shape[0], 1))
    
#    Prob_X_expanded = 1/(1+np.exp( -np.inner(w,X_expanded)) )
    
#    return Prob_X_expanded
    

## test the probability function
#dummy_weights = np.linspace(-1, 1, 6)
#ans_part1 = probability(X_expanded[:1, :], dummy_weights)[0]

#def compute_loss(X, y, w):
#    """
#    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
#    and weight vector w [6], compute scalar loss function L using formula above.
#    Keep in mind that our loss is averaged over all samples (rows) in X.
#    """
#    Prob_YX = probability(X, w)

#    #l = - (np.inner(y, np.log(Prob_YX) ) + np.inner(1-y, 1-np.log(Prob_YX) ) )/y.size
#    l = - (y*np.log(Prob_YX)  + (1-y)*(np.log(1-Prob_YX) ) )
    
#    #l_sum = np.sum(l)/y.size
#    return np.average(l)

## use output of this cell to fill answer field 
#ans_part2 = compute_loss(X_expanded, y, dummy_weights)

#print (ans_part2)


#def compute_grad(X, y, w):
#    """
#    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
#    and weight vector w [6], compute vector [6] of derivatives of L over each weights.
#    Keep in mind that our loss is averaged over all samples (rows) in X.
#    """
#    Prob_YX = probability(X, w)
#    grad1 = np.dot(y, X)
#    grad2 = np.dot(Prob_YX, X)
#    return (grad2-grad1)/y.size


#ans_part3 = np.linalg.norm(compute_grad(X_expanded, y, dummy_weights))
#print (ans_part3)


#from IPython import display

#h = 0.01
#x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#def visualize(X, y, w, history):
#    """draws classifier prediction with matplotlib magic"""
#    Z = probability(expand(np.c_[xx.ravel(), yy.ravel()]), w)
#    Z = Z.reshape(xx.shape)
#    plt.subplot(1, 2, 1)
#    plt.contourf(xx, yy, Z, alpha=0.8)
#    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
#    plt.xlim(xx.min(), xx.max())
#    plt.ylim(yy.min(), yy.max())
    
#    plt.subplot(1, 2, 2)
#    plt.plot(history)
#    plt.grid()
#    ymin, ymax = plt.ylim()
#    plt.ylim(0, ymax)
#    display.clear_output(wait=True)
#    plt.show()
    


##visualize(X, y, dummy_weights, [0.5, 0.5, 0.25])


## please use np.random.seed(42), eta=0.1, n_iter=100 and batch_size=4 for deterministic results

#np.random.seed(42)
#w = np.array([0, 0, 0, 0, 0, 1])

#eta= 0.1 # learning rate

#n_iter = 100
#batch_size = 4
#loss = np.zeros(n_iter)
#plt.figure(figsize=(12, 5))

#for i in range(n_iter):
#    ind = np.random.choice(X_expanded.shape[0], batch_size)
#    loss[i] = compute_loss(X_expanded, y, w)
#    #if i % 10 == 0:
#    #    visualize(X_expanded[ind, :], y[ind], w, loss)

#    # Keep in mind that compute_grad already does averaging over batch for you!

#    wnew = w - eta*compute_grad(X_expanded[ind, :], y[ind], w)
#    if np.linalg.norm(wnew - w) < 1e-8:
#        break
#    else:
#        w = wnew



##visualize(X, y, w, loss)
##plt.clf()


##SGD with momentum

##Momentum is a method that helps accelerate SGD in the relevant direction and 
##dampens oscillations as can be seen in image below. It does this by adding a 
##fraction αα of the update vector of the past time step to the current update vector. 
## please use np.random.seed(42), eta=0.05, alpha=0.9, n_iter=100 and batch_size=4 for deterministic results
#np.random.seed(42)
#w = np.array([0, 0, 0, 0, 0, 1])

#eta = 0.05 # learning rate
#alpha = 0.9 # momentum
#nu = np.zeros_like(w)

#n_iter = 100
#batch_size = 4
#loss = np.zeros(n_iter)
#plt.figure(figsize=(12, 5))

#for i in range(n_iter):
#    ind = np.random.choice(X_expanded.shape[0], batch_size)
#    loss[i] = compute_loss(X_expanded, y, w)
#    #if i % 10 == 0:
#    #    visualize(X_expanded[ind, :], y[ind], w, loss)

#    # TODO:<your code here>
#    if i == 0:
#        vt = eta*compute_grad(X_expanded[ind, :], y[ind], w)
#    else:
#        vt = alpha*vt + eta*compute_grad(X_expanded[ind, :], y[ind], w)
    

#    wnew = w - vt

#    if np.linalg.norm(wnew - w) < 1e-8:
#        break
#    else:
#        w = wnew


##visualize(X, y, w, loss)
##plt.clf()

##RMSprop
##Implement RMSPROP algorithm, which use squared gradients to adjust learning rate:

## please use np.random.seed(42), eta=0.1, alpha=0.9, n_iter=100 and batch_size=4 for deterministic results
#np.random.seed(42)

#w = np.array([0, 0, 0, 0, 0, 1.])

#eta = 0.1 # learning rate
#alpha = 0.9 # moving average of gradient norm squared
#g2 = None # we start with None so that you can update this value correctly on the first iteration
#eps = 1e-8

#n_iter = 100
#batch_size = 4
#loss = np.zeros(n_iter)
#plt.figure(figsize=(12,5))
#for i in range(n_iter):
#    ind = np.random.choice(X_expanded.shape[0], batch_size)
#    loss[i] = compute_loss(X_expanded, y, w)
#    if i % 10 == 0:
#        visualize(X_expanded[ind, :], y[ind], w, loss)

#    # TODO:<your code here>
#    gt = compute_grad( X_expanded[ind, :], y[ind], w )
#    if i == 0:
#        #Gt = (1-alpha)*np.sum( gt**2 )
#        Gt = (1-alpha)*np.linalg.norm( gt )
#    else:
#        #Gt = alpha*Gt + (1-alpha)*np.sum( gt**2 )
#        Gt = alpha*Gt + (1-alpha)*np.linalg.norm( gt )
    

#    wnew = w - eta/np.sqrt(Gt+eps)*gt

#    if np.linalg.norm(wnew - w) < 1e-8:
#        break
#    else:
#        w = wnew

#visualize(X, y, w, loss)
#plt.clf()
