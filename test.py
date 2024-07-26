#%pylab
#%matplotlib inline
from matplotlib import pyplot as plt
from numpy import *
import numpy as np
import random
random.seed(12)

import mnist
mnist.datasets_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
# take only certain digits
Digits = array([0,1,2])
nDigits = len(Digits)

labels = mnist.train_labels()
digs = []
for i in range(len(labels)):
    if labels[i] in Digits:
        digs.append(i)

labels = labels[digs]
images = mnist.train_images()[digs,4:-4,4:-4]   # start with 20X20 pixel images

ex = 2

fig, axes = plt.subplots(1,2)
#fig, axes = subplots(1,2)
#axes[0].imshow(images[ex,:,:] * -1, cmap='gray')
axes[0].imshow(images[ex,:,:] * -1, cmap='gray')
axes[0].set_title('20X20 pixels')
print(labels[ex])

from skimage.measure import block_reduce
Dimages = block_reduce(images, (1,2,2), func=mean)  # downsample images to 10X10 pixels
axes[1].imshow(Dimages[ex,:,:] * -1, cmap='gray')
axes[1].set_title('10X10 pixels')

#plt.show()
# define training set
TrainSize = int(labels.shape[0]*0.4)
X_Train_Set = Dimages[:TrainSize]
Y_Train_Set = labels[:TrainSize]

# define test set
TestSize = int(labels.shape[0]*0.6)
X_Test_Set = Dimages[TrainSize:TrainSize+TestSize]
Y_Test_Set = labels[TrainSize:TrainSize+TestSize]

# number of sources and targets
nSources = size(X_Train_Set[0])
nTargets = nDigits
nGrounds = 1

# create random grapgh
nNodes = nSources * 3    # number of network nodes
nEdges = nNodes * 5      # number of network edges
lNode, rNode = array([np.random.choice(array(nNodes), size=2, replace=False) for i in range(nEdges)]).T

# square grid
nGrid = 23
nNodes = nGrid**2
#nEdges = nNodes * 5      # number of network edges
#lNode, rNode = array([np.random.choice(array(nNodes), size=2, replace=False) for i in range(nEdges)]).T
lNode, rNode = [], []
for i in range(nGrid):
  for j in range(nGrid):
    k = j + i * nGrid
    if(j < nGrid - 1):
      lNode.append(k)
      rNode.append(k + 1)
    if(i < nGrid - 1):
      lNode.append(k)
      rNode.append(k + nGrid)
lNode = array(lNode, dtype=int)
rNode = array(rNode, dtype=int)
nEdges = len(lNode)

# create sparse network structures

# Choose input and output nodes
# RNodes = choice(arange(nNodes), size=nSources+nTargets+nGrounds, replace=False)
# SourceNodes = RNodes[:nSources]
# TargetNodes = RNodes[nSources:nSources+nTargets]
# GroundNodes = [RNodes[-1]]
# SourceEdges = []
# TargetEdges = []

# Choose input and output edges
REdges = np.random.choice(arange(1, nEdges), size=nSources+nTargets, replace=False)
SourceEdges = REdges[:nSources]
TargetEdges = REdges[nSources:]
GroundNodes = [0]
SourceNodes = []
TargetNodes = []

# Make sparse structures for network response computation
from SparseLinearNetwork import SparseIncidenceConstraintMatrix
sDMF, sDMC, sBLF, sBLC, sDot = SparseIncidenceConstraintMatrix(SourceNodes, SourceEdges, TargetNodes, TargetEdges, GroundNodes, nNodes, lNode, rNode)
print('Network size: ', nNodes,' Nodes, ', nEdges, ' Edges')

# Training voltage inputs
ff = zeros([TrainSize, sDMF.shape[1]])     # free constraints templeate vector
ff[:, nNodes+nGrounds:] = reshape(X_Train_Set, [TrainSize, nSources])

# Test voltage inputs
ffT = zeros([TestSize, sDMF.shape[1]])     # free constraints templeate vector
ffT[:, nNodes+nGrounds:] = reshape(X_Test_Set, [TestSize, nSources])

# initialize conductance values
import numpy.random as rand
from scipy.sparse import spdiags, diags
K0 = 1. + (rand.rand(nEdges) - 0.5) * 0.1
sK0 = spdiags(K0, 0, nEdges, nEdges, format='csc')          # diagonal matrix with the conductance values on the diagonal elements

# Compute network responses on training set
import pypardiso
PF = pypardiso.spsolve(sBLF + sDMF.T*sK0*sDMF, ff.T)
FST = sDot.dot(PF)
MFST = mean(FST, 1)  # mean response for all training examples

# use mean response to training examples to set desired target values
fc = zeros([TrainSize, sDMC.shape[1]])                      # clamped constraints template vector
Desired = zeros([TrainSize, sDMC.shape[1]-sDMF.shape[1]])   # Desired outputs
DesiredT = zeros([TestSize, sDMC.shape[1]-sDMF.shape[1]])   # Desired test outputs

Targets = zeros([nDigits, nDigits])
amp = 2.
for i in range(nDigits):
    ids = where(Y_Train_Set==Digits[i])[0]
    # Clamping targets for each class
    Targets[i] = MFST + amp * (mean(FST[:,ids], 1) - MFST)

    fc[ids, nNodes+nGrounds:nNodes+nGrounds+nSources] = reshape(X_Train_Set[ids], [len(ids), nSources])
    fc[ids, nNodes+nGrounds+nSources:] = Targets[i]
    Desired[ids] = Targets[i]

    idsT = where(Y_Test_Set==Digits[i])[0]
    DesiredT[idsT] = Targets[i]


# Initial network response to training examples
PF = pypardiso.spsolve(sBLF + sDMF.T*sK0*sDMF, ff.T)
# Response of the output degrees of freedom (output nodes\edges)
FST = (sDot.dot(PF)).T
# Distance between output responses to each one of the digit classes
Ds = array(sqrt(sum([[(Targets[c] - FST[i])**2 for i in range(TrainSize)] for c in range(nDigits)], 2)))
# Class is decided by the closest digit target to the output response
minDs = argmin(Ds, axis=0)
# Classification accuracy
Chg = (minDs != Y_Train_Set)
Acc = sum(1-Chg)/TrainSize

# Same as above, but for test examples
PF = pypardiso.spsolve(sBLF + sDMF.T*sK0*sDMF, ffT.T)
FST = (sDot.dot(PF)).T
Ds = array(sqrt(sum([[(Targets[c] - FST[i])**2 for i in range(TestSize)] for c in range(nDigits)], 2)))
minDs = argmin(Ds, axis=0)
Chg = (minDs != Y_Test_Set)
AccT = sum(1-Chg)/TestSize

print('Initial training accuracy: ', round(Acc,3))
print('Initial test accuracy: ',  round(AccT,3))

# Training using coupled learning

K = K0.copy()
sK = spdiags(K, 0, nEdges, nEdges, format='csc')

Steps = 5000    # number of steps
eta = 1.e-3      # nudge parameter
lr = 1.e-4       # learning rate

# Training errors and accuracies
Cs = []
Accs = []

# Responses and accuracy of training set
PF = pypardiso.spsolve(sBLF + sDMF.T*sK*sDMF, ff.T)
C = 0.5 * sum((sDot.dot(PF) - Desired.T)**2)/TrainSize
FST = (sDot.dot(PF)).T
Ds = array(sqrt(sum([[(Targets[c] - FST[i])**2 for i in range(TrainSize)] for c in range(nDigits)], 2)))
minDs = argmin(Ds, axis=0)
Chg = (minDs != Y_Train_Set)
Acc = sum(1-Chg)/TrainSize
Cs.append(C)
Accs.append(Acc)

# Test errors and accuracies
CTs = []
AccTs = []

# Responses and accuracy of test set
PF = pypardiso.spsolve(sBLF + sDMF.T*sK*sDMF, ffT.T)
C = 0.5 * sum((sDot.dot(PF) - DesiredT.T)**2)/TestSize
FST = (sDot.dot(PF)).T
Ds = array(sqrt(sum([[(Targets[c] - FST[i])**2 for i in range(TestSize)] for c in range(nDigits)], 2)))
minDs = argmin(Ds, axis=0)
Chg = (minDs != Y_Test_Set)
AccT = sum(1-Chg)/TestSize
CTs.append(C)
AccTs.append(AccT)

print('Step 0  ;  Training accuracy: ', round(Acc,3), ',  Test accuracy: ', round(AccT,3))

BestK = K0
BestAccT = AccT

printSteps = 50    # print accuracy every number of training steps

#iterate over training steps
for steps in range(1,Steps+1):
    # Free state
    PF = pypardiso.spsolve(sBLF + sDMF.T*sK*sDMF, ff.T)
    FST = (sDot.dot(PF)).T
    DPF = sDMF.dot(PF)
    PPF = DPF**2

    # correctly classified input examples
    Ds = array(sqrt(sum([[(Targets[c] - FST[i])**2 for i in range(TrainSize)] for c in range(nDigits)], 2)))
    minDs = argmin(Ds, axis=0)
    Chg = (minDs != Y_Train_Set)

    # Clamped state computation
    #FST = (sDot.dot(PF)).T
    Nudge = FST.T + eta * (Desired.T - FST.T)
    fc[:,-nDigits:] = Nudge.T
    PC = pypardiso.spsolve(sBLC + sDMC.T*sK*sDMC, fc.T)
    DPC = sDMC.dot(PC)
    PPC = DPC**2

    # Coupled learning rule
    DKL = + 0.5 * (PPC - PPF) / eta
    K2 = K - lr * mean(DKL * Chg, axis=1) * TrainSize/(sum(Chg)+1.e-10)   # Cross entropy rule
    K2 = K2.clip(1.e-6,1.e4)

    DK = K2-K
    K = K2
    sK = spdiags(K, 0, nEdges, nEdges, format='csc')

    if steps%printSteps == 0:
        # Training errors and accuracies
        PF = pypardiso.spsolve(sBLF + sDMF.T*sK*sDMF, ff.T)
        C = 0.5 * sum((sDot.dot(PF) - Desired.T)**2)/TrainSize
        FST = (sDot.dot(PF)).T
        Ds = array(sqrt(sum([[(Targets[c] - FST[i])**2 for i in range(TrainSize)] for c in range(nDigits)], 2)))
        minDs = argmin(Ds, axis=0)
        Chg = (minDs != Y_Train_Set)
        Acc = sum(1-Chg)/TrainSize
        Cs.append(C)
        Accs.append(Acc)

        # Test errors and accuracies
        PF = pypardiso.spsolve(sBLF + sDMF.T*sK*sDMF, ffT.T)
        C = 0.5 * sum((sDot.dot(PF) - DesiredT.T)**2)/TestSize
        FST = (sDot.dot(PF)).T
        Ds = array(sqrt(sum([[(Targets[c] - FST[i])**2 for i in range(TestSize)] for c in range(nDigits)], 2)))
        minDs = argmin(Ds, axis=0)
        Chg = (minDs != Y_Test_Set)
        AccT = sum(1-Chg)/TestSize
        CTs.append(C)
        AccTs.append(AccT)

        print('Step', steps, '  ;  Training accuracy: ', round(Acc,3), ',  Test accuracy: ', round(AccT,3))

        if AccT > BestAccT:
            BestAccT = AccT
            BestK = K
            print('Current best test accuracy: ', round(AccT,3))

        if Acc == 1.:
            break

# Classification on training set

from sklearn.decomposition import PCA
from sklearn import svm

fig, axes = plt.subplots(1,2, figsize=(10,4))

# Before training
sK = spdiags(K0, 0, nEdges, nEdges, format='csc')
PF = pypardiso.spsolve(sBLF + sDMF.T*sK*sDMF, ff.T)
pca = PCA(n_components=2, whiten=True)
pca.fit(sDot.dot(PF).T)
X_pca = pca.transform(sDot.dot(PF).T)
for i in range(nDigits):
    ids = (Y_Train_Set == Digits[i])
    axes[0].plot(X_pca.T[0][ids], X_pca.T[1][ids], 'x', label=Digits[i])
axes[0].set_title('Training set before training')

reduced_data = pca.transform(sDot.dot(PF).T)
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
model = svm.SVC(kernel='linear')
clf = model.fit(reduced_data, Y_Train_Set)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
axes[0].contour(xx,yy,Z,colors='k',alpha=0.2)


# After training
sK = spdiags(BestK, 0, nEdges, nEdges, format='csc')
PF = pypardiso.spsolve(sBLF + sDMF.T*sK*sDMF, ff.T)
pca = PCA(n_components=2, whiten=True)
pca.fit(sDot.dot(PF).T)
X_pca = pca.transform(sDot.dot(PF).T)
for i in range(nDigits):
    ids = (Y_Train_Set == Digits[i])
    axes[1].plot(X_pca.T[0][ids], X_pca.T[1][ids], 'x', label=Digits[i])
axes[1].legend()
axes[1].set_title('Training set after training')

reduced_data = pca.transform(sDot.dot(PF).T)
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
model = svm.SVC(kernel='linear')
clf = model.fit(reduced_data, Y_Train_Set)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
axes[1].contour(xx,yy,Z,colors='k',alpha=0.2)

# Classification on test set

from sklearn.decomposition import PCA
from sklearn import svm

fig, axes = plt.subplots(1,2, figsize=(10,4))

# Before training
sK = spdiags(K0, 0, nEdges, nEdges, format='csc')
PF = pypardiso.spsolve(sBLF + sDMF.T*sK*sDMF, ffT.T)
pca = PCA(n_components=2, whiten=True)
pca.fit(sDot.dot(PF).T)
X_pca = pca.transform(sDot.dot(PF).T)
for i in range(nDigits):
    ids = (Y_Test_Set == Digits[i])
    axes[0].plot(X_pca.T[0][ids], X_pca.T[1][ids], 'x', label=Digits[i])
axes[0].set_title('Test set before training')

reduced_data = pca.transform(sDot.dot(PF).T)
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
model = svm.SVC(kernel='linear')
clf = model.fit(reduced_data, Y_Test_Set)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
axes[0].contour(xx,yy,Z,colors='k',alpha=0.2)


# After training
sK = spdiags(BestK, 0, nEdges, nEdges, format='csc')
PF = pypardiso.spsolve(sBLF + sDMF.T*sK*sDMF, ffT.T)
pca = PCA(n_components=2, whiten=True)
pca.fit(sDot.dot(PF).T)
X_pca = pca.transform(sDot.dot(PF).T)
for i in range(nDigits):
    ids = (Y_Test_Set == Digits[i])
    axes[1].plot(X_pca.T[0][ids], X_pca.T[1][ids], 'x', label=Digits[i])
axes[1].legend()
axes[1].set_title('Test set after training')

reduced_data = X_pca
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
model = svm.SVC(kernel='linear')
clf = model.fit(reduced_data, Y_Test_Set)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
axes[1].contour(xx,yy,Z,colors='k',alpha=0.2)

# Network graphs

import networkx as nx
G = nx.from_edgelist(c_[lNode,rNode])

colors = ['blue'] * nNodes
for i in SourceEdges:
  colors[lNode[i]] = 'red'
  colors[rNode[i]] = 'red'
for i in TargetEdges:
  colors[lNode[i]] = 'green'
  colors[rNode[i]] = 'green'

fig, axes = plt.subplots(1,2, figsize=(14,6))

K = K0
nx.draw_kamada_kawai(G, ax=axes[0], node_size=10, width=1.*K/K.max())
axes[0].set_title('Network before training\n Test accuracy: '+ str(round(AccTs[0],3)), size=18)

K = BestK
nx.draw_kamada_kawai(G, ax=axes[1], node_size=10, width=2.*K/K.max())
axes[1].set_title('Network after training\n Test accuracy: '+ str(round(BestAccT,3)), size=18)

plt.show()
