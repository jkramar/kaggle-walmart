# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import random
import theano
import numpy as np
import theano.tensor as T
import theano.sparse
import lasagne
from lasagne.layers import InputLayer,DenseLayer,get_output,get_all_params,dropout
from lasagne.nonlinearities import leaky_rectify,softmax,rectify
from lasagne.objectives import categorical_crossentropy
from lasagne.regularization import l1, l2, apply_penalty, regularize_layer_params_weighted
from lasagne.updates import nesterov_momentum
from lasagne.utils import create_param
from lasagne.init import Constant, GlorotUniform, GlorotNormal, Sparse, Orthogonal
import scipy.io
from scipy.sparse import csr_matrix
from scipy.io import mmread, netcdf
import os.path

import sparse_layers

import imp
imp.reload(sparse_layers)
from sparse_layers import SparseDropoutLayer, CondenseLayer
# %cd ~/Documents/shared/kaggle/walmart
# %cd /mnt/hgfs/shared/kaggle/walmart

def my_softmax(x):
    e_x = T.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

X=csr_matrix(mmread("X"),dtype=theano.config.floatX)
y=csr_matrix(mmread("tripTypes"),dtype=theano.config.floatX).todense()
test_X=csr_matrix(mmread("testX"),dtype=theano.config.floatX)

train_indices = np.ravel(np.where(y[:75000].sum(1)==1)[0])
valid_indices = np.ravel(np.where(y[75000:].sum(1)==1)[0]+75000)
train_X = X[train_indices]
valid_X = X[valid_indices]
train_y = y[train_indices]
valid_y = y[valid_indices]

def sf(x): return theano.shared(np.asarray(x, theano.config.floatX))

chunksizes = [69,69,69,69,5196,97715,69,5196,97715,100,12]
scales=[sf(1) for chunk in chunksizes]
rms=T.sqrt(sum(T.sqr(scale)*float(chunksize) for scale,chunksize in zip(scales,chunksizes))/float(sum(chunksizes)))
vscale=T.concatenate([T.repeat(scale/rms,chunksize) for scale,chunksize in zip(scales,chunksizes)])


minibatch_size = 5000
minibatches = [(indices, X[indices], y[indices]) for indices in np.array_split(train_indices, len(train_indices)/minibatch_size)]

inp_x = theano.sparse.csr_fmatrix()

l_in=InputLayer((None,X.shape[1]),name="inputs",input_var=inp_x)

l_hiddens = [CondenseLayer(l_in, num_units=100, nonlinearity=rectify, W=Orthogonal())]
for i in xrange(0):
    l_hiddens.append(DenseLayer(dropout(l_hiddens[-1]), num_units=100, nonlinearity=rectify))
l_out = DenseLayer(dropout(l_hiddens[-1]), num_units=y.shape[1], nonlinearity=softmax, W=Orthogonal())

def reset():
    if any(np.isnan(scale.get_value()) for scale in scales):
        for scale in scales:
            scale.set_value(1.)
    for l in l_hiddens:
        l.b.set_value(Constant()(l.b.get_value().shape))
        l.W.set_value(Orthogonal()(l.W.get_value().shape))
    l_out.b.set_value(Constant()(l_out.b.get_value().shape))
    l_out.W.set_value(Orthogonal()(l_out.W.get_value().shape))
    for p in (p for u in (updates_ada,updates_other,updates_scal) for p in u if p not in get_all_params(l_out)):
        p.set_value(Constant()(p.get_value().shape))
chunky_l2 = apply_penalty(get_all_params(l_out,regularizable=True),l2)-l2(l_hiddens[0].W)+l2(l_hiddens[0].W/T.reshape(vscale,(206279,1)))
chunky_l1 = apply_penalty(get_all_params(l_out,regularizable=True),l1)-l1(l_hiddens[0].W)+l1(l_hiddens[0].W/T.reshape(vscale,(206279,1)))
simple_l2 = apply_penalty(get_all_params(l_out,regularizable=True),l2)
#l_out2 = DenseLayer(dropout(l_hiddens2[-1]), num_units=y.shape[1])
#l_out = lasagne.layers.NonlinearityLayer(lasagne.layers.ElemwiseSumLayer((l_out1,l_out2),.5), softmax)

#categorical_crossentropy(get_output(l_out)[train_indice])

target=T.fmatrix(name="target")
#f=theano.function([l_in.input_var],get_output(l_out),allow_input_downcast=True)
#f(X[0,:].toarray())

loss=categorical_crossentropy(get_output(l_out),target).mean()
# train_loss_smoo=categorical_crossentropy(get_output(l_out,deterministic=True)[train_indices,],target[train_indices,]).mean()
# valid_loss=categorical_crossentropy(get_output(l_out)[valid_indices,],target[valid_indices,]).mean()
# valid_loss_smoo=categorical_crossentropy(get_output(l_out,deterministic=True)[valid_indices,],target[valid_indices,]).mean()
# objective=train_loss+.0001*l2(l_out.W)
loss_smoo=categorical_crossentropy(get_output(l_out,deterministic=True),target).mean()

# updates=nesterov_momentum(loss,get_all_params(l_out),learning_rate=.03)
momentum=sf(0)
learning_rate=sf(.1)

class force_cpu:
    def __enter__(self):
        self.c = theano.shared.constructors
        theano.shared.constructors = filter(lambda x: "cuda" not in x.func_code.co_filename, self.c)
    def __exit__(self, type, value, traceback):
        theano.shared.constructors = self.c
        
#updates_latest=lasagne.updates.adagrad(loss+.001*simple_l2,list(l_out.params)+list(l_hiddens[-1].params),learning_rate=.03)
#update_latest=theano.function([l_in.input_var,target],[loss],updates=updates_latest,allow_input_downcast=True)
#params = get_all_params(l_out)
#grads = T.grad(train_loss,params)

#updates=lasagne.updates.sgd(lasagne.updates.total_norm_constraint(grads, 1),get_all_params(l_out),learning_rate=.1)

#TODOTODOTODO NEXT try adagradding with l1 penalty, see what weight makes sense
# with chunky_l2, weight .001 worked

l2_wt=sf(.0005)
l1_wt=sf(.0001)

objective = loss+l2_wt*chunky_l2+l1_wt*chunky_l1
updates_scal=lasagne.updates.nesterov_momentum(objective,scales,learning_rate=.03)
with force_cpu():
    updates_hid=lasagne.updates.nesterov_momentum(objective,l_hiddens[0].params.keys(),learning_rate=learning_rate,momentum=momentum)
updates_other=lasagne.updates.nesterov_momentum(objective,l_out.params.keys(),learning_rate=learning_rate,momentum=momentum)
# updates_ada=lasagne.updates.adagrad(objective,get_all_params(l_out),learning_rate=.03)
update=theano.function([l_in.input_var,target],[loss],updates=updates_other,allow_input_downcast=True)
update_hid=theano.function([l_in.input_var,target],[loss],updates=updates_hid,allow_input_downcast=True)
#update_ada=theano.function([l_in.input_var,target],[loss],updates=updates_ada,allow_input_downcast=True)
update_scal=theano.function([l_in.input_var,target],[],updates=updates_scal,allow_input_downcast=True)
check=theano.function([l_in.input_var,target],[loss_smoo],allow_input_downcast=True)
predict=theano.function([l_in.input_var],[theano.Out(get_output(l_out,deterministic=True),borrow=True)],allow_input_downcast=True)

for fno in xrange(12,1000):
    reset()
    
    #diagn=theano.function([l_in.input_var,train_indices,valid_indices,target],[train_loss,valid_loss],allow_input_downcast=True,name="jill")
    filename="a_fitted_nnet_t_"+str(fno)+".nc"
    assert not os.path.exists(filename)
    f=netcdf.netcdf_file(filename,"w")
    f.createDimension("train",train_X.shape[0])
    f.createDimension("valid",valid_X.shape[0])
    f.createDimension("test",test_X.shape[0])
    f.createDimension("preds",37)
    v_train=f.createVariable("train",np.float,("train","preds"))
    v_valid=f.createVariable("valid",np.float,("valid","preds"))
    v_test=f.createVariable("test",np.float,("test","preds"))
    it = 0
    
    best_t = check(train_X,train_y)[0]
    best_it, best, v_train[:], v_valid[:], v_test[:] = (
        it, check(valid_X,valid_y), predict(train_X)[0], predict(valid_X)[0], predict(test_X)[0])
    
    
    random.shuffle(minibatches)
    try:
        best_it = it
        for i in xrange(1000):
            for j, X_m, y_m in minibatches:
                it += 1
                if it > best_it + 200: raise KeyboardInterrupt
                #(update_ada if i<10 else update)(X_m, y_m)
                momentum.set_value(1-.5/it)
                learning_rate.set_value(min(.1,it*.001))
                good_t = update(X_m, y_m)[0]
                # good_t = update_ada(X_m, y_m)[0]
                if good_t > best_t + .5: raise KeyboardInterrupt
                best_t = min(good_t, best_t)
                if not (it%15):
                    update_scal(train_X,train_y)
                    good = check(valid_X,valid_y)[0]
                    print it, good, check(train_X,train_y)[0]
                    sys.stdout.flush()
                    if good < best:
                        best_it, best, v_train[:], v_valid[:], v_test[:] = (
                            it, good, predict(train_X)[0], predict(valid_X)[0], predict(test_X)[0])
    except KeyboardInterrupt: pass
    print "switching"
    try:
        best_it = it
        momentum.set_value(.9)
        learning_rate.set_value(.03)
        for i in xrange(1000):
            for j, X_m, y_m in minibatches:
                it += 1
                if it > best_it + 200: raise KeyboardInterrupt
                #(update_ada if i<10 else update)(X_m, y_m)
                #momentum.set_value(1-.5/it)
                good_t = update(X_m, y_m)[0]
                if good_t > best_t + .5: raise KeyboardInterrupt
                best_t = min(good_t, best_t)
                if not (it%15):
                    update_scal(train_X,train_y)
                    good = check(valid_X,valid_y)[0]
                    print it, good, check(train_X,train_y)[0]
                    sys.stdout.flush()
                    if good < best:
                        best_it, best, v_train[:], v_valid[:], v_test[:] = (
                            it, good, predict(train_X)[0], predict(valid_X)[0], predict(test_X)[0])
    except KeyboardInterrupt: pass
        # print update(X,xrange(75000),xrange(75000,95674),y)
    
    # 75
    f.flush()


