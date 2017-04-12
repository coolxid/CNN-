import numpy as np


class ClassifierTrainer(object):
  """ The trainer class performs SGD with momentum on a cost function """
  def __init__(self):
    self.step_cache = {} # for storing velocities in momentum update

    # variables used for SMORMS3
    self.step_cache_square={}
    self.step_cache_mem={}

  def train(self, X, y, X_val, y_val,model, loss_function, reg=0.0,learning_rate=1e-2, momentum=0, learning_rate_decay=0.95,update='momentum', sample_batches=True,
            num_epochs=30,batch_size=100, acc_frequency=None,verbose=False):
   
    N = X.shape[0]
    if sample_batches:
      iterations_per_epoch =  N / batch_size # using SGD
    else:
      iterations_per_epoch = 1 # using GD
    num_iters = num_epochs * np.int(np.ceil(iterations_per_epoch))
    epoch = 0
    best_val_acc = 0.0
    best_model = {}
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    for it in range(num_iters):
      if it % 10 == 0:  print ('starting iteration ', it)

      # get batch of data
      if sample_batches:
        batch_mask = np.random.choice(N, batch_size)
        X_batch = X[batch_mask]
        y_batch = y[batch_mask]
      else:
        # no SGD used, full gradient descent
        X_batch = X
        y_batch = y

      # evaluate cost and gradient
      cost, grads = loss_function(X_batch, model, y_batch, reg)
      loss_history.append(cost)

      # perform a parameter update
      for p in model:
        # compute the parameter step
        if update == 'sgd':
          dx = -learning_rate * grads[p]
        elif update == 'momentum':
          if not p in self.step_cache: 
            self.step_cache[p] = np.zeros(grads[p].shape)
          # dx = np.zeros_like(grads[p]) # you can remove this after
          
          self.step_cache[p] = (momentum * self.step_cache[p] - learning_rate * grads[p])
          dx = self.step_cache[p]
          #####################################################################
          #                      END OF YOUR CODE                             #
          #####################################################################
        elif update == 'nestrov-momentum':
          if not p in self.step_cache: 
            self.step_cache[p] = np.zeros(grads[p].shape)
          
         
          previous=self.step_cache[p]
          self.step_cache[p] = (momentum * self.step_cache[p] - learning_rate * grads[p])
          dx = -momentum * previous + (1+momentum)*self.step_cache[p]
         
        elif update == 'rmsprop':
          decay_rate = 0.99 # you could also make this an option
          if not p in self.step_cache: 
            self.step_cache[p] = np.zeros(grads[p].shape)
          # dx = np.zeros_like(grads[p]) # you can remove this after
         
          self.step_cache[p] = decay_rate * self.step_cache[p] + (1-decay_rate) * grads[p]**2
          dx = -learning_rate * grads[p] / np.sqrt(self.step_cache[p] + 1e-8)
         
        elif update =="SMORMS3":
          if not p in self.step_cache: 
            self.step_cache[p] = np.zeros(grads[p].shape)
            pass
        else:
          raise ValueError('Unrecognized update type "%s"' % update)

        # update the parameters
        model[p] += dx

      # every epoch perform an evaluation on the validation set
      first_it = (it == 0)
      epoch_end = (it + 1) % iterations_per_epoch == 0
      acc_check = (acc_frequency is not None and it % acc_frequency == 0)
      if first_it or epoch_end or acc_check:
        if it > 0 and epoch_end:
          # decay the learning rate
          learning_rate *= learning_rate_decay
          epoch += 1

        # evaluate train accuracy
        if N > 1000:
          train_mask = np.random.choice(N, 1000)
          X_train_subset = X[train_mask]
          y_train_subset = y[train_mask]
        else:
          X_train_subset = X
          y_train_subset = y
        scores_train = loss_function(X_train_subset, model)
        y_pred_train = np.argmax(scores_train, axis=1)
        train_acc = np.mean(y_pred_train == y_train_subset)
        train_acc_history.append(train_acc)

        # evaluate val accuracy
        scores_val = loss_function(X_val, model)
        y_pred_val = np.argmax(scores_val, axis=1)
        val_acc = np.mean(y_pred_val ==  y_val)
        val_acc_history.append(val_acc)
        
        # keep track of the best model based on validation accuracy
        if val_acc > best_val_acc:
          # make a copy of the model
          best_val_acc = val_acc
          best_model = {}
          for p in model:
            best_model[p] = model[p].copy()

        # print progress if needed
        if verbose:
          print ('Finished epoch %d / %d: cost %f, train: %f, val %f, lr %e'
                 % (epoch, num_epochs, cost, train_acc, val_acc, learning_rate))

    if verbose:
      print ('finished optimization. best validation accuracy: %f' % (best_val_acc, ))
    # return the best model and the training history statistics
    return best_model, loss_history, train_acc_history, val_acc_history











