sur.reweight.noDb = function(w, sl.train, osl.test, Di){ 
  # w: weight given by other training sample given by rf or np: length n.train
  # Di is for ind softlabel
  # Db is for bw softlabel
  # osl.test = one softlabel from X.test
  # print(w)
  n.train = length(sl.train)
  # -------indicator method------- #
  ind = (abs(osl.test - sl.train) < Di)
  
  # while(sum(ind) == 0){
  #   Di = Di + 0.05
  #   ind = (abs(osl.test - sl.train) < Di)
  #   # print(ind)
  # }
  
  # if (sum(ind) == 0){
  #   ind = as.logical(rep(1, n.train))
  # }
  
  # calculate the rule of thumb Db
  sl.train.truncated = sl.train[ind]
  Db = 1.06*sd(sl.train.truncated)*length(sl.train.truncated)^(-1/5) # rule of thumb Db
  # print(Db)
  while(is.na(Db)){ # Db = NA meaning that there are only 1 element != T in the ind
    Di = Di + 1
    ind = abs(osl.test - sl.train) < Di
    sl.train.truncated = sl.train[ind]
    Db = 1.06*sd(sl.train.truncated)*length(sl.train.truncated)^(-1/5) # rule of thumb Db
  }
  
  if(Db < 1e-5){ # Db = 0 meaning that the sl.train.truncated have the same value for each element
    Db = 1
  }
  # if (Db == 0 | is.na(Db)){
  #   Db = 1
  # }
  
  # if (sum(ind*w) == 0){
  #   # new.w = rep(0, n.train) #old
  #   # w = rep(0, n.train)
  #   w = w
  # } else{
  #   logw = log(ind*w)
  #   maxlw = max(logw)
  #   logsumw = maxlw+log(sum(exp(logw - maxlw)))
  #   w = exp(logw - logsumw)
  #   # new.w = ind*w/sum(ind*w) 
  # }
  
  if (sum(ind*w) > 1e-5){
    logw = log(ind*w)
    maxlw = max(logw)
    logsumw = maxlw+log(sum(exp(logw - maxlw)))
    w = exp(logw - logsumw)
    # new.w = ind*w/sum(ind*w) 
  }
  # #-----------------------------#
  
  #-----------reweighting method----------#
  dp = osl.test - sl.train
  p.w = compweight(Db, t(dp))
  
  # if (sum(p.w*w) == 0){
  #   # new.w = rep(0, n.train) #old
  #   # w = rep(0, n.train)
  #   w = rep(0, n.train)
  # } else{
  #   logw = log(p.w*w)
  #   maxlw = max(logw)
  #   logsumw = maxlw+log(sum(exp(logw - maxlw)))
  #   # new.w = exp(logw - logsumw) #old
  #   w = exp(logw - logsumw)
  #   # new.w = p.w*w/sum(p.w*w)
  # }
  
  if (sum(p.w*w) > 1e-5){
    logw = log(p.w*w)
    maxlw = max(logw)
    logsumw = maxlw+log(sum(exp(logw - maxlw)))
    # new.w = exp(logw - logsumw) #old
    w = exp(logw - logsumw)
    # new.w = p.w*w/sum(p.w*w)
  }
  # -------------------------------------- #

  # return(new.w) #old
  return(w)
}

SlRf.weight.noDb = function(wrf.list, Y.train, sl.train, sl.test, Di){ 
  # written on 8.8 compute the newweight combined with the info from softlabel kernel with given Di Db
  n.test = length(sl.test)
  # print(wrf.list[[1]])
  newweight = lapply(1:n.test, function(x) sur.reweight.noDb(wrf.list[[x]], sl.train, sl.test[x], Di))# newweight is a list of length n.test, each element is a vector of length n.train
  Yhat.test = sapply(1:n.test, function(x) sum(newweight[[x]]*Y.train))
  return(list(Yhat = Yhat.test, w.list = newweight))
}

Sl.weight.noDb = function(w.list, Y.train, sl.train, sl.val, Di){ 
  myresult = SlRf.weight.noDb(w.list, Y.train, sl.train, sl.val, Di)
  Yhat.val = myresult$Yhat
  rw.list = myresult$w.list
  return(list(Yhat = Yhat.val, w.list = rw.list))
}

# compute the weight.list given X.train and X.val #
slnp.noDb = function(X.train, Y.train, sl.train, X.val, Y.val, sl.val, Di){
  # set.seed(1010)
  # ml.rf = randomForest(x = X.train, y = Y.train, keep.inbag = T, ntree = 100, maxnodes = 16)
  # wrf.list = rf.weight(ml.rf, X.train, X.val)
  n.val = dim(X.val)[1]
  n.train = dim(X.train)[1]
  
  # constant weight list
  wrf.list = lapply(1:n.val, function(x) rep(1, n.train))
  
  myresult = SlRf.weight.noDb(wrf.list, Y.train, sl.train, sl.val, Di)
  Yhat.val = myresult$Yhat
  rwrf.list = myresult$w.list
  mse.val = mean((Yhat.val - Y.val)^2)
  return(list(Yhat = Yhat.val, mse = mse.val, w.list = rwrf.list))
}


rfguided.slnp.noDb = function(X.train, Y.train, sl.train, X.val, Y.val, sl.val, Di){
  # set.seed(1010)
  # ml.rf = randomForest(x = X.train, y = Y.train, keep.inbag = T, ntree = 100, maxnodes = 16)
  ml.rf = randomForest(x = X.train, y = Y.train, keep.inbag = T, ntree = 100)
  wrf.list = rf.weight(ml.rf, X.train, X.val)
  myresult = SlRf.weight.noDb(wrf.list, Y.train, sl.train, sl.val, Di)
  Yhat.val = myresult$Yhat
  rwrf.list = myresult$w.list
  mse.val = mean((Yhat.val - Y.val)^2)
  return(list(Yhat = Yhat.val, mse = mse.val, rwrf.list = rwrf.list))
}

cv.rfguided.slnp.noDb = function(label, X, Y, Sl, Di.vec, nfolds){
  # n = dim(X)[1]
  # set.seed(1011)
  flds = createFolds(label, k = nfolds, list = TRUE, returnTrain = FALSE)
  # sl = predsoftlabel(ordin.ml, X, status.coef)
  lDi = length(Di.vec)
  
  # mse.mtx = matrix(0, nrow = lDb, ncol = lDi)
  mse.mtx = list()
  for (k in 1:nfolds){
    # for np training #
    X.train = X[unlist(flds[-k]), ]
    X.val = X[unlist(flds[k]), ]
    Y.train = Y[unlist(flds[-k])]
    Y.val = Y[unlist(flds[k])]
    Sl.train = Sl[unlist(flds[-k])]
    Sl.val = Sl[unlist(flds[k])]
    
    n.train = dim(X.train)[1]
    n.val = dim(X.val)[1]
    # p = dim(X.train)[2]
    
    # lDb by lDi matrix
    # mse.mtx[[k]] = sapply(1:lDi, function(y) sapply(1:lDb, function(x) rfguided.slnp(X.train, Y.train, Sl.train, X.val, Y.val, Sl.val, Di.vec[y], Db.vec[x])$mse)) 
    mse.mtx[[k]] = sapply(Di.vec, function(x) rfguided.slnp.noDb(X.train, Y.train, Sl.train, X.val, Y.val, Sl.val, x)$mse)
    # print(mse.mtx[[k]])
  }
  
  # mse.array = array(unlist(mse.mtx), dim = c(lDb, lDi, nfolds))
  # measure.mtx = apply(mse.array, 3, mean)
  
  # min.id = which.min(measure.mtx)
  # min.id.Db = (min.id - 1)%%lDb + 1
  # min.id.Di = ceiling(min.id/lDb)
  # 
  # Db.selected = Db.vec[min.id.Db]
  # Di.selected = Di.vec[min.id.Di]
  
  mse.vec = apply(do.call(rbind, mse.mtx), 2, mean)
  Di.selected = Di.vec[which.min(mse.vec)]
  return(list(Di = Di.selected, mse.vec = mse.vec))
}

cv.SlRfPen.noDb = function(label, X, Y, lambda.vec, alpha, nfolds, sl, Di.vec){ 
  flds = createFolds(label, k = nfolds, list = TRUE, returnTrain = FALSE)
  
  llam = length(lambda.vec)
  lDi = length(Di.vec)
  
  mse.list = list()
  for (k in 1:nfolds){
    X.train = X[unlist(flds[-k]), ]
    X.val = X[unlist(flds[k]), ]
    Y.train = Y[unlist(flds[-k])]
    Y.val = Y[unlist(flds[k])]
    sl.train = sl[unlist(flds[-k])]
    sl.val = sl[unlist(flds[k])]
    ml.rf = randomForest(x = X.train, y = Y.train, keep.inbag = T, ntree = 100)
    wrf.list = rf.weight(ml.rf, X.train, X.val)
    mse.list[[k]] = list()
    for (i in 1:lDi){
      # compute new weights
      newweight.list = SlRf.weight.noDb(wrf.list, Y.train, sl.train, sl.val, Di.vec[i])$w.list
      Yhat.mtx = penalized.origin.method(X.train, Y.train, X.val, newweight.list, lambda.vec, alpha)$Yhat
      mse.list[[k]][[i]] = apply(Yhat.mtx - Y.val, 2, function(x) mean(x^2))
    }
    mse.list[[k]] = do.call(rbind, mse.list[[k]])
  }
  
  mse.array = array(unlist(mse.list), dim = c(lDi, llam, nfolds))
  measure.mtx = apply(mse.array, c(1,2), function(x) mean(x, na.rm = T))
  
  min.id = which.min(measure.mtx)
  min.id.Di = (min.id - 1)%%lDi + 1
  min.id.lam = ceiling(min.id/lDi)
  # 
  Di.selected = Di.vec[min.id.Di]
  lam.selected = lambda.vec[min.id.lam]
  
  return(list(Di = Di.selected, lambda = lam.selected))
}

# --------------------------------- no rf -------------------------------------- # 
cv.SlPen.noDb = function(label, X, Y, lambda.vec, alpha, nfolds, sl, Di.vec){ 
  flds = createFolds(label, k = nfolds, list = TRUE, returnTrain = FALSE)
  llam = length(lambda.vec)
  lDi = length(Di.vec)
  # mse.noRf.list = list()
  corr.noRf.list = list()
  for (k in 1:nfolds){
    # for np training #
    X.train = X[unlist(flds[-k]), ]
    X.val = X[unlist(flds[k]), ]
    Y.train = Y[unlist(flds[-k])]
    Y.val = Y[unlist(flds[k])]
    sl.train = sl[unlist(flds[-k])]
    sl.val = sl[unlist(flds[k])]
    # mse.noRf.list[[k]] = list()
    corr.noRf.list[[k]] = list()
    for (i in 1:lDi){
      newweight.list = slnp.noDb(X.train, Y.train, sl.train, X.val, Y.val, sl.val, Di.vec[i])$w.list
      Yhat.mtx = penalized.origin.method(X.train, Y.train, X.val, newweight.list, lambda.vec, alpha)$Yhat
      # mse.noRf.list[[k]][[i]] = apply(Yhat.mtx - Y.val, 2, function(x) mean(x^2))
      corr.noRf.list[[k]][[i]] = apply(Yhat.mtx, 2, function(x) cor(Y.val, x))
    }
    # mse.noRf.list[[k]] = do.call(rbind, mse.noRf.list[[k]])
    corr.noRf.list[[k]] = do.call(rbind, corr.noRf.list[[k]])
  }
  # mse.noRf.array = array(unlist(mse.noRf.list), dim = c(lDi, llam, nfolds))
  corr.noRf.array = array(unlist(corr.noRf.list), dim = c(lDi, llam, nfolds))
  
  # measure.noRf.mtx = apply(mse.noRf.array, c(1,2), mean)
  measure.noRf.mtx = apply(corr.noRf.array, c(1,2), mean)
  
  # min.id = which.min(measure.noRf.mtx)
  # min.id.Di = (min.id - 1)%%lDi + 1
  # min.id.lam = ceiling(min.id/lDi)
  # Di.selected = Di.vec[min.id.Di]
  # lam.selected = lambda.vec[min.id.lam]
  
  max.id = which.max(measure.noRf.mtx)
  max.id.Di = (max.id - 1)%%lDi + 1
  max.id.lam = ceiling(max.id/lDi)
  Di.selected = Di.vec[max.id.Di]
  lam.selected = lambda.vec[max.id.lam]

  return(list(Di = Di.selected, lambda = lam.selected))
}

# main function include determining whether to include RF
cv.bothPen.noDb = function(label, X, Y, lambda.vec, alpha, nfolds, sl, Di.vec){ 
  # set.seed(1011)
  flds = createFolds(label, k = nfolds, list = TRUE, returnTrain = FALSE)
  
  llam = length(lambda.vec)
  lDi = length(Di.vec)
  
  mse.Rf.list = list()
  mse.noRf.list = list()
  for (k in 1:nfolds){
    # for np training #
    X.train = X[unlist(flds[-k]), ]
    X.val = X[unlist(flds[k]), ]
    Y.train = Y[unlist(flds[-k])]
    Y.val = Y[unlist(flds[k])]
    sl.train = sl[unlist(flds[-k])]
    sl.val = sl[unlist(flds[k])]
    # set.seed(1010)
    ml.rf = randomForest(x = X.train, y = Y.train, keep.inbag = T, ntree = 100)
    wrf.list = rf.weight(ml.rf, X.train, X.val)
    mse.Rf.list[[k]] = list()
    mse.noRf.list[[k]] = list()
    for (i in 1:lDi){
      # compute new weights
      newweight.list = SlRf.weight.noDb(wrf.list, Y.train, sl.train, sl.val, Di.vec[i])$w.list
      Yhat.mtx = penalized.origin.method(X.train, Y.train, X.val, newweight.list, lambda.vec, alpha)$Yhat
      mse.Rf.list[[k]][[i]] = apply(Yhat.mtx - Y.val, 2, function(x) mean(x^2))
      
      newweight.list = slnp.noDb(X.train, Y.train, sl.train, X.val, Y.val, sl.val, Di.vec[i])$w.list
      Yhat.mtx = penalized.origin.method(X.train, Y.train, X.val, newweight.list, lambda.vec, alpha)$Yhat
      mse.noRf.list[[k]][[i]] = apply(Yhat.mtx - Y.val, 2, function(x) mean(x^2))
    }
    mse.Rf.list[[k]] = do.call(rbind, mse.Rf.list[[k]])
    mse.noRf.list[[k]] = do.call(rbind, mse.noRf.list[[k]])
  }
  
  mse.Rf.array = array(unlist(mse.Rf.list), dim = c(lDi, llam, nfolds))
  measure.Rf.mtx = apply(mse.Rf.array, c(1,2), function(x) mean(x, na.rm = T))
  sd.Rf.mtx = apply(mse.Rf.array, c(1,2), function(x) sd(x, na.rm = T))
  
  mse.noRf.array = array(unlist(mse.noRf.list), dim = c(lDi, llam, nfolds))
  measure.noRf.mtx = apply(mse.noRf.array, c(1,2), mean)
  sd.noRf.mtx = apply(mse.noRf.array, c(1,2), function(x) sd(x, na.rm = T))
  
  id.which = which.min(c(min(measure.Rf.mtx), min(measure.noRf.mtx))) # 1 stands for Rf 2 stands for noRf
  if (id.which == 1){
    measure.mtx = measure.Rf.mtx
    sd.mtx = sd.Rf.mtx
  }else{
    measure.mtx = measure.noRf.mtx
    sd.mtx = sd.noRf.mtx
  }
  min.id = which.min(measure.mtx)
  min.id.Di = (min.id - 1)%%lDi + 1
  min.id.lam = ceiling(min.id/lDi)
  
  # # -----new 1se----- 
  # min.1se = sd.mtx[min.id.Di, min.id.lam]
  # TF.mtx = measure.mtx < (min(measure.mtx)+min.1se) # indicator of whether the measure is within 1se
  # # search the largest lambda within the measure.mtx+1se
  # flag = 0
  # i_lam = 0
  # while (flag == 0){
  #   i_lam = i_lam + 1
  #   if (sum(TF.mtx[,i_lam])!=0){
  #     flag = i_lam
  #   }
  # }
  # # ----------------- 
  
  # min
  Di.selected = Di.vec[min.id.Di]
  lam.selected = lambda.vec[min.id.lam]
  
  # 1se
  # lam.selected = lambda.vec[flag]
  # Di.selected = Di.vec[which.min(measure.mtx[,flag])]
  
  return(list(Di = Di.selected, lambda = lam.selected, id.which = id.which))
}

cv.bothPen.noDb.nolambda = function(label, X, Y, alpha, nfolds, sl, Di.vec){ 
  # set.seed(1011)
  flds = createFolds(label, k = nfolds, list = TRUE, returnTrain = FALSE)
  
  #llam = length(lambda.vec)
  llam = 100
  lDi = length(Di.vec)
  
  mse.Rf.list = list()
  mse.noRf.list = list()
  for (k in 1:nfolds){
    # for np training #
    X.train = X[unlist(flds[-k]), ]
    X.val = X[unlist(flds[k]), ]
    Y.train = Y[unlist(flds[-k])]
    Y.val = Y[unlist(flds[k])]
    sl.train = sl[unlist(flds[-k])]
    sl.val = sl[unlist(flds[k])]
    # set.seed(1010)
    ml.rf = randomForest(x = X.train, y = Y.train, keep.inbag = T, ntree = 100)
    wrf.list = rf.weight(ml.rf, X.train, X.val)
    mse.Rf.list[[k]] = list()
    mse.noRf.list[[k]] = list()
    for (i in 1:lDi){
      # compute new weights
      newweight.list = SlRf.weight.noDb(wrf.list, Y.train, sl.train, sl.val, Di.vec[i])$w.list
      # Yhat.mtx = penalized.origin.method(X.train, Y.train, X.val, newweight.list, lambda.vec, alpha)$Yhat
      Yhat.mtx = penalized.origin.method.nolambda(X.train, Y.train, X.val, newweight.list, alpha)$Yhat
      
      mse.Rf.list[[k]][[i]] = apply(Yhat.mtx - Y.val, 2, function(x) mean(x^2))
      
      newweight.list = slnp.noDb(X.train, Y.train, sl.train, X.val, Y.val, sl.val, Di.vec[i])$w.list
      # Yhat.mtx = penalized.origin.method(X.train, Y.train, X.val, newweight.list, lambda.vec, alpha)$Yhat
      Yhat.mtx = penalized.origin.method.nolambda(X.train, Y.train, X.val, newweight.list, alpha)$Yhat
      mse.noRf.list[[k]][[i]] = apply(Yhat.mtx - Y.val, 2, function(x) mean(x^2))
    }
    mse.Rf.list[[k]] = do.call(rbind, mse.Rf.list[[k]])
    mse.noRf.list[[k]] = do.call(rbind, mse.noRf.list[[k]])
  }
  
  mse.Rf.array = array(unlist(mse.Rf.list), dim = c(lDi, llam, nfolds))
  measure.Rf.mtx = apply(mse.Rf.array, c(1,2), function(x) mean(x, na.rm = T))
  sd.Rf.mtx = apply(mse.Rf.array, c(1,2), function(x) sd(x, na.rm = T))
  
  mse.noRf.array = array(unlist(mse.noRf.list), dim = c(lDi, llam, nfolds))
  measure.noRf.mtx = apply(mse.noRf.array, c(1,2), mean)
  sd.noRf.mtx = apply(mse.noRf.array, c(1,2), function(x) sd(x, na.rm = T))
  
  id.which = which.min(c(min(measure.Rf.mtx), min(measure.noRf.mtx))) # 1 stands for Rf 2 stands for noRf
  if (id.which == 1){
    measure.mtx = measure.Rf.mtx
    sd.mtx = sd.Rf.mtx
  }else{
    measure.mtx = measure.noRf.mtx
    sd.mtx = sd.noRf.mtx
  }
  min.id = which.min(measure.mtx)
  min.id.Di = (min.id - 1)%%lDi + 1
  min.id.lam = ceiling(min.id/lDi)
  
  # # -----new 1se----- 
  # min.1se = sd.mtx[min.id.Di, min.id.lam]
  # TF.mtx = measure.mtx < (min(measure.mtx)+min.1se) # indicator of whether the measure is within 1se
  # # search the largest lambda within the measure.mtx+1se
  # flag = 0
  # i_lam = 0
  # while (flag == 0){
  #   i_lam = i_lam + 1
  #   if (sum(TF.mtx[,i_lam])!=0){
  #     flag = i_lam
  #   }
  # }
  # # ----------------- 
  
  # min
  Di.selected = Di.vec[min.id.Di]
  #lam.selected = lambda.vec[min.id.lam]
  
  # 1se
  # lam.selected = lambda.vec[flag]
  # Di.selected = Di.vec[which.min(measure.mtx[,flag])]
  
  return(list(Di = Di.selected, lambda.id = min.id.lam, id.which = id.which))
}

cv.ordinlog.both.noDb = function(label, X, Y, lam.vec, alpha, initial.x, nfolds, lDi = 20){
  flds = createFolds(label, k = nfolds, list = TRUE, returnTrain = FALSE)
  llam = length(lam.vec)
  mse.Rf.list = list()
  mse.noRf.list = list()
  
  cl = makeCluster(4) # number of cores you can use
  registerDoParallel(cl)
  
  for (i in 1:nfolds){
    X.train = X[unlist(flds[-i]), ]
    X.val = X[unlist(flds[i]), ]
    Y.train = Y[unlist(flds[-i])]
    Y.val = Y[unlist(flds[i])]
    label.train = label[unlist(flds[-i])]
    label.val = label[unlist(flds[i])]
    
    n.train = dim(X.train)[1]
    n.val = dim(X.val)[1]
    
    #      ml.list = apply(as.matrix(lam.vec), 1, function(x) ordin.logistic.en(label.train, X.train, x, alpha, initial.x))
    
    ml.list = foreach(lam=lam.vec, .export = c("ordin.logistic.en", "penalike.en", "myphi", "mygrad.en")) %dopar% {
      ordin.logistic.en(label.train, X.train, lam, alpha, initial.x)
    }
    Slhat.train.list = lapply(ml.list, function(ml) X.train%*%ml$w)
    Slhat.val.list = lapply(ml.list, function(ml) X.val%*%ml$w)
    
    # new detect outliers and delete outliers
    Slhat.train.extvalues.list = lapply(Slhat.train.list, function(sl) boxplot(sl, plot = F)$stats[c(1,5),1])
    Slhat.train.list = lapply(1:length(Slhat.train.list), function(ix){
      Slhat.train.list[[ix]][Slhat.train.list[[ix]] > Slhat.train.extvalues.list[[ix]][2]] = Slhat.train.extvalues.list[[ix]][2]
      Slhat.train.list[[ix]][Slhat.train.list[[ix]] < Slhat.train.extvalues.list[[ix]][1]] = Slhat.train.extvalues.list[[ix]][1]
      Slhat.train.list[[ix]]
    })
    
    Slhat.val.extvalues.list = lapply(Slhat.val.list, function(sl) boxplot(sl, plot = F)$stats[c(1,5),1])
    Slhat.val.list = lapply(1:length(Slhat.val.list), function(ix){
      Slhat.val.list[[ix]][Slhat.val.list[[ix]] > Slhat.val.extvalues.list[[ix]][2]] = Slhat.val.extvalues.list[[ix]][2]
      Slhat.val.list[[ix]][Slhat.val.list[[ix]] < Slhat.val.extvalues.list[[ix]][1]] = Slhat.val.extvalues.list[[ix]][1]
      Slhat.val.list[[ix]]
    })
    
    ml.rf = randomForest(x = X.train, y = Y.train, keep.inbag = T, ntree = 100)
    wrf.list = rf.weight(ml.rf, X.train, X.val)
    w.list = lapply(1:n.val, function(x) rep(1, n.train))
    
    # mse.Rf.list[[i]] = list()
    # mse.noRf.list[[i]] = list()
    
    mse.list = foreach(j = seq(1,llam), .export = c("get.mse.vec", "SlRf.weight.noDb", "Sl.weight.noDb", "sur.reweight.noDb", "compweight")) %dopar% {
      get.mse.vec(lDi, Slhat.train.list[[j]], Slhat.val.list[[j]], Y.train, Y.val, wrf.list, w.list)
    }
    mse.array = array(unlist(mse.list), dim = c(lDi, 2, llam))
    mse.array = aperm(mse.array, c(1,3,2))
    
    mse.Rf.list[[i]] = mse.array[,,1] # lDi, llam
    mse.noRf.list[[i]] = mse.array[,,2]
    # for (j in 1:length(lam.vec)){
    #   Di.vec = seq(sd(Slhat.train.list[[j]])/5, sd(Slhat.train.list[[j]])*2, length.out = lDi)
    #   mse.Rf.list[[i]][[j]] = list()
    #   mse.noRf.list[[i]][[j]] = list()
    #   for (d in 1:lDi){
    #     Yhat.SlRf = SlRf.weight.noDb(wrf.list, Y.train, Slhat.train.list[[j]], Slhat.val.list[[j]], Di.vec[d])$Yhat
    #     Yhat.Sl = Sl.weight.noDb(w.list, Y.train, Slhat.train.list[[j]], Slhat.val.list[[j]], Di.vec[d])$Yhat
    #     mse.Rf.list[[i]][[j]][[d]] = mean((Yhat.SlRf - Y.val)^2)
    #     mse.noRf.list[[i]][[j]][[d]] = mean((Yhat.Sl - Y.val)^2)
    #   }
    #   mse.Rf.list[[i]][[j]] = do.call(c, mse.Rf.list[[i]][[j]]) # vector of length lDi
    #   mse.noRf.list[[i]][[j]] = do.call(c, mse.noRf.list[[i]][[j]])
    # }
    # mse.Rf.list[[i]] = do.call(cbind, mse.Rf.list[[i]]) # 20 by llam
    # mse.noRf.list[[i]] = do.call(cbind, mse.noRf.list[[i]])
  }
  stopCluster(cl)
  mse.Rf.array = array(unlist(mse.Rf.list), dim = c(lDi, llam, nfolds))
  mse.noRf.array = array(unlist(mse.noRf.list), dim = c(lDi, llam, nfolds))
  measure.Rf.mtx = apply(mse.Rf.array, c(1,2), function(x) mean(x, na.rm = T))
  measure.noRf.mtx = apply(mse.noRf.array, c(1,2), function(x) mean(x, na.rm = T))
  sd.Rf.mtx = apply(mse.Rf.array, c(1,2), function(x) sd(x, na.rm = T))
  sd.noRf.mtx = apply(mse.noRf.array, c(1,2), function(x) sd(x, na.rm = T))
  
  id.which = which.min(c(min(measure.Rf.mtx), min(measure.noRf.mtx))) # 1 stands for Rf 2 stands for noRf
  if (id.which == 1){
    measure.mtx = measure.Rf.mtx
    sd.mtx = sd.Rf.mtx
  }else{
    measure.mtx = measure.noRf.mtx
    sd.mtx = sd.noRf.mtx
  }
  min.id = which.min(measure.mtx)
  min.id.Di = (min.id - 1)%%lDi + 1
  min.id.lam = ceiling(min.id/lDi)
  
  # Di.vec = seq(sd(Slhat.train.list[[min.id.lam]])/5, sd(Slhat.train.list[[min.id.lam]])*2, length.out = lDi)
  # Di.selected = Di.vec[min.id.Di]
  lam.selected = lam.vec[min.id.lam]
  
  ordin.ml.best = ordin.logistic.en(label, X, lam.selected, alpha, initial.x)
  
  return(list(gam = lam.selected, Di.id = min.id.Di, id = id.which, ordin.ml= ordin.ml.best))
}

get.mse.vec = function(lDi, Sl.train, Sl.val, Y.train, Y.val, wrf.list, w.list){
  Di.vec = seq(sd(Sl.train)/5, sd(Sl.train)*2, length.out = lDi)
  mse.Rf.list = list()
  mse.noRf.list = list()
  for (d in 1:lDi){
    Yhat.SlRf = SlRf.weight.noDb(wrf.list, Y.train, Sl.train, Sl.val, Di.vec[d])$Yhat
    Yhat.Sl = Sl.weight.noDb(w.list, Y.train, Sl.train, Sl.val, Di.vec[d])$Yhat
    mse.Rf.list[[d]] = mean((Yhat.SlRf - Y.val)^2)
    mse.noRf.list[[d]] = mean((Yhat.Sl - Y.val)^2)
  }
  mse.Rf.vec = do.call(c, mse.Rf.list) # vector of length 20
  mse.noRf.vec = do.call(c, mse.noRf.list)
  return(list(Rf = mse.Rf.vec, noRf = mse.noRf.vec))
}












