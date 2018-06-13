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
  
  # -----new 1se----- #
  min.1se = sd.mtx[min.id.Di, min.id.lam]
  TF.mtx = measure.mtx < (min(measure.mtx)+min.1se) # indicator of whether the measure is within 1se
  # search the largest lambda within the measure.mtx+1se
  flag = 0
  i_lam = 0
  while (flag == 0){
    i_lam = i_lam + 1
    if (sum(TF.mtx[,i_lam])!=0){
      flag = i_lam
    }
  }
  # ----------------- #
  
  # min
  # Di.selected = Di.vec[min.id.Di]
  # lam.selected = lambda.vec[min.id.lam]
  
  # 1se
  lam.selected = lambda.vec[flag]
  Di.selected = Di.vec[which.min(measure.mtx[,flag])]
  
  return(list(Di = Di.selected, lambda = lam.selected, id.which = id.which))
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

#########################################################################################################
gs.compweight = function(X.train, X.test, sl.train, sl.test, Di){
  # bandwidth of gs kernel computed from rule of thumb
  n.train = dim(X.train)[1]
  n.test = dim(X.test)[1]
  p = dim(X.train)[2]
  
  ind.list = lapply(sl.test, function(x) (abs(x - sl.train) < Di)) # length n.test; each element n.train by 1
  # print(ind.list)
  # bw = apply(X.train, 2, function(x) 1.06*sd(x)*length(x)^(-1/5)) # Silverman's rule of thumb for univariate X
  # bw = apply(X.train, 2, function(x) sd(x)*n.train^(-1/(p+4))) # multivariate bw from paper multibandwidth.pdf
  # bw.list = lapply(1:n.test, function(x) apply(X.train[ind.list[[x]], ], 2, function(y) ifelse(sd(y)<1e-5, 1, sd(y))*(sum(ind.list[[x]])^(-1/(p+4))) ) )
  bw.list = lapply(1:n.test, function(x) apply(X.train[ind.list[[x]], ], 2, function(y) ifelse(sd(y)<1e-5, 1, sd(y))*(sum(ind.list[[x]])^(-1/(p+4)))*(4/(p+2))^(1/(p+4))*10 ) )
  # print(bw.list)
  diff.mtx.list = diff.matrix(X.train, X.test) # length n.test
  # w.list = lapply(diff.mtx.list, function(x) compweight(bw, t(x)))
  w.list = lapply(1:n.test, function(x) compweight(bw.list[[x]], t(diff.mtx.list[[x]])))
  return(w.list)
}

gsslnp.noDb = function(X.train, Y.train, sl.train, X.test, Y.test, sl.test, Di){
  # set.seed(1010)
  # ml.rf = randomForest(x = X.train, y = Y.train, keep.inbag = T, ntree = 100, maxnodes = 16)
  # wrf.list = rf.weight(ml.rf, X.train, X.test)
  n.test = dim(X.test)[1]
  n.train = dim(X.train)[1]
  
  # constant weight list
  # wrf.list = lapply(1:n.test, function(x) rep(1, n.train))
  # gaussian weight list
  gsw.list = gs.compweight(X.train, X.test, sl.train, sl.test, Di)
  # print(gsw.list[[1]])
  
  myresult = SlRf.weight.noDb(gsw.list, Y.train, sl.train, sl.test, Di)
  
  Yhat.test = myresult$Yhat
  rwrf.list = myresult$w.list
  mse.test = mean((Yhat.test - Y.test)^2)
  return(list(Yhat = Yhat.test, mse = mse.test, w.list = rwrf.list))
}

cv.GSSlPen.noDb = function(label, X, Y, lambda.vec, alpha, nfolds, sl, Di.vec){ 
  # written on 8.8 compute and select the best lambda of local linear regression with local weights given by SlRfPen; Di Db fixed 
  # set.seed(1011)
  flds = createFolds(label, k = nfolds, list = TRUE, returnTrain = FALSE)
  
  llam = length(lambda.vec)
  lDi = length(Di.vec)
  
  mse.list = list()
  for (k in 1:nfolds){
    # for np training #
    X.train = X[unlist(flds[-k]), ]
    X.val = X[unlist(flds[k]), ]
    Y.train = Y[unlist(flds[-k])]
    Y.val = Y[unlist(flds[k])]
    sl.train = sl[unlist(flds[-k])]
    sl.val = sl[unlist(flds[k])]
    # set.seed(1010)
    # ml.rf = randomForest(x = X.train, y = Y.train, keep.inbag = T, ntree = 100, maxnodes = 16)
    # wrf.list = rf.weight(ml.rf, X.train, X.val)
    mse.list[[k]] = list()
    for (i in 1:lDi){
      # compute new weights
      # newweight.list = SlRf.weight.noDb(wrf.list, Y.train, sl.train, sl.val, Di.vec[i])$w.list
      newweight.list = gsslnp.noDb(X.train, Y.train, sl.train, X.val, Y.val, sl.val, Di.vec[i])$w.list
      # print(newweight.list)
      Yhat.mtx = penalized.origin.method(X.train, Y.train, X.val, newweight.list, lambda.vec, alpha)$Yhat
      mse.list[[k]][[i]] = apply(Yhat.mtx - Y.val, 2, function(x) mean(x^2))
    }
    mse.list[[k]] = do.call(rbind, mse.list[[k]])
  }
  mse.array = array(unlist(mse.list), dim = c(lDi, llam, nfolds))
  measure.mtx = apply(mse.array, c(1,2), mean)
  # print(mse.array)
  # print(measure.mtx)
  # c(lDb, lDi, nfolds)
  
  min.id = which.min(measure.mtx)
  min.id.Di = (min.id - 1)%%lDi + 1
  min.id.lam = ceiling(min.id/lDi)
  # 
  Di.selected = Di.vec[min.id.Di]
  lam.selected = lambda.vec[min.id.lam]
  
  return(list(Di = Di.selected, lambda = lam.selected, mse.array = mse.array, mse.mtx = measure.mtx))
}








