rfrw.surnpfit = function(wrf.list, Y.train, softlabel.train, softlabel.test, Di, Db){ 
  # softlabel.train n.train by 1
  # softlabel.test n.test by 1
  n.test = length(softlabel.test)
# wte = lapply(1:n.test, function(x) surcompweight(bw, dxte.list[[x]], softlabel.train, softlabel.test[x], Di, Db))
  wte = lapply(1:n.test, function(x) surreweight(wrf.list[[x]], softlabel.train, softlabel.test[x], Di, Db)) # wte is a list of length n.test, each element is a vector of length n.train
  Yhat.test = sapply(1:n.test, function(x) sum(wte[[x]]*Y.train))  
  return(list(Yhat.test = Yhat.test, w.test = wte))
}

surreweight = function(w, softlabel.train, osoftlabel.test, Di, Db){ 
  # w: weight given by other training sample given by rf or np: length n.train
  # Di is for ind softlabel
  # Db is for bw softlabel
  # keep one fixed
  # osoftlabel.test = one softlabel from X.test
  
  n.train = length(softlabel.train)
  # Phatit.test = sapply(1:n.train, function(x) oPhat.test[predlabel.train[x]]) # Phat.test[max label] max label from training sample
  # D1 = 1
  # #-------indicator method------#
  ind = (abs(osoftlabel.test - softlabel.train) < Di)
  # w = compweight(bw, dx)
  if (sum(ind*w) == 0){
    new.w = rep(0, n.train)
  } else{
    logw = log(ind*w)
    maxlw = max(logw)
    logsumw = maxlw+log(sum(exp(logw - maxlw)))
    w = exp(logw - logsumw)
    # new.w = ind*w/sum(ind*w)
  }
  # #-----------------------------#
  
  #-----------reweighting method----------#
  # dp = Phatit.test - maxPhat.train # n.train by 1
  dp = osoftlabel.test - softlabel.train
  p.w = compweight(Db, t(dp))
  
  # w = compweight(bw, dx)
  
  if (sum(p.w*w) == 0){
    new.w = rep(0, n.train)
  } else{
    logw = log(p.w*w)
    maxlw = max(logw)
    logsumw = maxlw+log(sum(exp(logw - maxlw)))
    new.w = exp(logw - logsumw)
    
    # new.w = p.w*w/sum(p.w*w)
  }
  # -------------------------------------- #
  return(new.w)
}

compweight = function(bw, dx){ 
  # dx: p by n.train difference matrix
  # # bw: 1 by p bandwidth vector
  
  # # -original- #
  # invH = diag(1/bw^2)
  # # ---------- #
  
  # change to ... #
  if (length(bw) >1){
    invH = diag(1/bw^2)
  } else if (length(bw) == 1){
    invH = 1/bw^2
  }
  # ------------- #
  
  
  # H = diag(bw^2)
  
  ## version 1: normally we use this
  # w = exp(-1/2*t(dx)%*%solve(H)%*%dx)
  # return(diag(w))
  
  ## version 2: but when exp(-dx) dx is too large
  ## http://stackoverflow.com/questions/5802592/dealing-with-very-small-numbers-in-r
  logw = -1/2*diag(t(dx)%*%invH%*%dx)
  # logw = -1/2*diag(t(dx)%*%solve(H)%*%dx)
  maxlw = max(logw)
  logsumw = maxlw+log(sum(exp(logw - maxlw)))
  w = exp(logw - logsumw) # this is a vector of exp(ai)/sum(exp(ai)) ; length: n.train
  
  # ----- if using method 1 for checking lb, uncomment below, otherwise if using method 2, comment below ----- #
  # max.w = max(w)
  # min.w = min(w)
  # if (abs(max.w - min.w - 1) < 1e-4 || is.na(max.w) || is.na(min.w)) {
  #   w = rep(0, length(logw))
  # }
  # ----------------------------------- #
  
  return(w)
}

#if (real.rf == 1){
rf.weight = function(ml.rf, X.train, X.test){
    # real random forest
    
    #X.train = X.list[[1]]
    #X.test = X.list[[2]]
   # pred.train = predict(ml.rf, newdata = data.frame(X.train), nodes = TRUE)
   # pred.test = predict(ml.rf, newdata = data.frame(X.test), nodes = TRUE)
    pred.train = predict(ml.rf, newdata = X.train, nodes = TRUE)
    pred.test = predict(ml.rf, newdata = X.test, nodes = TRUE)
    nodes.train = attributes(pred.train)$nodes
    nodes.test = attributes(pred.test)$nodes
    inbagtime = ml.rf$inbag # n.train by n.tree
    
    ntrees = dim(nodes.train)[2]
    n.test = dim(nodes.test)[1]
    # n.train = dim(nodes.train)[1]
    
    w.list = list()
    k.list = list()
    wrf.list = list()
    
    for (i in 1:n.test){
      w.list[[i]] = sapply(1:ntrees, function(x) as.numeric(nodes.test[i,x] == nodes.train[,x]))
      # length n.test; each element n.train by ntrees
      
      # k.list[[i]] = apply(w.list[[i]], 2, sum)
      # # length n.test; each element 1 by ntrees
      # woverk = sapply(1:ntrees, function(x) w.list[[i]][,x]/k.list[[i]][x])
      
      wtimesn = w.list[[i]]*inbagtime
      # 8.8 consider bootrap sampling
      k.list[[i]] = apply(wtimesn, 2, sum)
      # length n.test; each element 1 by ntrees
      woverk = sapply(1:ntrees, function(x) wtimesn[,x]/k.list[[i]][x])
      # n.train by ntrees
      
      wrf.list[[i]] = apply(woverk, 1, sum)/ntrees
    }
    return(wrf.list) # a list of n.test vectors of length n.train
}

cv.rfguided.slnp = function(label, X, Y, Sl, Di.vec, Db.vec, nfolds){
  # n = dim(X)[1]
  set.seed(1011)
  flds = createFolds(label, k = nfolds, list = TRUE, returnTrain = FALSE)
  # sl = predsoftlabel(ordin.ml, X, status.coef)
  lDi = length(Di.vec)
  lDb = length(Db.vec)
  
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
    
    # mse.mtx = mse.mtx + sapply(1:lDi, function(y) sapply(1:lDb, function(x) rfguided.slnp(X.train, Y.train, Sl.train, X.val, Y.val, Sl.val, Di.vec[y], Db.vec[x])$mse)) 
    # lDb by lDi matrix
    mse.mtx[[k]] = sapply(1:lDi, function(y) sapply(1:lDb, function(x) rfguided.slnp(X.train, Y.train, Sl.train, X.val, Y.val, Sl.val, Di.vec[y], Db.vec[x])$mse)) 
    # print(mse.mtx[[k]])
  }
  mse.array = array(unlist(mse.mtx), dim = c(lDb, lDi, nfolds))
  # print(mse.array)
  
  measure.mtx = apply(mse.array, c(1,2), mean)
  # print(measure.mtx)
  # measure.sd = apply(mse.array, 3, sd)
    
  # min.id = which.min(mse.mtx)
  # min.id.Db = (min.id - 1)%%lDb + 1
  # min.id.Di = ceiling(min.id/lDb)
  
  min.id = which.min(measure.mtx)
  min.id.Db = (min.id - 1)%%lDb + 1
  min.id.Di = ceiling(min.id/lDb)
  
  Db.selected = Db.vec[min.id.Db]
  Di.selected = Di.vec[min.id.Di]
  
  return(list(Di = Di.selected, Db = Db.selected))
}

rfguided.slnp = function(X.train, Y.train, sl.train, X.val, Y.val, sl.val, Di, Db){
  # set.seed(1010)
  # ml.rf = randomForest(x = X.train, y = Y.train, keep.inbag = T, ntree = 100, maxnodes = 16)
  ml.rf = randomForest(x = X.train, y = Y.train, keep.inbag = T, ntree = 100)
  # data.train = data.frame(Y = Y.train, X.train)
  # ml.rf = randomForest(Y~., data = data.train, keep.inbag = T)
  
  wrf.list = rf.weight(ml.rf, X.train, X.val)
  myresult = rfrw.surnpfit(wrf.list, Y.train, sl.train, sl.val, Di, Db)
  Yhat.val = myresult$Yhat.test
  rwrf.list = myresult$w.test
  mse.val = mean((Yhat.val - Y.val)^2)
  return(list(Yhat = Yhat.val, mse = mse.val, rwrf.list = rwrf.list))
}


origin.method = function(X.train, Y.train, X.test, wrf.list){
  n.test = dim(X.test)[1]
  # p = dim(X.test)[2]
  diff.mtx.list = diff.matrix(X.train, X.test)
  beta.matrix = sapply(1:n.test, function(x) t(lm(Y.train~diff.mtx.list[[x]], weights = wrf.list[[x]])$coef)) # (p+1) by n.test
  return(list(Yhat = beta.matrix[1,], beta.matrix = beta.matrix))
}

diff.matrix = function(X.train, X.test){
  n.test = dim(X.test)[1]
  diff.mtx.list = lapply(1:n.test, function(x) t(t(X.train) - X.test[x, ]))
  return(diff.mtx.list) # length n.test
}


penalized.origin.method = function(X.train, Y.train, X.test, wrf.list, lambda.vec, alpha){ 
  n.test = dim(X.test)[1]
  p = dim(X.test)[2]
  nlambda = length(lambda.vec)
  
  diff.mtx.list = diff.matrix(X.train, X.test)
  # beta.array = sapply(1:n.test, function(x) 
  #   if(sd(Y.train[wrf.list[[x]]!=0])==0 | length(Y.train[wrf.list[[x]]!=0])<=1) {
  #     matrix(rep(c(mean(Y.train[wrf.list[[x]]!=0]), rep(0, p)), nlambda), ncol = nlambda)}else{
  #       as.matrix(coef(glmnet(x = diff.mtx.list[[x]], y = Y.train, weights = wrf.list[[x]], alpha = alpha, lambda = lambda.vec)))
  #     }
  #   , simplify = "array") # (p+1) by nlambda by n.test
  
  beta.array = sapply(1:n.test, function(x) 
    if(sd(Y.train[wrf.list[[x]]!=0])==0 | length(Y.train[wrf.list[[x]]!=0])<=10) {
      matrix(rep(c(mean(Y.train[wrf.list[[x]]!=0]), rep(0, p)), nlambda), ncol = nlambda)}else{
        as.matrix(coef(glmnet(x = diff.mtx.list[[x]], y = Y.train, weights = wrf.list[[x]], alpha = alpha, lambda = lambda.vec)))
      }
    , simplify = "array") # (p+1) by nlambda by n.test
  
  # adding local.feature.weight for each testing sample
  beta.array = aperm(beta.array, c(1,3,2)) # (p+1) by n.test by nlambda
  # return(beta.array[1,,]) # beta.array[1,,] is matrix of dim n.test by nlambda each column returns the yhat.test corresponding to certain lambda
  return(list(Yhat = beta.array[1,,], betahat = beta.array[-1,,]))
}


cv.penalized.origin.method = function(label, X, Y, lambda.vec, alpha, nfolds){
  # X = X.list[[1]]
  # Y = Y.list[[1]]
  # label = label.list[[1]]
  # nfolds = 2
  
  # n = dim(X)[1]
  
  set.seed(1011)
  flds = createFolds(label, k = nfolds, list = TRUE, returnTrain = FALSE)
  
  # llam = length(lambda.vec)
  # mse.vec = rep(0, llam)
  measure.list = list()
  for (k in 1:nfolds){
    # for np training #
    X.train = X[unlist(flds[-k]), ]
    X.val = X[unlist(flds[k]), ]
    Y.train = Y[unlist(flds[-k])]
    Y.val = Y[unlist(flds[k])]
    
    set.seed(1010)
    ml.rf = randomForest(x = X.train, y = Y.train, keep.inbag = T, ntree = 100)
    # ml.rf = randomForest(x = X.train, y = Y.train, keep.inbag = T, ntree = 100, maxnodes = 16)
    
    wrf.list = rf.weight(ml.rf, X.train, X.val)
    Yhat.mtx = penalized.origin.method(X.train, Y.train, X.val, wrf.list, lambda.vec, alpha)$Yhat
    measure.list[[k]] = apply(Yhat.mtx - Y.val, 2, function(x) mean(x^2))
  }
  
  measure.mtx = do.call(rbind, measure.list)
  
  measure.vec = apply(measure.mtx, 2, mean)
  lambda.min = lambda.vec[which.min(measure.vec)]
  
  measure.sd = apply(measure.mtx, 2, sd)
  
  # lowerbd = min(measure.vec)-measure.sd[which.min(measure.vec)]
  upperbd = min(measure.vec) + measure.sd[which.min(measure.vec)]
  
  lambda.1se = max(lambda.vec[measure.vec <= upperbd])
  
  return(list(lambda.min = lambda.min, lambda.1se = lambda.1se, min.sd = measure.sd[which.min(measure.vec)], measure.vec = measure.vec))
}

cv.SlRfPen = function(label, X, Y, lambda.vec, alpha, nfolds, sl, Di, Db){ 
  # written on 8.8 compute and select the best lambda of local linear regression with local weights given by SlRfPen; Di Db fixed 
  # X = X.list[[1]]
  # Y = Y.list[[1]]
  # label = label.list[[1]]
  # nfolds = 2
  
  # n = dim(X)[1]
  set.seed(1011)
  flds = createFolds(label, k = nfolds, list = TRUE, returnTrain = FALSE)
  
  llam = length(lambda.vec)
  # mse.vec = rep(0, llam)
  mse.list = list()
  for (k in 1:nfolds){
    # for np training #
    X.train = X[unlist(flds[-k]), ]
    X.val = X[unlist(flds[k]), ]
    Y.train = Y[unlist(flds[-k])]
    Y.val = Y[unlist(flds[k])]
    sl.train = sl[unlist(flds[-k])]
    sl.val = sl[unlist(flds[k])]
    # n.train = dim(X.train)[1]
    # n.val = dim(X.val)[1]
    # p = dim(X.train)[2]
    
    set.seed(1010)
    # ml.rf = randomForest(x = X.train, y = Y.train, keep.inbag = T, ntree = 100, maxnodes = 16)
    ml.rf = randomForest(x = X.train, y = Y.train, keep.inbag = T, ntree = 100)
    wrf.list = rf.weight(ml.rf, X.train, X.val)
    # compute new weights 
    # rfrw.surnpfit(wrf.list, Y.train, sl.train, sl.val, Di, Db)
    newweight.list = SlRf.weight(wrf.list, sl.train, sl.val, Di, Db)
    # print(newweight.list)
    Yhat.mtx = penalized.origin.method(X.train, Y.train, X.val, newweight.list, lambda.vec, alpha)$Yhat
    # mse.vec = mse.vec + apply(Yhat.mtx - Y.val, 2, function(x) mean(x^2)) 8.23
    # mse.vec = mse.vec + apply(Yhat.mtx - Y.val, 2, function(x) mean(abs(x)))
    mse.list[[k]] = apply(Yhat.mtx - Y.val, 2, function(x) mean(x^2))
  }
  # mse.vec = do.call(rbind, mse.list)
  mse.mtx = do.call(rbind, mse.list)
  # print(mse.mtx)
  mse.vec = apply(mse.mtx, 2, mean)
  # print(mse.vec)
  lambda.selected = lambda.vec[which.min(mse.vec)]
  return(lambda.selected)
}

SlRf.weight = function(wrf.list, sl.train, sl.test, Di, Db){ 
  # written on 8.8 compute the newweight combined with the info from softlabel kernel with given Di Db
  n.test = length(sl.test)
  newweight = lapply(1:n.test, function(x) surreweight(wrf.list[[x]], sl.train, sl.test[x], Di, Db)) 
  # newweight is a list of length n.test, each element is a vector of length n.train
  return(newweight)
}


# -----------------------------slnp choosing the best Di and Db------------------------------ #
cv.slnp = function(label, X, Y, Sl, Di.vec, Db.vec, nfolds){
  # n = dim(X)[1]
  set.seed(1011)
  flds = createFolds(label, k = nfolds, list = TRUE, returnTrain = FALSE)
  lDi = length(Di.vec)
  lDb = length(Db.vec)
  
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
    
    # lDb by lDi matrix
    mse.mtx[[k]] = sapply(1:lDi, function(y) sapply(1:lDb, function(x) slnp(X.train, Y.train, Sl.train, X.val, Y.val, Sl.val, Di.vec[y], Db.vec[x])$mse)) 
    # print(mse.mtx[[k]])
  }
  mse.array = array(unlist(mse.mtx), dim = c(lDb, lDi, nfolds))
  # print(mse.array)
  
  measure.mtx = apply(mse.array, c(1,2), mean)
  # print(measure.mtx)
  # measure.sd = apply(mse.array, 3, sd)
  
  # min.id = which.min(mse.mtx)
  # min.id.Db = (min.id - 1)%%lDb + 1
  # min.id.Di = ceiling(min.id/lDb)
  
  min.id = which.min(measure.mtx)
  min.id.Db = (min.id - 1)%%lDb + 1
  min.id.Di = ceiling(min.id/lDb)
  
  Db.selected = Db.vec[min.id.Db]
  Di.selected = Di.vec[min.id.Di]
  
  return(list(Di = Di.selected, Db = Db.selected))
}

slnp = function(X.train, Y.train, sl.train, X.val, Y.val, sl.val, Di, Db){
  n.val = dim(X.val)[1]
  n.train = dim(X.train)[1]
  
  # constant weight list
  wrf.list = lapply(1:n.val, function(x) rep(1, n.train))
  
  myresult = rfrw.surnpfit(wrf.list, Y.train, sl.train, sl.val, Di, Db)
  Yhat.val = myresult$Yhat.test
  rwrf.list = myresult$w.test
  mse.val = mean((Yhat.val - Y.val)^2)
  return(list(Yhat = Yhat.val, mse = mse.val, rwrf.list = rwrf.list))
}

cv.SlPen = function(label, X, Y, lambda.vec, alpha, nfolds, sl, Di, Db){ 
  # written on 8.8 compute and select the best lambda of local linear regression with local weights given by SlRfPen; Di Db fixed 
  
  # n = dim(X)[1]
  set.seed(1011)
  flds = createFolds(label, k = nfolds, list = TRUE, returnTrain = FALSE)
  
  llam = length(lambda.vec)
  # mse.vec = rep(0, llam)
  mse.list = list()
  for (k in 1:nfolds){
    # for np training #
    X.train = X[unlist(flds[-k]), ]
    X.val = X[unlist(flds[k]), ]
    Y.train = Y[unlist(flds[-k])]
    Y.val = Y[unlist(flds[k])]
    sl.train = sl[unlist(flds[-k])]
    sl.val = sl[unlist(flds[k])]
    
    newweight.list = slnp(X.train, Y.train, sl.train, X.val, Y.val, sl.val, Di, Db)$rwrf.list
    
    Yhat.mtx = penalized.origin.method(X.train, Y.train, X.val, newweight.list, lambda.vec, alpha)$Yhat
    # mse.vec = mse.vec + apply(Yhat.mtx - Y.val, 2, function(x) mean(x^2)) 8.23
    # mse.vec = mse.vec + apply(Yhat.mtx - Y.val, 2, function(x) mean(abs(x)))
    mse.list[[k]] = apply(Yhat.mtx - Y.val, 2, function(x) mean(x^2))
  }
  # mse.vec = do.call(rbind, mse.list)
  mse.mtx = do.call(rbind, mse.list)
  # print(mse.mtx)
  mse.vec = apply(mse.mtx, 2, mean)
  # print(mse.vec)
  lambda.selected = lambda.vec[which.min(mse.vec)]
  return(lambda.selected)
}
# ----------------------------------------------------------------------------------------------- #

getlambda.vec = function(X.train, Y.train, X.test, wrf.list, alpha, local.feature.weight = matrix(1, nrow = nrow(X.test), ncol = ncol(X.test))){
  n.test = dim(X.test)[1]
  p = dim(X.test)[2]
  diff.mtx.list = diff.matrix(X.train, X.test)
  
  lambda.list = list()
  num = 0
  for (i in 1:n.test){
    if(sd(Y.train[wrf.list[[i]]!=0])!=0 & length(Y.train[wrf.list[[i]]!=0])>1) {
      num = num+1
      fit = glmnet(x = diff.mtx.list[[i]]%*%diag(local.feature.weight[i,]), y = Y.train, weights = wrf.list[[i]], alpha = alpha)
      # print(fit)
      lambda.list[[num]] = fit$lambda
    }
  }
  lambda.mtx = do.call(cbind, lambda.list)
  lambda.vec = apply(lambda.mtx, 1, mean)
  # print(lambda.vec0)
  # llam = length(lambda.vec0)
  # lambda.vec = rev(2^(seq(log2(lambda.vec0[llam]), log2(lambda.vec0[1]), length.out = llam)))
  # print(cbind(rev(2^(seq(log2(lambda.vec0[llam]), log2(lambda.vec0[1]), length.out = llam))), rev(exp(seq(log(lambda.vec0[llam]), log(lambda.vec0[1]), length.out = llam)))))
  # print(log2(lambda.vec0[llam]))
  # print(log(lambda.vec0[llam]))
  # print(cbind(rev(2^(seq(log2(lambda.vec0[llam]), log2(lambda.vec0[1]), length.out = llam))), rev(exp(seq(log(lambda.vec0[llam]), log(lambda.vec0[1]), length.out = llam)))))
  # print(rev(seq(lambda.vec0[llam], lambda.vec0[1], length.out = llam)))
  return(list(lambda.list = lambda.list, lambda.mtx = lambda.mtx, lambda.vec = lambda.vec))
}

predict.penalized.origin.method = function(X.train, Y.train, X.test, wrf.list, lambda, alpha, local.feature.weight = matrix(1, nrow = nrow(X.test), ncol = ncol(X.test))){
  n.test = dim(X.test)[1]
  p = dim(X.test)[2]
  diff.mtx.list = diff.matrix(X.train, X.test)

  beta = c()
  for (i in 1:n.test){
    if(sd(Y.train[wrf.list[[i]]!=0])==0 | length(Y.train[wrf.list[[i]]!=0])<=1) {
      beta[i] = mean(Y.train[wrf.list[[i]]!=0])
    }
    else{
      fit = glmnet(x = diff.mtx.list[[i]]%*%diag(local.feature.weight[i,]), y = Y.train, weights = wrf.list[[i]], alpha = alpha)
      # print(fit)
      beta.i = as.vector(coef(fit, s=lambda))
      # print(beta.i[1])
      beta[i] = beta.i[1]
    }
  }
  return(beta)
}


