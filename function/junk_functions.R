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