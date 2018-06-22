library(caret)
library(foreach)
library(doParallel)

mygrad = function(x, label, X, lambda){
  K = length(levels(label))
  p = ncol(X)
  n = nrow(X)
  designmtx = t(sapply(1:n, function(x) diag(K)[as.numeric(label)[x],]))
  designmtx.i0 = cbind(rep(0,n), designmtx) # n by K+1
  designmtx.i1 = cbind(designmtx, rep(0,n)) # n by K+1
  
  theta = c(-1e4, x[1:(K-1)], 1e4) # K+1 vector
  w = x[-(1:(K-1))] # p vector
  
  theta0 = designmtx.i0%*%theta # theta_yi
  theta1 = designmtx.i1%*%theta # theta_yi-1
  etimes0 = (1 - myphi(theta0 - X%*%w))-1/(1-exp(theta0 - theta1)) # scalar (n by 1)
  etimes1 = (1 - myphi(theta1 - X%*%w))-1/(1-exp(theta1 - theta0)) #
  
  deriv.theta = -apply(t(sapply(1:n, function(x) designmtx.i0[x, 2:K]*etimes0[x])) + t(sapply(1:n, function(x) designmtx.i1[x, 2:K]*etimes1[x])), 2, sum) # K-1 by 1
  
  deriv.w = apply(t(sapply(1:n, function(x) (1-myphi(designmtx.i0%*%theta - X%*%w)-myphi(designmtx.i1%*%theta - X%*%w))[x]*X[x,])), 2, sum) + lambda*w
  return(c(deriv.theta, deriv.w))
}

myphi = function(t){
  return(1/(1+exp(-t)))
}

penalike = function(x, label, X, lambda){
  K = length(levels(label))
  p = ncol(X)
  n = nrow(X)
  designmtx = t(sapply(1:n, function(x) diag(K)[as.numeric(label)[x],]))
  designmtx.i0 = cbind(rep(0,n), designmtx) # n by K+1
  designmtx.i1 = cbind(designmtx, rep(0,n)) # n by K+1
  
  theta = c(-1e4, x[1:(K-1)], 1e4) # K+1 vector
  w = x[-(1:(K-1))] # p vector

  return(-sum(log(myphi(designmtx.i0%*%theta - X%*%w)-myphi(designmtx.i1%*%theta - X%*%w)))+lambda/2*sum(w^2))
}

ordin.logistic = function(label, X, lambda, initial.x){ # optimize over the objective
  K = length(levels(label))
  p = dim(X)[2]
  
  ui.theta = cbind(-diag(K-2), 0) + cbind(0, diag(K-2))
  ui = cbind(ui.theta, matrix(0, nrow = K-2, ncol = p))
  ci = rep(0, K-2)
  
  ml = constrOptim(initial.x, penalike, mygrad, label = label, X = X, lambda = lambda, ui = ui, ci = ci) # 
  
  return(list(theta = ml$par[1:(K-1)], w = ml$par[-(1:(K-1))], conv = ml$convergence, lambda = lambda))
}

postprob = function(ordin.ml, X){
  K = length(ordin.ml$theta) + 1
  P0.mtx = sapply(1:(K-1), function(x) 1/(1+exp(as.vector(X%*%ordin.ml$w)-ordin.ml$theta[x])))
  P.mtx = cbind(0, P0.mtx, 1)
  P_label.mtx = sapply(1:K, function(x) P.mtx[,x+1]-P.mtx[,x])
  return(P_label.mtx)
}

score2status = function(Y){
  coef.score2status = solve(matrix(c(min(Y), max(Y), 1, 1), nrow = 2))%*%c(0,1)
  status = cbind(Y, 1)%*%coef.score2status
  return(status)
}

predsoftlabel = function(ordin.ml, X, status.coef){
  Phat = postprob(ordin.ml, X)
  # Phat = predict(cumulative.logit.train, newx = mydata)$predicted
  # 8.11 for K classes
  softlabel = apply(Phat, 1, function(x) sum(x[-1]*status.coef))
  return(softlabel = softlabel)
}

path.ordinlog = function(label, X, Sl, lam.vec, initial.x){
  lenlam = length(lam.vec)
  ml.list = apply(as.matrix(lam.vec), 1, function(x) ordin.logisic(label, X, x, initial.x))
  postprob.list = lapply(1:lenlam, function(x) postprob(ml.list[[x]], X))
  Sl_on_prob_coef.mtx = sapply(1:lenlam, function(x) lm(Sl ~ 0 + postprob.list[[x]][, -1])$coef) # K-1 by nlambda matrix
  return(list(ordin.ml.list = ml.list, Sl.coef.mtx = Sl_on_prob_coef.mtx, gamma.vec = lam.vec))
}

cv.ordinlog = function(label, X, Y, lam.vec, initial.x, nfolds, measure.type){
  flds = createFolds(label, k = nfolds, list = TRUE, returnTrain = FALSE)
  if (measure.type == "mse"){
    mse.vec = rep(0,length(lam.vec))
    for (i in 1:nfolds){
      X.train = X[unlist(flds[-i]), ]
      X.val = X[unlist(flds[i]), ]
      Y.train = Y[unlist(flds[-i])]
      Y.val = Y[unlist(flds[i])]
      label.train = label[unlist(flds[-i])]
      label.val = label[unlist(flds[i])]
      Sl.train = Sl[unlist(flds[-i])]
      Sl.val = Sl[unlist(flds[i])]
      
      ml.list = apply(as.matrix(lam.vec), 1, function(x) ordin.logistic(label.train, X.train, x, initial.x))
      postprob.train.list = lapply(1:length(ml.list), function(x) postprob(ml.list[[x]], X.train))
      postprob.val.list = lapply(1:length(ml.list), function(x) postprob(ml.list[[x]], X.val))
      Sl_on_prob_coef.list = lapply(1:length(ml.list), function(x) lm(Sl.train ~ 0 + postprob.train.list[[x]][, -1])$coef)
      
      Slhat.val.list = lapply(1:length(ml.list), function(x) postprob.val.list[[x]][, -1]%*%Sl_on_prob_coef.list[[x]])
      mse.vec = mse.vec + as.vector(sapply(1:length(ml.list), function(x) sum((Slhat.val.list[[x]] - Sl.val)^2)))
    }
    mse.vec = mse.vec/nfolds
    
    lam.min = lam.vec[which.min(mse.vec)]
    ordin.ml.best = ordin.logistic(label, X, lam.min, initial.x)
    postprob.best = postprob(ordin.ml.best, X)

    Sl_on_prob_coef.best = lm(Sl ~ 0 + postprob.best[, -1])$coef
    
    return(list(measure.vec = mse.vec, measure.type = measure.type, lam = lam.min, ordin.ml= ordin.ml.best, Sl_on_prob_coef = Sl_on_prob_coef.best))
  } else if (measure.type == "corr"){
    corr.list = list()
    for (i in 1:nfolds){
      X.train = X[unlist(flds[-i]), ]
      X.val = X[unlist(flds[i]), ]
      Y.train = Y[unlist(flds[-i])]
      Y.val = Y[unlist(flds[i])]
      label.train = label[unlist(flds[-i])]
      label.val = label[unlist(flds[i])]
      
      ml.list = apply(as.matrix(lam.vec), 1, function(x) ordin.logistic(label.train, X.train, x, initial.x))
      Slhat.val.list = lapply(ml.list, function(ml) X.val%*%ml$w)

      corr.list[[i]] = as.vector(sapply(Slhat.val.list, function(Slhat) cor(Slhat, Y.val)))
    }
    corr.vec = apply(do.call(rbind, corr.list), 2, mean)
    
    lam.max = lam.vec[which.min(abs(abs(corr.vec)-1))]
    ordin.ml.best = ordin.logistic(label, X, lam.max, initial.x)

    return(list(measure.vec = corr.vec, measure.type = measure.type, lam = lam.max, ordin.ml= ordin.ml.best))
  }
}


penalike.en = function(x, label, X, lambda, alpha){

  K = length(levels(label))
  p = ncol(X)
  n = nrow(X)
  x0 = x[1:(K-1+p)]
  designmtx = t(sapply(1:n, function(x) diag(K)[as.numeric(label)[x],]))
  designmtx.i0 = cbind(rep(0,n), designmtx) # n by K+1
  designmtx.i1 = cbind(designmtx, rep(0,n)) # n by K+1
  
  theta = c(-1e4, x0[1:(K-1)], 1e4) # K+1 vector
  w = x0[-(1:(K-1))] # p vector
  t = x[(K-1+p+1):(K-1+2*p)]

  likelihood = -sum(log(myphi(designmtx.i0%*%theta - X%*%w)-myphi(designmtx.i1%*%theta - X%*%w)))
  penalty = lambda*((1-alpha)*sum(w^2) + alpha*sum(t))
  return(likelihood + penalty)
}

mygrad.en = function(x, label, X, lambda, alpha){
  K = length(levels(label))
  p = ncol(X)
  n = nrow(X)
  x0 = x[1:(K-1+p)]
  designmtx = t(sapply(1:n, function(x) diag(K)[as.numeric(label)[x],]))
  designmtx.i0 = cbind(rep(0,n), designmtx) # n by K+1
  designmtx.i1 = cbind(designmtx, rep(0,n)) # n by K+1
  
  theta = c(-1e4, x0[1:(K-1)], 1e4) # K+1 vector
  w = x0[-(1:(K-1))] # p vector
  t = x[(K-1+p+1):(K-1+2*p)]
  
  theta0 = designmtx.i0%*%theta # theta_yi
  theta1 = designmtx.i1%*%theta # theta_yi-1
  etimes0 = (1 - myphi(theta0 - X%*%w))-1/(1-exp(theta0 - theta1)) # scalar (n by 1)
  etimes1 = (1 - myphi(theta1 - X%*%w))-1/(1-exp(theta1 - theta0)) #
  
  deriv.theta = -apply(t(sapply(1:n, function(x) designmtx.i0[x, 2:K]*etimes0[x])) + t(sapply(1:n, function(x) designmtx.i1[x, 2:K]*etimes1[x])), 2, sum) # K-1 by 1
  
  deriv.w = apply(t(sapply(1:n, function(x) (1-myphi(designmtx.i0%*%theta - X%*%w)-myphi(designmtx.i1%*%theta - X%*%w))[x]*X[x,])), 2, sum) + 2*lambda*(1-alpha)*w
  
  deriv.t = alpha*lambda*rep(1, p) 
  
  return(c(deriv.theta, deriv.w, deriv.t))
}

ordin.logistic.en = function(label, X, lambda, alpha, initial.x){ # optimize over the objective
  K = length(levels(label))
  p = dim(X)[2]
  
  u1.theta = cbind(-diag(K-2), 0) + cbind(0, diag(K-2)) # K-2 by K-1
  u1 = cbind(u1.theta, matrix(0, nrow = K-2, ncol = 2*p)) # K-2 by K-1+2*p
  c1 = rep(0, K-2)
  
  u2.wt = cbind(diag(p), diag(p)) # p by 2p
  u3.wt = cbind(-diag(p), diag(p)) # p by 2p
  u2 = cbind(matrix(0, nrow = p, ncol = K-1), u2.wt) # p by K-1+2p
  u3 = cbind(matrix(0, nrow = p, ncol = K-1), u3.wt)
  c2 = rep(0,p)
  c3 = rep(0,p)
  
  ui = rbind(u1, u2, u3)
  ci = c(c1, c2, c3)
  
  ml = constrOptim(initial.x, f = penalike.en, grad = mygrad.en, label = label, X = X, lambda = lambda, alpha = alpha, ui = ui, ci = ci) # 
  
  return(list(theta = ml$par[1:(K-1)], w = ml$par[K:(K-1+p)], t = ml$par[(K+p):(K-1+2*p)], conv = ml$convergence, lambda = lambda, alpha = alpha))
}

cv.ordinlog.en = function(label, X, Y, lam.vec, alpha, initial.x, nfolds, measure.type){
  flds = createFolds(label, k = nfolds, list = TRUE, returnTrain = FALSE)
  if (measure.type == "mse"){
    mse.vec = rep(0,length(lam.vec))
    for (i in 1:nfolds){
      X.train = X[unlist(flds[-i]), ]
      X.val = X[unlist(flds[i]), ]
      label.train = label[unlist(flds[-i])]
      label.val = label[unlist(flds[i])]
      Sl.train = Sl[unlist(flds[-i])]
      Sl.val = Sl[unlist(flds[i])]
      
      ml.list = apply(as.matrix(lam.vec), 1, function(x) ordin.logistic.en(label.train, X.train, x, alpha, initial.x))
      postprob.train.list = lapply(1:length(ml.list), function(x) postprob(ml.list[[x]], X.train))
      postprob.val.list = lapply(1:length(ml.list), function(x) postprob(ml.list[[x]], X.val))
      
      Sl_on_prob_coef.list = lapply(1:length(ml.list), function(x) lm(Sl.train ~ 0 + postprob.train.list[[x]][, -1])$coef)

      Slhat.val.list = lapply(1:length(ml.list), function(x) postprob.val.list[[x]][, -1]%*%Sl_on_prob_coef.list[[x]])

      mse.vec = mse.vec + as.vector(sapply(1:length(ml.list), function(x) sum((Slhat.val.list[[x]] - Sl.val)^2)))
    }
    mse.vec = mse.vec/nfolds
    
    lam.min = lam.vec[which.min(mse.vec)]
    ordin.ml.best = ordin.logistic.en(label, X, lam.min, alpha, initial.x)
    postprob.best = postprob(ordin.ml.best, X)
    
    Sl_on_prob_coef.best = lm(Sl ~ 0 + postprob.best[, -1])$coef
    
    return(list(measure.vec = mse.vec, measure.type = measure.type, lam = lam.min, ordin.ml= ordin.ml.best, Sl_on_prob_coef = Sl_on_prob_coef.best))
  } else if (measure.type == "corr"){
    corr.list = list()
    
    cl = makeCluster(4) # number of cores you can use
    registerDoParallel(cl)
    
    for (i in 1:nfolds){
      X.train = X[unlist(flds[-i]), ]
      X.val = X[unlist(flds[i]), ]
      Y.train = Y[unlist(flds[-i])]
      Y.val = Y[unlist(flds[i])]
      label.train = label[unlist(flds[-i])]
      label.val = label[unlist(flds[i])]
      
#      ml.list = apply(as.matrix(lam.vec), 1, function(x) ordin.logistic.en(label.train, X.train, x, alpha, initial.x))
    
      ml.list = foreach(lam=lam.vec, .export = c("ordin.logistic.en", "penalike.en", "myphi", "mygrad.en")) %dopar% {
        ordin.logistic.en(label.train, X.train, lam, alpha, initial.x)
      }
      
      Slhat.val.list = lapply(ml.list, function(ml) X.val%*%ml$w)
      
      # new detect outliers and delete outliers
      Slhat.extvalues.list = lapply(Slhat.val.list, function(sl) boxplot(sl, plot = F)$stats[c(1,5),1])
      Slhat.val.list = lapply(1:length(Slhat.val.list), function(ix){
        Slhat.val.list[[ix]][Slhat.val.list[[ix]] > Slhat.extvalues.list[[ix]][2]] = Slhat.extvalues.list[[ix]][2]
        Slhat.val.list[[ix]][Slhat.val.list[[ix]] < Slhat.extvalues.list[[ix]][1]] = Slhat.extvalues.list[[ix]][1]
        Slhat.val.list[[ix]]
      })
      
      corr.list[[i]] = as.vector(sapply(Slhat.val.list, function(Slhat) cor(Slhat, Y.val)))
    }
    stopCluster(cl)
    corr.vec = apply(do.call(rbind, corr.list), 2, mean)
    
    lam.max = lam.vec[which.min(abs(abs(corr.vec)-1))]
    ordin.ml.best = ordin.logistic.en(label, X, lam.max, alpha, initial.x)

    return(list(measure.vec = corr.vec, measure.type = measure.type, lam = lam.max, ordin.ml= ordin.ml.best))
  }
}
