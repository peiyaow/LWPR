library(MASS)
library(msm)
Posdef <- function (n, ev = runif(n, 0, 10)) 
{ 
  Z <- matrix(ncol=n, rnorm(n^2)) 
  decomp <- qr(Z) 
  Q <- qr.Q(decomp) 
  R <- qr.R(decomp) 
  d <- diag(R) 
  ph <- d / abs(d) 
  O <- Q %*% diag(ph) 
  Z <- t(O) %*% diag(ev) %*% O 
  return(Z) 
} 

mu0 = function(s, C){
  return(C%*%c(1,s))
}

mu_s = function(s, C){
  return(C%*%c(1, s))
}

# beta1 = function(s, ics, seed){
#   set.seed(seed)
#   if (ics){
#     x0 = runif(1, -2, 0)
#     # x0 = -1
#     y0 = runif(1, 0, -x0)
#     # y0 = 0.5
#   }else{
#     x0 = runif(1, 3, 5)
#     # x0 = 4
#     y0 = runif(1, 0, x0)
#     # y0 = 2
#   }
#   return(-y0/x0*s + y0)
# }

# beta2 = function(s, ics, seed){
#   set.seed(seed)
#   if (ics){
#     x0 = runif(1, 3, 5)
#     # x0 = 4
#     b = runif(1, 2, 4)
#     # b = 3
#     k = runif(1, -b/x0^2, 0)
#     # k = -0.1
#   }else{
#     x0 = runif(1, 3, 5)
#     # x0 = 4
#     b = runif(1, 0, 2)
#     # b = 1
#     k = runif(1, max(-b/(3-x0)^2, 0), (3-b)/x0^2)
#     # k = 0.1
#   }
#   return(k*(s-x0)^2 + b)
# }

# beta3 = function(s, ics, seed){
#   set.seed(seed)
#   b = runif(1,3,6)
#   # b = 5
#   if (!ics){
#     kprime = runif(1, -b/3, 0)
#     # kprime = -1
#     y = (s>=0 & s<1)*(kprime*s + b) + (s>=1 & s<2)*(kprime*s + 2/3*b - kprime) + (s>=2)*(kprime*s+1/3*b-2*kprime)
#   }else{
#     kprime = runif(1, 0, b/3)
#     # kprime = 1
#     y = (s>=0 & s<1)*(kprime*s + 1/3*b - kprime) + (s>=1 & s<2)*(kprime*s + 2/3*b - 2*kprime) + (s>=2)*(kprime*s+b-3*kprime)
#   }
#   return(y)
# }

mybeta = function(s, id.ics){
  ps = length(id.ics)
  id.beta = sample(1:3, size = ps, replace = T)
  Beta_s = t(sapply(1:ps, function(x) do.call(paste("beta", as.character(id.beta[x]), sep = ""), list(s=s, ics = id.ics[x]))))
  return(Beta_s)
}

# sparseX = function(n, p, p.sparse){
#   myX = list()
#   for (i in 1:p){
#     id.sparse = sample(1:n, size = ceiling(p.sparse*n), replace = F)
#     myX[[i]] = rep(0, n)
#     myX[[i]][-id.sparse] = abs(rnorm(n - ceiling(p.sparse*n)))
#   }
#   return(do.call(cbind, myX))
# }

mysimulation = function(C2, C3, w){
  X1 = mvrnorm(n = n, rep(0, p1), Sigma_1)
  X2 = mvrnorm(n = n, rep(0, p2), Sigma_2)
  X3 = mvrnorm(n = n, rep(0, p3), Sigma_3)
  X0 = mvrnorm(n = n, rep(0, p0), Sigma_0)
  
  linpred1 = -1.8 - cbind(X1, X2, X3, X0)%*%w
  linpred2 = 1.8 - cbind(X1, X2, X3, X0)%*%w
  
  P1 = exp(linpred1)/(1+exp(linpred1))
  P12 = exp(linpred2)/(1+exp(linpred2))
  P.mtx = cbind(P1, P12-P1, 1-P12)
  label = as.ordered(apply(P.mtx, 1, which.max))
  # table(label)
  
  s = cbind(X1, X2, X3, X0)%*%w
  o = order(s)
  s = s[o] - min(s)
  X = cbind(X1, X2, X3, X0)
  X = X[o,]
  label = label[o]
  X1 = X1[o,]
  X2 = X2[o,]
  X3 = X3[o,]
  X0 = X0[o,]
  
  # mybeta1 = sapply(s, function(y) mu_s(y, C1))
  ipt = c(0,12,20)
  mybeta1 = lapply(1:3, function(ix) matrix(ipt[ix], nrow = p1, ncol = sum(label == ix)))
  mybeta1 = do.call(cbind, mybeta1)
  # mybeta1 = rbind(mybeta1, matrix(0, nrow = p1-5, ncol = n))
  
  mybeta2 = sapply(sqrt(s), function(y) jitter(mu_s(y, C2)))
  # mybeta2 = rbind(mybeta2, matrix(0, nrow = p2-5, ncol = n))
  
  mybeta3 = sapply(log(s+1), function(y) jitter(mu_s(y, C3)))
  # mybeta3 = rbind(mybeta3, matrix(0, nrow = p3-3, ncol = n))
  
  # 1
  err = mvrnorm(n = n, 0, rho_e)[,1]
  
  # 2
  # err = sapply(1:3, function(ix) mvrnorm(sum(label == ix), 0, ix))
  
  # 3
  # err = list()
  # err[[1]] = rtnorm(sum(label == 1), mean=0, sd=1, lower=-1, upper=0)
  # err[[2]] = rtnorm(sum(label == 2), mean=0, sd=2, lower=-2, upper=2)
  # err[[3]] = rtnorm(sum(label == 3), mean=0, sd=3, lower=0, upper=3)
  # err = do.call(c, err)
  
  signal = mybeta1[1,] + diag((X2)%*%mybeta2/10 + (X3)%*%mybeta3/10)
 
  Y = signal + err
  
  return(list(X = X, Y = Y, signal = signal, label = label, s = s))
}

mysimulation1 = function(C1, C2, C3, Cc, w){
  X1 = mvrnorm(n = n, rep(0, p1), Sigma_1)
  X2 = mvrnorm(n = n, rep(0, p2), Sigma_2)
  X3 = mvrnorm(n = n, rep(0, p3), Sigma_3)
  Xc = mvrnorm(n = n, rep(0, pc), Sigma_c)
  X0 = mvrnorm(n = n, rep(0, p0), Sigma_0)
  
  X = cbind(X1, X2, X3, Xc, X0)
  
  linpred1 = -4 - X%*%w
  linpred2 = 4 - X%*%w
  
  P1 = exp(linpred1)/(1+exp(linpred1))
  P12 = exp(linpred2)/(1+exp(linpred2))
  P.mtx = cbind(P1, P12-P1, 1-P12)
  label = as.ordered(apply(P.mtx, 1, which.max))
  # table(label)
  
  s = X%*%w
  o = order(s)
  s = s[o] - min(s)
  
  X = X[o,]
  label = label[o]
  X1 = X1[o,]
  X2 = X2[o,]
  X3 = X3[o,]
  Xc = Xc[o,]
  X0 = X0[o,]
  
  # mybeta1 = sapply(s, function(y) mu_s(y, C1))
  # ipt = c(0,12,20)
  # mybeta1 = lapply(1:3, function(ix) matrix(ipt[ix], nrow = p1, ncol = sum(label == ix)))
  # mybeta1 = do.call(cbind, mybeta1)
  mybeta1 = sapply(log(s+1), function(y) jitter(mu_s(y, C1)))
  
  mybeta2 = sapply(log(s+1), function(y) jitter(mu_s(y, C2)))
  # mybeta2 = rbind(mybeta2, matrix(0, nrow = p2-5, ncol = n))
  
  mybeta3 = sapply(log(s+1), function(y) jitter(mu_s(y, C3)))
  # mybeta3 = rbind(mybeta3, matrix(0, nrow = p3-3, ncol = n))
  
  mybetac = sapply(log(s+1), function(y) jitter(mu_s(y, Cc)))
  # 1
  err = mvrnorm(n = n, 0, rho_e)[,1]
  
  # 2
  # err = sapply(1:3, function(ix) mvrnorm(sum(label == ix), 0, ix))
  
  # 3
  # err = list()
  # err[[1]] = rtnorm(sum(label == 1), mean=0, sd=1, lower=-1, upper=0)
  # err[[2]] = rtnorm(sum(label == 2), mean=0, sd=2, lower=-2, upper=2)
  # err[[3]] = rtnorm(sum(label == 3), mean=0, sd=3, lower=0, upper=3)
  # err = do.call(c, err)
  table(label)
  ipt = c(rep(20, table(label)[1]), rep(30, table(label)[2]), rep(40, table(label)[3]))
  # signal = mybeta1[1,] + diag((X2)%*%mybeta2/10 + (X3)%*%mybeta3/10)
  signal = ipt + c(diag(X1[label == 1,]%*%mybeta1[,label ==1])/40, diag(X2[label == 2,]%*%mybeta2[,label ==2])/40, diag(X3[label == 3,]%*%mybeta3[,label == 3])/40) + diag(Xc%*%mybetac/25)
  
  Y = signal + err
  # plot(Y)
  return(list(X = X, Y = Y, signal = signal, label = label, s = s))
}

mysimulation2 = function(Cc, w){
  Xc = mvrnorm(n = n, rep(0, pc), Sigma_c)
  X0 = mvrnorm(n = n, rep(0, p0), Sigma_0)
  
  X = cbind(Xc, X0)
  
  linpred1 = -2.5 - X%*%w
  linpred2 = 2.5 - X%*%w
  
  P1 = exp(linpred1)/(1+exp(linpred1))
  P12 = exp(linpred2)/(1+exp(linpred2))
  P.mtx = cbind(P1, P12-P1, 1-P12)
  label = as.ordered(apply(P.mtx, 1, which.max))
  table(label)
  
  s = X%*%w
  o = order(s)
  s = s[o] - min(s)
  
  X = X[o,]
  label = label[o]
  Xc = Xc[o,]
  X0 = X0[o,]

  mybetac = sapply(log(s+1), function(y) jitter(Cc%*%c(1, y, sqrt(y))))
  err = mvrnorm(n = n, 0, rho_e)[,1]
  # ipt = c(rep(20, table(label)[1]), rep(30, table(label)[2]), rep(40, table(label)[3]))
  signal = diag(Xc%*%mybetac/15)
  
  Y = signal + err
  plot(Y)
  return(list(X = X, Y = Y, signal = signal, label = label, s = s))
}

mysimulation3 = function(n, p1, p2, p3, pc, p0, Sigma_1, Sigma_2, Sigma_3, Sigma_c, Sigma_0, rho_e, w){
  X1 = mvrnorm(n = n, rep(0, p1), Sigma_1)
  X2 = mvrnorm(n = n, rep(0, p2), Sigma_2)
  X3 = mvrnorm(n = n, rep(0, p3), Sigma_3)
  Xc = mvrnorm(n = n, rep(0, pc), Sigma_c)
  X0 = mvrnorm(n = n, rep(0, p0), Sigma_0)
  
  X = cbind(X1, X2, X3, Xc, X0)
  
  linpred1 = -4 - X%*%w
  linpred2 = 4 - X%*%w
  
  P1 = exp(linpred1)/(1+exp(linpred1))
  P12 = exp(linpred2)/(1+exp(linpred2))
  P.mtx = cbind(P1, P12-P1, 1-P12)
  label = as.ordered(apply(P.mtx, 1, which.max))
  # table(label)
  
  s = X%*%w
  o = order(s)
  s = s[o] - min(s)
  
  X = X[o,]
  label = label[o]
  X1 = X1[o,]
  X2 = X2[o,]
  X3 = X3[o,]
  Xc = Xc[o,]
  X0 = X0[o,]
  
  n.class = sapply(1:3, function(id) sum(label==id))
  
  mybeta1 = matrix(rep(c(rep(1, n.class[1]), rep(0, n.class[2]+n.class[3])), p1), ncol = p1)
  mybeta2 = matrix(rep(c(rep(0, n.class[1]), rep(2, n.class[2]), rep(0, n.class[3])), p2), ncol = p2)
  mybeta3 = matrix(rep(c(rep(0, n.class[1]+n.class[2]), rep(3, n.class[3])), p3), ncol = p3)
  mybetac = matrix(rep(c(rep(1, n.class[1]), rep(1.5, n.class[2]), rep(2, n.class[3])), pc), ncol = pc)
  
  err = mvrnorm(n = n, 0, rho_e)[,1]
  # ipt = c(rep(2, table(label)[1]), rep(3, table(label)[2]), rep(4, table(label)[3]))
  signal = s/5 + diag(X1%*%t(mybeta1)) + diag(X2%*%t(mybeta2)) + diag(X3%*%t(mybeta3)) + diag(Xc%*%t(mybetac))
  
  Y = signal + err
  # plot(signal)
  # plot(Y)
  return(list(X = X, Y = Y, signal = signal, label = label, s = s))
}

mysimulation4 = function(n, p1, p2, p3, pc, p0, Sigma_1, Sigma_2, Sigma_3, Sigma_c, Sigma_0, rho_e, w){
  X1 = mvrnorm(n = n, rep(0, p1), Sigma_1)
  X2 = mvrnorm(n = n, rep(0, p2), Sigma_2)
  X3 = mvrnorm(n = n, rep(0, p3), Sigma_3)
  Xc = mvrnorm(n = n, rep(0, pc), Sigma_c)
  X0 = mvrnorm(n = n, rep(0, p0), Sigma_0)
  
  X = cbind(X1, X2, X3, Xc, X0)
  
  linpred1 = -4 - X%*%w
  linpred2 = 4 - X%*%w
  
  P1 = exp(linpred1)/(1+exp(linpred1))
  P12 = exp(linpred2)/(1+exp(linpred2))
  P.mtx = cbind(P1, P12-P1, 1-P12)
  label = as.ordered(apply(P.mtx, 1, which.max))
  # table(label)
  
  s = X%*%w
  o = order(s)
  # s = s[o] - min(s)
  
  X = X[o,]
  label = label[o]
  X1 = X1[o,]
  X2 = X2[o,]
  X3 = X3[o,]
  Xc = Xc[o,]
  X0 = X0[o,]
  
  n.class = sapply(1:3, function(id) sum(label==id))
  
  mybeta1 = matrix(rep(c(rep(1, n.class[1]), rep(0, n.class[2]+n.class[3])), p1), ncol = p1)
  mybeta2 = matrix(rep(c(rep(0, n.class[1]), rep(2, n.class[2]), rep(0, n.class[3])), p2), ncol = p2)
  mybeta3 = matrix(rep(c(rep(0, n.class[1]+n.class[2]), rep(3, n.class[3])), p3), ncol = p3)
  mybetac = matrix(rep(c(rep(1, n.class[1]), rep(1.5, n.class[2]), rep(2, n.class[3])), pc), ncol = pc)
  
  err = mvrnorm(n = n, 0, rho_e)[,1]
  
  # signal = s/5 + diag(X1%*%t(mybeta1)) + diag(X2%*%t(mybeta2)) + diag(X3%*%t(mybeta3)) + diag(Xc%*%t(mybetac))
  signal = diag(X1%*%t(mybeta1)) + diag(X2%*%t(mybeta2)) + diag(X3%*%t(mybeta3)) + diag(Xc%*%t(mybetac))
  
  Y = signal + err
  return(list(X = X, Y = Y, signal = signal, label = label, s = s))
}

mysimulation5 = function(n, p1, p2, p3, pc, p0, Sigma_1, Sigma_2, Sigma_3, Sigma_c, Sigma_0, rho_e, w){
  X1 = mvrnorm(n = n, rep(0, p1), Sigma_1)
  X2 = mvrnorm(n = n, rep(0, p2), Sigma_2)
  X3 = mvrnorm(n = n, rep(0, p3), Sigma_3)
  Xc = mvrnorm(n = n, rep(0, pc), Sigma_c)
  X0 = mvrnorm(n = n, rep(0, p0), Sigma_0)
  
  X = cbind(X1, X2, X3, Xc, X0)
  
  linpred1 = -4 - X%*%w
  linpred2 = 4 - X%*%w
  
  P1 = exp(linpred1)/(1+exp(linpred1))
  P12 = exp(linpred2)/(1+exp(linpred2))
  P.mtx = cbind(P1, P12-P1, 1-P12)
  label = as.ordered(apply(P.mtx, 1, which.max))
  # table(label)
  
  s = X%*%w
  o = order(s)
  # s = s[o] - min(s)
  
  X = X[o,]
  label = label[o]
  X1 = X1[o,]
  X2 = X2[o,]
  X3 = X3[o,]
  Xc = Xc[o,]
  X0 = X0[o,]
  
  n.class = sapply(1:3, function(id) sum(label==id))
  
  mybeta1 = matrix(rep(c(rep(1, n.class[1]), rep(0, n.class[2]+n.class[3])), p1), ncol = p1)
  mybeta2 = matrix(rep(c(rep(0, n.class[1]), rep(2, n.class[2]), rep(0, n.class[3])), p2), ncol = p2)
  mybeta3 = matrix(rep(c(rep(0, n.class[1]+n.class[2]), rep(3, n.class[3])), p3), ncol = p3)
  mybetac = matrix(rep(c(rep(1, n.class[1]), rep(1.5, n.class[2]), rep(2, n.class[3])), pc), ncol = pc)
  
  err = mvrnorm(n = n, 0, rho_e)[,1]
  
  # signal = s/5 + diag(X1%*%t(mybeta1)) + diag(X2%*%t(mybeta2)) + diag(X3%*%t(mybeta3)) + diag(Xc%*%t(mybetac))
  signal = diag(X1%*%t(mybeta1)) + diag(X2%*%t(mybeta2)) + diag(X3%*%t(mybeta3)) + diag(Xc%*%t(mybetac))
  
  Y = signal + err
  
  # choose some label and assign another label to it
  percent = .2

  ix_change_label = sample(seq(1, length(label)), floor(percent*length(label)))
  P.mtx = P.mtx[o,]
  for (ix in ix_change_label){
    if (label[ix] == 1 | label[ix] == 3){
#      print(label[ix])
      label[ix] = 2
    }else{
#      print(label[ix])
      prob1 = P.mtx[ix,1]/sum(P.mtx[ix,c(1,3)])
#      print(prob1)
      label[ix] = c(1,3)[rbinom(1, 1, prob1)+1]
    }
  }
  return(list(X = X, Y = Y, signal = signal, label = label, s = s))
}


