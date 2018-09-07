# library(glmnet)
# library(randomForest)
# library(MASS)
# library(msm)
setwd("/Users/MonicaW/Documents/R/np/simulation/")
load("~/Documents/R/np/simulation/sim_par23.RData")
# n = 150
# p0 = 50
# p1 = 5
# p2 = 5
# p3 = 5
# p = p0+p1+p2+p3

# Sigma_1 = diag(p1)
# Sigma_2 = diag(p2)
# Sigma_3 = diag(p3)
# Sigma_0 = diag(p0)

rho = 0.5
for (i in 1:p1){
  for (j in 1:p1){
    Sigma_1[i, j] = rho^abs(i-j)
  }
}

for (i in 1:p2){
  for (j in 1:p2){
    Sigma_2[i, j] = rho^abs(i-j)
  }
}

for (i in 1:p3){
  for (j in 1:p3){
    Sigma_3[i, j] = rho^abs(i-j)
  }
}

for (i in 1:p0){
  for (j in 1:p0){
    Sigma_0[i, j] = rho^abs(i-j)
  }
}

for (i in 1:pc){
  for (j in 1:pc){
    Sigma_c[i, j] = rho^abs(i-j)
  }
}

save.image(file = "sim_par33.RData")
# # save.image(file = "sim_par1.RData")

# rho_e = 1

# C2 = cbind(runif(p2, 0, 1), runif(p2, 0, 1))
# C3 = cbind(runif(p3, 0, 1), runif(p3, 0, 1))
# w = c(rep(1, p1), rep(1, p2), rep(1, p3), rep(0, p0))


# save.image(file = "sim_par1.RData")

# data.train = mysimulation(C2, C3, w)
# data.test = mysimulation(C2, C3, w)
# X = data.train$X
# Y = data.train$Y
# label = data.train$label
# X.test = data.test$X
# Y.test = data.test$Y
# label.test = data.test$label
# 
# # boxplot(data.train$Y~data.train$label)
# 
# 
# ml.rf = randomForest(x = X, y = Y)
# ml.lasso = cv.glmnet(x = X, y = Y)
# ml.ridge = cv.glmnet(x = X, y = Y, alpha = 0)
# 
# mae.rf = mean(abs(predict(ml.rf, X = X.test) - Y.test))
# mae.lasso = mean(abs(predict(ml.lasso, newx = X.test) - Y.test))
# mae.ridge = mean(abs(predict(ml.ridge, newx = X.test) - Y.test))
# corr.rf = cor(predict(ml.rf, X = X.test), Y.test)
# corr.lasso = cor(predict(ml.lasso, newx = X.test), Y.test)
# corr.ridge = cor(predict(ml.ridge, newx = X.test), Y.test)


