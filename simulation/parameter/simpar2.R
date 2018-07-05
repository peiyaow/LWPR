# library(glmnet)
# library(randomForest)
# library(MASS)
# library(msm)
setwd("/Users/MonicaW/Documents/R/np/simulation/")
# set.seed(706)
n = 150
p0 = 200
p1 = 10
p2 = 10
p3 = 10
pc = 20
p = p0+p1+p2+p3+pc
Sigma_1 = diag(p1)
Sigma_2 = diag(p2)
Sigma_3 = diag(p3)
Sigma_0 = diag(p0)
Sigma_c = diag(pc)
rho_e = 1
w = c(rep(1, p1), rep(1, p2), rep(1, p3), rep(1, pc), rep(0, p0))

save.image(file = "sim_par23.RData")

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


