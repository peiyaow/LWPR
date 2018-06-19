load("~/Documents/R/np/simulation/sim_par33.RData")
setwd("~/Documents/R/np/simulation/result/")
source("~/Documents/R/np/simulation/sim.fun.R")

# myseeds = floor(1e4 * runif(50))
# save(myseeds, file = "myseed.RData")
load("~/Documents/R/np/simulation/sim.myseed.RData")
library(glmnet)
library(randomForest)
library(caret)
nfolds = 5

for (i in 1:50){
  set.seed(myseeds[i])
  
  data.train = mysimulation3(n, p1, p2, p3, pc, p0, Sigma_1, Sigma_2, Sigma_3, Sigma_c, Sigma_0, rho_e, w)
  data.test = mysimulation3(n, p1, p2, p3, pc, p0, Sigma_1, Sigma_2, Sigma_3, Sigma_c, Sigma_0, rho_e, w)
  X = data.train$X
  Y = data.train$Y
  label = data.train$label
  X.test = data.test$X
  Y.test = data.test$Y
  label.test = data.test$label
  
  ml.rf = randomForest(x = X, y = Y)
  flds = createFolds(label, k = nfolds, list = FALSE, returnTrain = FALSE)
  
  ml.lasso = cv.glmnet(x = X, y = Y, nfolds = nfolds, foldid = flds)
  ml.ridge = cv.glmnet(x = X, y = Y, alpha = 0, nfolds = nfolds, foldid = flds)   
  ml.elast = cv.glmnet(x = X, y = Y, alpha = 0.5, nfolds = nfolds, foldid = flds)
  
  mae.rf = mean(abs(predict(ml.rf, X = X.test) - Y.test))
  mae.lasso = mean(abs(predict(ml.lasso, newx = X.test) - Y.test))
  mae.ridge = mean(abs(predict(ml.ridge, newx = X.test) - Y.test))
  mae.elast = mean(abs(predict(ml.elast, newx = X.test) - Y.test))
  corr.rf = cor(predict(ml.rf, X = X.test), Y.test)
  corr.lasso = cor(predict(ml.lasso, newx = X.test)[,1], Y.test)
  corr.ridge = cor(predict(ml.ridge, newx = X.test)[,1], Y.test)
  corr.elast = cor(predict(ml.elast, newx = X.test), Y.test)
  
  method.names = c("mae.rf", "mae.lasso", "mae.ridge", "mae.elast", "corr.rf", "corr.lasso", "corr.ridge", "corr.elast")
  file.name = "simulationresults33.csv"
  write.table(t(c(mae.rf, mae.lasso, mae.ridge, mae.elast, corr.rf, corr.lasso, corr.ridge, corr.elast)), file = file.name, sep = ',', append = T, col.names = ifelse(rep(file.exists(file.name), 8), F, method.names), row.names = F)
}






