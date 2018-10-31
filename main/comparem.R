# non parallel program for comparing methods
library(caret)
library(methods) # "is function issue by Rscript"
library(glmnet)
library(randomForest)
library(readr)
library(energy)
library(doParallel)
library(gdata)

load("/nas/longleaf/home/peiyao/LWPR/myseed1.RData")

for (i in 1:50){
  for (t in 1:3){
    X0 = as.matrix(read_table("/nas/longleaf/home/peiyao/LWPR/data/X1a.txt", col_names = F))
    Y0 = as.matrix(read_table("/nas/longleaf/home/peiyao/LWPR/data/Y5T.txt", col_names = F))[, 4+5*(t-1)]
    label0 = as.ordered(read_table("/nas/longleaf/home/peiyao/LWPR/data/label1.txt", col_names = F)$X1)
    
    # remove NAs in Y
    id.cut1 = !is.na(Y0)
    X0 = X0[id.cut1, ]
    Y0 = Y0[id.cut1]
    label0 = label0[id.cut1]
    
    # remove negatives in Y
    id.cut2 = (Y0 > 0)
    X0 = X0[id.cut2,]
    Y0 = Y0[id.cut2]
    label0 = label0[id.cut2]
    
    # only select MRI+PET
    X0 = X0[, 1:186]
    
    # within group median , each element from the list is the median within the group 
    # list of length 3, each element 1 by 186 group median of each feature
    X0.med.list = lapply(levels(label0), function(x) apply(X0[label0 == x, ], 2, function(y) median(y, na.rm = T)))
    
    # impute with median
    X0.missing = is.na(X0)
    for (i in 1:nrow(X0)){
      ip.med = X0.med.list[[as.numeric(label0[i])]]
      X0[i, X0.missing[i, ]] = ip.med[X0.missing[i, ]]
    }
    
    # only select NC and MCI patients
    X = X0[label0 < 4,]
    Y = Y0[label0 < 4]
    label = drop.levels(label0[label0 < 4])
    #print(X)

    # ---------------------- creating training and testing set ----------------------------- 
    n = dim(X)[1]
    p = dim(X)[2]
    
    myseed = myseeds[i]
    #print(myseed)
    set.seed(myseed)
    idtrain = unlist(createDataPartition(label, times = 1, p = 3/4))
    idtest = (1:n)[-idtrain]
    
    Y = log(Y+1)
    id = list(idtrain, idtest)
    Y.list = lapply(1:2, function(x) Y[id[[x]]]) # train,test
    X.list = lapply(1:2, function(x) X[id[[x]],]) 
    label.list = lapply(1:2, function(x) label[id[[x]]])
    # --------------------------------------------------------------------------------------
    
    # ----------------------------------- do scale ----------------------------------
    X1.mean = apply(X.list[[1]], 2, mean)
    X1.sd = apply(X.list[[1]], 2, sd)
    X1.sd = sapply(X1.sd, function(x) ifelse(x<1e-5, 1, x)) 
    
    X.list[[1]] = sweep(X.list[[1]], 2, X1.mean)
    X.list[[1]] = sweep(X.list[[1]], 2, X1.sd, "/")
    
    X.list[[2]] = sweep(X.list[[2]], 2, X1.mean)
    X.list[[2]] = sweep(X.list[[2]], 2, X1.sd, "/")
    print("Finish scaling features")
    #print(X.list)
    # -------------------------------------------------------------------------------
    
    # ----------------------------- Calculating interaction term ------------------------------
    X.interaction.list = list()
    X.plus.inter.list = list()
    for (m in 1:2){
      X.interaction.list[[m]] = list()
      k = 0
      for (ii in 1:p){
        for (jj in 1:p){
          if (ii < jj){
            k = k+1
            X.interaction.list[[m]][[k]] = X.list[[m]][,ii]*X.list[[m]][,jj]
          }
        }
      }
      X.interaction.list[[m]] = do.call(cbind, X.interaction.list[[m]])
      # X.plus.inter.list[[m]] = cbind(X.list[[m]], X.interaction.list[[m]])
    }
    
    X1.interaction.mean = apply(X.interaction.list[[1]], 2, mean)
    X1.interaction.sd = apply(X.interaction.list[[1]], 2, sd)
    X1.interaction.sd = sapply(X1.interaction.sd, function(x) ifelse(x<1e-5, 1, x)) 
    
    X.interaction.list[[1]] = sweep(X.interaction.list[[1]], 2, X1.interaction.mean)
    X.interaction.list[[1]] = sweep(X.interaction.list[[1]], 2, X1.interaction.sd, "/")
    
    X.interaction.list[[2]] = sweep(X.interaction.list[[2]], 2, X1.interaction.mean)
    X.interaction.list[[2]] = sweep(X.interaction.list[[2]], 2, X1.interaction.sd, "/")
    X.plus.inter.list = lapply(1:2, function(x) cbind(X.list[[x]], X.interaction.list[[x]]))
    
    # --- computing distance correlation to select favorite number of features including interaction features --- 
    # number of features to be selected
    #print(X.plus.inter.list)
    p_dc = 200 
    
    cl = makeCluster(4) # number of cores you can use
    registerDoParallel(cl)
    
    dc.vec = foreach(col_ix = 1:ncol(X.plus.inter.list[[1]]), .packages = "energy", .combine = "c") %dopar% {
      dcor(Y.list[[1]], X.plus.inter.list[[1]][,col_ix])
    }
    stopCluster(cl)
    # dc.vec = apply(X.plus.inter.list[[1]], 2, function(feature) dcor(Y.list[[1]], feature))
    print("Finish calculation distance correlation")
    
    X.selected.feature.id = order(dc.vec, decreasing = TRUE)[1:p_dc]
    X.selected.feature.list = lapply(1:2, function(ix) X.plus.inter.list[[ix]][, X.selected.feature.id])
    print("Finish addig top interaction terms into design matrix")
    # -----------------------------------------------------------------------------------------------------------
    
    # add to the last... probably to fix a random forest bug
    feature.ncol = ncol(X.selected.feature.list[[1]])
    colnames(X.selected.feature.list[[1]]) = as.character(seq(1, feature.ncol))
    colnames(X.selected.feature.list[[2]]) = as.character(seq(1, feature.ncol))
    # ------------------------------------------------------------------------------------------
    
    # ------------------rf------------------- 
    # set.seed(1010)
    ml.rf = randomForest(x = X.selected.feature.list[[1]], y = Y.list[[1]], keep.inbag = T, ntree = 100)
    Yhat.rf = exp(predict(ml.rf, newdata = X.selected.feature.list[[2]]))
    mae.rf = mean(abs(exp(Y.list[[2]])-Yhat.rf))
    corr.rf = cor(Yhat.rf, exp(Y.list[[2]]))
    #mae.class.rf = sapply(levels(label.list[[2]]), function(ix) mean(abs(Y.list[[2]][label.list[[2]]==ix]-Yhat.rf[label.list[[2]]==ix])))
    #corr.class.rf = sapply(levels(label.list[[2]]), function(ix) cor(Yhat.rf[label.list[[2]]==ix], Y.list[[2]][label.list[[2]]==ix]))
    # ---------------------------------------
    
    # --------------- glmnet ----------------
    # set.seed(1011)
    nfolds = 5
    flds = createFolds(label.list[[1]], k = nfolds, list = F, returnTrain = FALSE)
    model.lasso = cv.glmnet(x = X.selected.feature.list[[1]], y = Y.list[[1]], nfolds = nfolds, foldid = flds)
    Yhat.lasso = exp(predict(model.lasso, newx = X.selected.feature.list[[2]], s = "lambda.min"))
    mae.lasso = mean(abs(exp(Y.list[[2]])-Yhat.lasso))
    corr.lasso = cor(Yhat.lasso, exp(Y.list[[2]]))
    # mae.class.lasso = sapply(levels(label.list[[2]]), function(ix) mean(abs(Y.list[[2]][label.list[[2]]==ix]-Yhat.lasso[label.list[[2]]==ix])))
    # corr.class.lasso = sapply(levels(label.list[[2]]), function(ix) cor(Yhat.lasso[label.list[[2]]==ix], Y.list[[2]][label.list[[2]]==ix]))
    
    model.ridge = cv.glmnet(x = X.selected.feature.list[[1]], y = Y.list[[1]], alpha = 0, nfolds = nfolds, foldid = flds)
    Yhat.ridge = exp(predict(model.ridge, newx = X.selected.feature.list[[2]], s = "lambda.min"))
    mae.ridge = mean(abs(exp(Y.list[[2]])-Yhat.ridge))
    corr.ridge = cor(Yhat.ridge, exp(Y.list[[2]]))
    # mae.class.ridge = sapply(levels(label.list[[2]]), function(ix) mean(abs(Y.list[[2]][label.list[[2]]==ix]-Yhat.ridge[label.list[[2]]==ix])))
    # corr.class.ridge = sapply(levels(label.list[[2]]), function(ix) cor(Yhat.ridge[label.list[[2]]==ix], Y.list[[2]][label.list[[2]]==ix]))
    
    model.elast = cv.glmnet(x = X.selected.feature.list[[1]], y = Y.list[[1]], alpha = 0.5, nfolds = nfolds, foldid = flds)
    Yhat.elast = exp(predict(model.elast, newx = X.selected.feature.list[[2]], s = "lambda.min"))
    mae.elast = mean(abs(exp(Y.list[[2]])-Yhat.elast))
    corr.elast = cor(Yhat.elast, exp(Y.list[[2]]))
    # mae.class.elast = sapply(levels(label.list[[2]]), function(ix) mean(abs(Y.list[[2]][label.list[[2]]==ix]-Yhat.elast[label.list[[2]]==ix])))
    # corr.class.elast = sapply(levels(label.list[[2]]), function(ix) cor(Yhat.elast[label.list[[2]]==ix], Y.list[[2]][label.list[[2]]==ix]))
    # ---------------------------------------
    
    method.names = c("mae.rf", "mae.ridge", "mae.elast", "mae.lasso", "corr.rf", "corr.ridge", "corr.elast", "corr.lasso")
    
    file.name = c("ADNI1+t=", as.character(t), ".csv")
    file.name = paste(file.name, collapse ="")
    write.table(t(c(mae.rf, mae.ridge, mae.elast, mae.lasso, corr.rf, corr.ridge, corr.elast, corr.lasso)), file = file.name, sep = ',', append = T, col.names = ifelse(rep(file.exists(file.name), 8), F, method.names), row.names = F)
    
    # file1.name = c("ADNI1+class+t=", as.character(t), ".csv")
    # file1.name = paste(file1.name, collapse ="")
    
    # write.table(t(c(mae.class.rf, mae.class.ridge, mae.class.elast, mae.class.lasso, corr.class.rf, corr.class.ridge, corr.class.elast, corr.class.lasso)[c(c(1,4,7,10), c(1,4,7,10)+1, c(1,4,7,10)+2, c(13,16,19,22), c(13,16,19,22)+1, c(13,16,19,22)+2)]), file = file1.name, sep = ',', append = T, col.names = F, row.names = F)
    }
}
# }
