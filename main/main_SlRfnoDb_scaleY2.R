# use distance correlation selected feature for ordinal logistic

# ---------------------- reading shell command --------------------- 
args = (commandArgs(TRUE))
# cat(args, "\n")
for (i in 1:length(args)) {
  eval(parse(text = args[[i]]))
}
# ------------------------------------------------------------------ 

library(caret)
library(methods) # "is function issue by Rscript"
library(energy)
library(glmnet)
library(randomForest)
library(foreach)
library(doParallel)

# load data
source("/nas/longleaf/home/peiyao/LWPR/main/loaddataXipmedY3T.R")
source("/nas/longleaf/home/peiyao/LWPR/function/fun.rfguided.R")
source("/nas/longleaf/home/peiyao/LWPR/function/fun.noDb.R")
source("/nas/longleaf/home/peiyao/LWPR/function/ordinlog1.R")

name.id = paste0("t=", as.character(t), "+alpha=", as.character(alpha))
# ---------------------- creating training and testing set ----------------------------- 
n = dim(X)[1]
p = dim(X)[2]

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
# if the std is really small just subtract the mean in the following step 
X1.sd = sapply(X1.sd, function(x) ifelse(x<1e-5, 1, x)) 
X.list[[1]] = t(apply(X.list[[1]], 1, function(x) (x - X1.mean)/X1.sd))
X.list[[2]] = t(apply(X.list[[2]], 1, function(x) (x - X1.mean)/X1.sd))
print("Finish scaling features")
# -------------------------------------------------------------------------------

# ----------------------------- Calculating interaction term ------------------------------
X.interaction.list = list()
X.plus.inter.list = list()
for (m in 1:2){
  X.interaction.list[[m]] = list()
  k = 0
  for (i in 1:p){
    for (j in 1:p){
      if (i < j){
        k = k+1
        X.interaction.list[[m]][[k]] = X.list[[m]][,i]*X.list[[m]][,j]
      }
    }
  }
  X.interaction.list[[m]] = do.call(cbind, X.interaction.list[[m]])
  #  X.plus.inter.list[[m]] = cbind(X.list[[m]], X.interaction.list[[m]])
}
X1.interaction.mean = apply(X.interaction.list[[1]], 2, mean)
X1.interaction.sd = apply(X.interaction.list[[1]], 2, sd)
# if the std is really small just subtract the mean in the following step 
X1.interaction.sd = sapply(X1.interaction.sd, function(x) ifelse(x<1e-5, 1, x)) 
X.interaction.list[[1]] = t(apply(X.interaction.list[[1]], 1, function(x) (x - X1.interaction.mean)/X1.interaction.sd))
X.interaction.list[[2]] = t(apply(X.interaction.list[[2]], 1, function(x) (x - X1.interaction.mean)/X1.interaction.sd))
X.plus.inter.list = lapply(1:2, function(x) cbind(X.list[[x]], X.interaction.list[[x]]))


# --- computing distance correlation to select favorite number of features including interaction features --- 
# number of features to be selected
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
write.table(X.selected.feature.id, paste0("feature_id+", name.id, ".txt"), sep="\t", row.names = F, col.names = F, append = T)
print("Finish addig top interaction terms into design matrix")
# -----------------------------------------------------------------------------------------------------------

# add to the last... probably to fix a random forest bug
feature.ncol = ncol(X.selected.feature.list[[1]])
colnames(X.selected.feature.list[[1]]) = as.character(seq(1, feature.ncol))
colnames(X.selected.feature.list[[2]]) = as.character(seq(1, feature.ncol))
# ------------------------------------------------------------------------------------------

# ----------------------------------------- main -------------------------------------------
# ------------- parameters --------------
nfolds.log = 5 # ordinal logistic: Sl
nfolds.llr = 5 # local linear regression

alpha0 = 0
gamma.vec = exp(rev(seq(-2, 7, length.out = 50)))
# initial point for optimization, first K-1 parameters are thetas, then the first coming p are coefficients the last p are slack variable
initial.x = c(seq(-2, 2, length.out = length(levels(label.list[[1]]))-1), rep(0,p_dc), rep(1,p_dc))

measure.type = "corr"
if (alpha == 0){
  lambda.vec = c(1e6, exp(rev(seq(-2, 7, length.out = 99))))
}else{
  lambda.vec = c(1e6, exp(rev(seq(-7, 2, length.out = 99))))
}
# ----------------------------------------

# ------------------------ ordinal logistic: Sl --------------------------
ordinlog.list = cv.ordinlog.en(label.list[[1]], X.selected.feature.list[[1]], Y.list[[1]], gamma.vec, alpha0, initial.x, nfolds.log, "corr")
ordin.ml = ordinlog.list$ordin.ml
sl.list = lapply(1:2, function(x) as.vector(X.selected.feature.list[[x]]%*%ordin.ml$w))

# delete outliers
sl.list = lapply(1:2, function(ix){
  sl.list[[ix]][sl.list[[ix]] > boxplot(sl.list[[ix]], plot = F)$stats[5,1]] = boxplot(sl.list[[ix]], plot = F)$stats[5,1]
  sl.list[[ix]][sl.list[[ix]] < boxplot(sl.list[[ix]], plot = F)$stats[1,1]] = boxplot(sl.list[[ix]], plot = F)$stats[1,1]
  sl.list[[ix]]
})

# tuning parameter Di for SlRf
Di.vec = seq(sd(sl.list[[1]])/5, sd(sl.list[[1]])*2, length.out = 20)


paste0("sl_train+", name.id, ".txt")
write.table(sl.list[[1]], paste0("sl_train+", name.id, ".txt"), sep="\t", append = T, row.names = F, col.names = F)
write.table(sl.list[[2]], paste0("sl_test+", name.id, ".txt"), sep="\t", append = T, row.names = F, col.names = F)
write.table(Y.list[[1]], paste0("Y_train+", name.id, ".txt"), sep="\t", append = T, row.names = F, col.names = F)
write.table(Y.list[[2]], paste0("Y_test+", name.id, ".txt"), sep="\t", append = T, row.names = F, col.names = F)
write.table(label.list[[1]], paste0("label_train+", name.id, ".txt"), sep="\t", append = T, row.names = F, col.names = F)
write.table(label.list[[2]], paste0("label_test+", name.id, ".txt"), sep="\t", append = T, row.names = F, col.names = F)
print("Finish ordinal logistic regression")
# ------------------------------------------------------------------------

# ----------------------------- SlRf LWPR --------------------------------- 
par.list = cv.bothPen.noDb(label.list[[1]], X.selected.feature.list[[1]], Y.list[[1]], lambda.vec, alpha, nfolds.llr, sl.list[[1]], Di.vec)

Di.selected = par.list$Di
lambda.selected = par.list$lambda
id.which = par.list$id.which
print("Finish SlRf cross validation")

if(id.which == 1){
  ml.rf = randomForest(x = X.selected.feature.list[[1]], y = Y.list[[1]], keep.inbag = T, ntree = 100)
  wrf.list = rf.weight(ml.rf, X.selected.feature.list[[1]], X.selected.feature.list[[2]])
  mymethod.res = SlRf.weight.noDb(wrf.list, Y.list[[1]], sl.list[[1]], sl.list[[2]], Di.selected)
}else{
  mymethod.res = slnp.noDb(X.selected.feature.list[[1]], Y.list[[1]], sl.list[[1]], X.selected.feature.list[[2]], Y.list[[2]], sl.list[[2]], Di.selected)
}
print("Finish local fitting without penalization")

Yhat.mymethod = mymethod.res$Yhat
rwrf.list = mymethod.res$w.list
pom.list = penalized.origin.method(X.selected.feature.list[[1]], Y.list[[1]], X.selected.feature.list[[2]], rwrf.list, lambda.selected, alpha)
Yhat.mymethodPen = pom.list$Yhat
print("Finish local fitting with penalization")
# --------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

# ------------------------------ Caculating results ----------------------------------------
# -------------------- mae and corr --------------------------
Yhat.mymethod = exp(Yhat.mymethod)
Yhat.mymethodPen = exp(Yhat.mymethodPen)

mae.mymethod = mean(abs(Yhat.mymethod - exp(Y.list[[2]])))
corr.mymethod = cor(Yhat.mymethod, exp(Y.list[[2]]))
mae.mymethodPen = mean(abs(Yhat.mymethodPen - exp(Y.list[[2]])))
corr.mymethodPen = cor(Yhat.mymethodPen, exp(Y.list[[2]]))
# ------------------------------------------------------------

method.names = c("mae.mymethod", "mae.mymethodPen", "corr.mymethod", "corr.mymethodPen", "Di", "lambda", "id.which", "seed")
file.name = paste0("ADNI1+", name.id, ".csv")
write.table(t(c(mae.mymethod, mae.mymethodPen, corr.mymethod, corr.mymethodPen, Di.selected, lambda.selected, id.which, myseed)), file = file.name, sep = ',', append = T, col.names = ifelse(rep(file.exists(file.name), 8), F, method.names), row.names = F)
print("Finish all")
# -------------------------------------------------------------------------------------------


# mae.class.mymethod = sapply(levels(label.list[[2]]), function(ix) mean(abs(Y.list[[2]][label.list[[2]]==ix]-Yhat.mymethod[label.list[[2]]==ix])))
# corr.class.mymethod = sapply(levels(label.list[[2]]), function(ix) cor(Yhat.mymethod[label.list[[2]]==ix], Y.list[[2]][label.list[[2]]==ix]))
# mae.class.mymethodPen = sapply(levels(label.list[[2]]), function(ix) mean(abs(Y.list[[2]][label.list[[2]]==ix]-Yhat.mymethodPen[label.list[[2]]==ix])))
# corr.class.mymethodPen = sapply(levels(label.list[[2]]), function(ix) cor(Yhat.mymethodPen[label.list[[2]]==ix], Y.list[[2]][label.list[[2]]==ix]))
# 
# method.names = c("mae.mymethod", "mae.mymethodPen", "corr.mymethod", "corr.mymethodPen", "Di", "id.which")

# 
# file.name = c("ADNI1+t=", as.character(t), "+alpha0=", as.character(alpha0), "+alpha=", as.character(alpha),".csv") 
# file.name = paste(file.name, collapse ="")
# 
# write.table(t(c(mae.mymethod, mae.mymethodPen, corr.mymethod, corr.mymethodPen, Di.selected, id.which)), file = file.name, sep = ',', append = T, col.names = ifelse(rep(file.exists(file.name), 6), F, method.names), row.names = F)

# file1.name = c("ADNI1+class+t=", as.character(t), "+alpha0=", as.character(alpha0), "+alpha=", as.character(alpha),".csv") 
# file1.name = paste(file1.name, collapse ="")
# 
# write.table(t(c(mae.class.mymethod, mae.class.mymethodPen, corr.class.mymethod, corr.class.mymethodPen)[c(c(1,4), c(1,4)+1, c(1,4)+2, c(7,10), c(7,10)+1, c(7,10)+2)]), file = file1.name, sep = ',', append = T, col.names = F, row.names = F)
# 
