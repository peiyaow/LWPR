# ---------------------- reading shell command --------------------- 
args = (commandArgs(TRUE))
# cat(args, "\n")
for (i in 1:length(args)) {
  eval(parse(text = args[[i]]))
}
# ------------------------------------------------------------------ 

library(caret)
library(methods) # "is function issue by Rscript"
library(glmnet)
library(randomForest)
library(energy)

# load data
source("/nas/longleaf/home/peiyao/LWPR/main/loaddataXipmedY3T.R")
source("/nas/longleaf/home/peiyao/LWPR/function/fun.rfguided.R")
source("/nas/longleaf/home/peiyao/LWPR/function/fun.noDb.R")
source("/nas/longleaf/home/peiyao/LWPR/function/ordinlog1.R")

# ---------------------- creating training and testing set ----------------------------- 
n = dim(X)[1]
p = dim(X)[2]

set.seed(myseed)
idtrain = unlist(createDataPartition(label, times = 1, p = 3/4))
idtest = (1:n)[-idtrain]

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
  X.plus.inter.list[[m]] = cbind(X.list[[m]], X.interaction.list[[m]])
}

# --- computing distance correlation to select favorite number of features including interaction features --- 
# number of features to be selected
p_dc = 200 
dc.vec = apply(X.plus.inter.list[[1]], 2, function(feature) dcor(Y.list[[1]], feature))
print("Finish calculation distance correlation")

X.selected.feature.id = order(dc.vec, decreasing = TRUE)[1:p_dc]
X.selected.feature.list = lapply(1:2, function(ix) X.plus.inter.list[[ix]][, X.selected.feature.id])
write.table(X.selected.feature.id, "feature_id.txt", sep="\t", row.names = F, col.names = F)
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
nfolds.np = 5 # SlRf
nfolds.llr = 5 # local linear regression

alpha0 = 0
gamma.vec = exp(rev(seq(-2, 7, length.out = 50)))
# initial point for optimization, first K-1 parameters are thetas, then the first coming p are coefficients the last p are slack variable
initial.x = c(seq(-2, 2, length.out = length(levels(label.list[[1]]))-1), rep(0,p), rep(1,p))

measure.type = "corr"
if (alpha == 0){
  lambda.vec = c(1e6, exp(rev(seq(-2, 7, length.out = 99))))
}else{
  lambda.vec = c(1e6, exp(rev(seq(-7, 2, length.out = 99))))
}
# ----------------------------------------

# ------------------------ ordinal logistic: Sl --------------------------
ordinlog.list = cv.ordinlog.en(label.list[[1]], X.list[[1]], Y.list[[1]], gamma.vec, alpha0, initial.x, nfolds.log, "corr")
ordin.ml = ordinlog.list$ordin.ml
sl.list = lapply(1:2, function(x) as.vector(X.list[[x]]%*%ordin.ml$w))
# tuning parameter Di for SlRf
Di.vec = seq(sd(sl.list[[1]])/5, sd(sl.list[[1]])*2, length.out = 20)

write.table(sl.list[[1]], "sl_train.txt", sep="\t", append = T, row.names = F, col.names = F)
write.table(sl.list[[2]], "sl_test.txt", sep="\t", append = T, row.names = F, col.names = F)
write.table(Y.list[[1]], "Y_train.txt", sep="\t", append = T, row.names = F, col.names = F)
write.table(Y.list[[2]], "Y_test.txt", sep="\t", append = T, row.names = F, col.names = F)
write.table(label.list[[1]], "label_train.txt", sep="\t", append = T, row.names = F, col.names = F)
write.table(label.list[[2]], "label_test.txt", sep="\t", append = T, row.names = F, col.names = F)
print("Finish ordinal logistic regression")
# ------------------------------------------------------------------------

