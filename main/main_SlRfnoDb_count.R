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

X.list[[1]] = sweep(X.list[[1]], 2, X1.mean)
X.list[[1]] = sweep(X.list[[1]], 2, X1.sd, "/")

X.list[[2]] = sweep(X.list[[2]], 2, X1.mean)
X.list[[2]] = sweep(X.list[[2]], 2, X1.sd, "/")

print("Finish scaling features")
# -------------------------------------------------------------------------------


# ----------------------------------------- main -------------------------------------------
# ------------- parameters --------------
nfolds.log = 5 # ordinal logistic: Sl
nfolds.llr = 5 # local linear regression

alpha0 = 0
gamma.vec = exp(rev(seq(-2, 7, length.out = 50)))
# initial point for optimization, first K-1 parameters are thetas, then the first coming p are coefficients the last p are slack variable
initial.x = c(seq(-2, 2, length.out = length(levels(label.list[[1]]))-1), rep(0,p), rep(1,p))

measure.type = "corr"
# if (alpha == 0){
#   lambda.vec = c(1e6, exp(rev(seq(-2, 7, length.out = 99))))
# }else{
#   lambda.vec = c(1e6, exp(rev(seq(-7, 2, length.out = 99))))
# }
# ----------------------------------------

# ------------------------ ordinal logistic: Sl --------------------------
ordinlog.list = cv.ordinlog.en(label.list[[1]], X.list[[1]], Y.list[[1]], gamma.vec, alpha0, initial.x, nfolds.log, "corr")
ordin.ml = ordinlog.list$ordin.ml
sl.list = lapply(1:2, function(x) as.vector(X.list[[x]]%*%ordin.ml$w))

# delete outliers
sl.list = lapply(1:2, function(ix){
  sl.list[[ix]][sl.list[[ix]] > boxplot(sl.list[[ix]], plot = F)$stats[5,1]] = boxplot(sl.list[[ix]], plot = F)$stats[5,1]
  sl.list[[ix]][sl.list[[ix]] < boxplot(sl.list[[ix]], plot = F)$stats[1,1]] = boxplot(sl.list[[ix]], plot = F)$stats[1,1]
  sl.list[[ix]]
})

# tuning parameter Di for SlRf
Di.vec = seq(sd(sl.list[[1]])/5, sd(sl.list[[1]])*2, length.out = 20)
print("Finish ordinal logistic regression")
# ------------------------------------------------------------------------

# ----------------------------- SlRf LWPR --------------------------------- 
# par.list = cv.bothPen.noDb(label.list[[1]], X.list[[1]], Y.list[[1]], lambda.vec, alpha, nfolds.llr, sl.list[[1]], Di.vec)
par.list = cv.bothPen.noDb.nolambda(label.list[[1]], X.list[[1]], Y.list[[1]], alpha, nfolds.llr, sl.list[[1]], Di.vec)

Di.selected = par.list$Di
lam.id = 50
id.which = par.list$id.which
print("Finish SlRf cross validation")

# Di.selected = par.list$Di
# lambda.selected = par.list$lambda
# id.which = par.list$id.which
# print("Finish SlRf cross validation")

if(id.which == 1){
  ml.rf = randomForest(x = X.list[[1]], y = Y.list[[1]], keep.inbag = T, ntree = 100)
  wrf.list = rf.weight(ml.rf, X.list[[1]], X.list[[2]])
  mymethod.res = SlRf.weight.noDb(wrf.list, Y.list[[1]], sl.list[[1]], sl.list[[2]], Di.selected)
}else{
  mymethod.res = slnp.noDb(X.list[[1]], Y.list[[1]], sl.list[[1]], X.list[[2]], Y.list[[2]], sl.list[[2]], Di.selected)
}
print("Finish local fitting without penalization")

Yhat.mymethod = mymethod.res$Yhat
rwrf.list = mymethod.res$w.list
pom.list = predict.penalized.origin.method.nolambda(X.list[[1]], Y.list[[1]], X.list[[2]], rwrf.list, lam.id, alpha)
Yhat.mymethodPen = pom.list$Yhat
betahat.mymethodPen = pom.list$betahat
print("Finish local fitting with penalization")

# Yhat.mymethod = mymethod.res$Yhat
# rwrf.list = mymethod.res$w.list
# pom.list = penalized.origin.method(X.list[[1]], Y.list[[1]], X.list[[2]], rwrf.list, lambda.selected, alpha)
# Yhat.mymethodPen = pom.list$Yhat
# betahat.mymethodPen = pom.list$betahat
# print("Finish local fitting with penalization")
# --------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

# ----------------------------------- compute the important features ----------------------------------------
# create several fake labels according to different intervels on the soft labels
thresholds.vec = quantile(sl.list[[2]], probs = seq(0, 1, 0.2))

fake.label = (sl.list[[2]] < thresholds.vec[2])*1 + (sl.list[[2]] >= thresholds.vec[2] & sl.list[[2]] < thresholds.vec[3])*2 + (sl.list[[2]] >= thresholds.vec[3] & sl.list[[2]] < thresholds.vec[4])*3 + (sl.list[[2]] >= thresholds.vec[4] & sl.list[[2]] < thresholds.vec[5])*4 + (sl.list[[2]] >= thresholds.vec[5])*5

# percentage #
nonzero.mtx = apply(betahat.mymethodPen, 2, function(x) abs(x)>1e-6) # p by n.test
count.vec.list = lapply(unique(fake.label), function(ix) apply(nonzero.mtx[,fake.label == ix], 1, sum))
n.count.vec.list = lapply(unique(fake.label), function(ix) length(fake.label[fake.label == ix]))
p.vec.list = lapply(unique(fake.label), function(ix) count.vec.list[[ix]]/n.count.vec.list[[ix]])

file.name = c("ADNI1+G1+t=", as.character(t),".csv")
file.name = paste(file.name, collapse ="")
write.table(t(p.vec.list[[1]]), file = file.name, sep = ',', append = T, col.names = F, row.names = F)

file.name = c("ADNI1+G2+t=", as.character(t),".csv")
file.name = paste(file.name, collapse ="")
write.table(t(p.vec.list[[2]]), file = file.name, sep = ',', append = T, col.names = F, row.names = F)

file.name = c("ADNI1+G3+t=", as.character(t),".csv")
file.name = paste(file.name, collapse ="")
write.table(t(p.vec.list[[3]]), file = file.name, sep = ',', append = T, col.names = F, row.names = F)

file.name = c("ADNI1+G4+t=", as.character(t),".csv")
file.name = paste(file.name, collapse ="")
write.table(t(p.vec.list[[4]]), file = file.name, sep = ',', append = T, col.names = F, row.names = F)

file.name = c("ADNI1+G5+t=", as.character(t),".csv")
file.name = paste(file.name, collapse ="")
write.table(t(p.vec.list[[5]]), file = file.name, sep = ',', append = T, col.names = F, row.names = F)

write.table(table(label.list[[2]], fake.label), file = paste(c("DX_distribution+t=", as.character(t), ".csv"), collapse =""), sep = ',', append = T, col.names = F, row.names = F)




# ------------------------------------------------------------------------------------------------------------
