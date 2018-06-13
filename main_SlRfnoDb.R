# --------------- kill devil reading shell command ---------------- #
args = (commandArgs(TRUE))
# cat(args, "\n")
for (i in 1:length(args)) {
  eval(parse(text = args[[i]]))
}
# ------------------------------------------------------------------ #

library(caret)
library(methods) # "is function issue by Rscript"
library(glmnet)
library(randomForest)

# load data...
source("/netscr/peiyao/np/XipmedY3T/loaddataXipmedY3T.R")
source("/netscr/peiyao/np/fun.rfguided.R")
source("/netscr/peiyao/np/fun.noDb.R")
source("/netscr/peiyao/np/ordinlog1.R")


# ----------------------------------------- creating training and testing set ------------------------------------------------ #
# Sl = score2status(Y)

n = dim(X)[1]
p = dim(X)[2]

set.seed(myseed)
idtrain = unlist(createDataPartition(label, times = 1, p = 3/4))
idtest = (1:n)[-idtrain]

id = list(idtrain, idtest)
Y.list = lapply(1:2, function(x) Y[id[[x]]]) # train,test
X.list = lapply(1:2, function(x) X[id[[x]],]) 
label.list = lapply(1:2, function(x) label[id[[x]]])
# Sl.list = lapply(1:2, function(x) Sl[id[[x]]])

#---------- do scale ----------#
# constant bw should do scale
X1.mean = apply(X.list[[1]], 2, mean)
X1.sd = apply(X.list[[1]], 2, sd)
X1.sd = sapply(X1.sd, function(x) ifelse(x<1e-5, 1, x))
X.list[[1]] = t(apply(X.list[[1]], 1, function(x) (x - X1.mean)/X1.sd))
X.list[[2]] = t(apply(X.list[[2]], 1, function(x) (x - X1.mean)/X1.sd))

# ----------------------------Calculating interaction term-------------------------------#
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

# computing distance correlation to select favorite number of features including interaction features 
library(energy)
p_dc = 200 # number of features to be selected

# 1
dc.vec = apply(X.plus.inter.list[[1]], 2, function(feature) dcor(Y.list[[1]], feature))
X.selected.feature.list = lapply(1:2, function(ix) X.plus.inter.list[[ix]][ , order(dc.vec, decreasing = TRUE)[1:p_dc]]) # selected design matrix including interaction terms

# 2
#X.selected.feature.list = lapply(1:2, function(ix) X.interaction.list[[ix]][ , order(dc.vec, decreasing = TRUE)[1:p_dc]]) # selected design matrix including interaction terms

# 3 # include code in 2
#X.selected.feature.list = lapply(1:2, function(ix) cbind(X.list[[ix]], X.selected.feature.list[[ix]]))

# add to the last...
feature.ncol = ncol(X.selected.feature.list[[1]])
colnames(X.selected.feature.list[[1]]) = as.character(seq(1, feature.ncol))
colnames(X.selected.feature.list[[2]]) = as.character(seq(1, feature.ncol))
# ---------------------------------------------------------------------------------------------------------------------------- #
nfolds.log = 5 # ordinal logistic: Sl
nfolds.np = 5 # SlRf
nfolds.llr = 5 # local linear regression

alpha0 = 0
gamma.vec = exp(rev(seq(-2, 7, length.out = 50)))

initial.x = c(seq(-2, 2, length.out = length(levels(label.list[[1]]))-1), rep(0,p), rep(1,p))
# generating initial point for optimization, first K-1 parameters are thetas, then the first coming p are coefficients the last p are slack variable

measure.type = "corr"

if (alpha == 0){
  lambda.vec = c(1e6, exp(rev(seq(-2, 7, length.out = 99))))
}else{
  lambda.vec = c(1e6, exp(rev(seq(-7, 2, length.out = 99))))
}

ordinlog.list = cv.ordinlog.en(label.list[[1]], X.list[[1]], Y.list[[1]], gamma.vec, alpha0, initial.x, nfolds.log, "corr")
ordin.ml = ordinlog.list$ordin.ml

sl.list = lapply(1:2, function(x) as.vector(X.list[[x]]%*%ordin.ml$w))

Di.vec = seq(sd(sl.list[[1]])/5, sd(sl.list[[1]])*2, length.out = 20)

# -----------------------------without interaction term--------------------------------- #
par.list = cv.bothPen.noDb(label.list[[1]], X.selected.feature.list[[1]], Y.list[[1]], lambda.vec, alpha, nfolds.llr, sl.list[[1]], Di.vec)

Di.selected = par.list$Di
lambda.selected = par.list$lambda
id.which = par.list$id.which

if(id.which == 1){
  ml.rf = randomForest(x = X.selected.feature.list[[1]], y = Y.list[[1]], keep.inbag = T, ntree = 100)
  wrf.list = rf.weight(ml.rf, X.selected.feature.list[[1]], X.selected.feature.list[[2]])
  mymethod.res = SlRf.weight.noDb(wrf.list, Y.list[[1]], sl.list[[1]], sl.list[[2]], Di.selected)
}else{
  mymethod.res = slnp.noDb(X.selected.feature.list[[1]], Y.list[[1]], sl.list[[1]], X.selected.feature.list[[2]], Y.list[[2]], sl.list[[2]], Di.selected)
  # SlRf.weight.noDb(wrf.list, Y.list[[1]], sl.list[[1]], sl.list[[2]], Di.selected)
}

Yhat.mymethod = mymethod.res$Yhat
rwrf.list = mymethod.res$w.list
pom.list = penalized.origin.method(X.selected.feature.list[[1]], Y.list[[1]], X.selected.feature.list[[2]], rwrf.list, lambda.selected, alpha)
Yhat.mymethodPen = pom.list$Yhat
# betahat.mymethodPen = pom.list$betahat
# -------------------------------------------------------------------------------------- #

# -------------------------------with interaction term---------------------------------- #
# par.list = cv.SlPen.noDb(label.list[[1]], X.selected.feature.list[[1]], Y.list[[1]], lambda.vec, alpha, nfolds.llr, sl.list[[1]], Di.vec)
# Di.selected = par.list$Di
# lambda.selected = par.list$lambda
# mymethod.res = slnp.noDb(X.selected.feature.list[[1]], Y.list[[1]], sl.list[[1]], X.selected.feature.list[[2]], Y.list[[2]], sl.list[[2]], Di.selected)
# Yhat.mymethod = mymethod.res$Yhat
# rwrf.list = mymethod.res$w.list
# pom.list = penalized.origin.method(X.selected.feature.list[[1]], Y.list[[1]], X.selected.feature.list[[2]], rwrf.list, lambda.selected, alpha)
# Yhat.mymethodPen = pom.list$Yhat
# ------------------------------------------------------------------------------------ #

# ---------------mae and corr---------------- #
mae.mymethod = mean(abs(Yhat.mymethod - Y.list[[2]]))
corr.mymethod = cor(Yhat.mymethod, Y.list[[2]])
mae.mymethodPen = mean(abs(Yhat.mymethodPen - Y.list[[2]]))
corr.mymethodPen = cor(Yhat.mymethodPen, Y.list[[2]])
# 
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
# 

method.names = c("mae.mymethod", "mae.mymethodPen", "corr.mymethod", "corr.mymethodPen", "Di", "lambda", "id.which")
file.name = c("ADNI1+t=", as.character(t), "+alpha0=", as.character(alpha0), "+alpha=", as.character(alpha),".csv")
file.name = paste(file.name, collapse ="")
write.table(t(c(mae.mymethod, mae.mymethodPen, corr.mymethod, corr.mymethodPen, Di.selected, lambda.selected, id.which)), file = file.name, sep = ',', append = T, col.names = ifelse(rep(file.exists(file.name), 7), F, method.names), row.names = F)

# file1.name = c("ADNI1+class+t=", as.character(t), "+alpha0=", as.character(alpha0), "+alpha=", as.character(alpha),".csv") 
# file1.name = paste(file1.name, collapse ="")
# 
# write.table(t(c(mae.class.mymethod, mae.class.mymethodPen, corr.class.mymethod, corr.class.mymethodPen)[c(c(1,4), c(1,4)+1, c(1,4)+2, c(7,10), c(7,10)+1, c(7,10)+2)]), file = file1.name, sep = ',', append = T, col.names = F, row.names = F)
# 
