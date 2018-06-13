# --------------- reading shell command ---------------- #
args = (commandArgs(TRUE))
for (i in 1:length(args)) {
  eval(parse(text = args[[i]]))
}
# ------------------------------------------------------------------ #

library(caret)
library(methods) # "is function issue by Rscript"
library(glmnet)
library(randomForest)

# load data
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

nfolds.log = 5 # ordinal logistic: Sl
nfolds.np = 5 # SlRf
nfolds.llr = 5 # local linear regression

alpha0 = 0
gamma.vec = exp(rev(seq(-2, 7, length.out = 50)))

initial.x = c(seq(-2, 2, length.out = length(levels(label.list[[1]]))-1), rep(0,p), rep(1,p))
# generating initial point for optimization, first K-1 parameters are thetas, then the first coming p are coefficients the last p are slack variable

measure.type = "corr"

lambda.vec = exp(rev(seq(-7, 2, length.out = 100)))

ordinlog.list = cv.ordinlog.en(label.list[[1]], X.list[[1]], Y.list[[1]], gamma.vec, alpha0, initial.x, nfolds.log, "corr")
ordin.ml = ordinlog.list$ordin.ml

sl.list = lapply(1:2, function(x) as.vector(X.list[[x]]%*%ordin.ml$w))

Di.vec = seq(sd(sl.list[[1]])/5, sd(sl.list[[1]])*2, length.out = 20)

# -----------------------------without interaction term--------------------------------- #
par.list = cv.bothPen.noDb(label.list[[1]], X.list[[1]], Y.list[[1]], lambda.vec, alpha, nfolds.llr, sl.list[[1]], Di.vec)

Di.selected = par.list$Di
lambda.selected = par.list$lambda
id.which = par.list$id.which

if(id.which == 1){
  ml.rf = randomForest(x = X.list[[1]], y = Y.list[[1]], keep.inbag = T, ntree = 100)
  wrf.list = rf.weight(ml.rf, X.list[[1]], X.list[[2]])
  mymethod.res = SlRf.weight.noDb(wrf.list, Y.list[[1]], sl.list[[1]], sl.list[[2]], Di.selected)
}else{
  mymethod.res = slnp.noDb(X.list[[1]], Y.list[[1]], sl.list[[1]], X.list[[2]], Y.list[[2]], sl.list[[2]], Di.selected)
}

Yhat.mymethod = mymethod.res$Yhat
rwrf.list = mymethod.res$w.list
pom.list = penalized.origin.method(X.list[[1]], Y.list[[1]], X.list[[2]], rwrf.list, lambda.selected, alpha)
Yhat.mymethodPen = pom.list$Yhat
betahat.mymethodPen = pom.list$betahat
# -------------------------------------------------------------------------------------- #

# -----------------------------------compute the important features----------------------------------------#
# create several fake labels according to different intervels on the soft labels

thresholds.vec = quantile(Y.list[[2]], probs = seq(0, 1, 0.2))

fake.label = (Y.list[[2]] < thresholds.vec[2])*1 + (Y.list[[2]] >= thresholds.vec[2] & Y.list[[2]] < thresholds.vec[3])*2 + (Y.list[[2]] >= thresholds.vec[3] & Y.list[[2]] < thresholds.vec[4])*3 + (Y.list[[2]] >= thresholds.vec[4] & Y.list[[2]] < thresholds.vec[5])*4 + (Y.list[[2]] >= thresholds.vec[5])*5

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
# --------------------------------------------------------------------------------------------------------#
