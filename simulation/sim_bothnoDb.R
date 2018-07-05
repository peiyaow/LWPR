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

load("/nas/longleaf/home/peiyao/LWPR/simulation/parameter/diag_cov/sim_par21.RData")
source("/nas/longleaf/home/peiyao/LWPR/function/fun.rfguided.R")
source("/nas/longleaf/home/peiyao/LWPR/function/fun.noDb.R")
source("/nas/longleaf/home/peiyao/LWPR/function/ordinlog1.R")
source("/nas/longleaf/home/peiyao/LWPR/simulation/sim.fun.R")

set.seed(myseed)
data.list = lapply(1:2, function(ix) mysimulation4(n, p1, p2, p3, pc, p0, Sigma_1, Sigma_2, Sigma_3, Sigma_c, Sigma_0, rho_e, w))

X.list = lapply(1:2, function(ix) data.list[[ix]]$X)
Y.list = lapply(1:2, function(ix) data.list[[ix]]$Y)
label.list = lapply(1:2, function(ix) data.list[[ix]]$label)
  
# X.list = list()
# X.list[[1]] = X
# X.list[[2]] = X.test
# Y.list = list()
# Y.list[[1]] = Y
# Y.list[[2]] = Y.test
# label.list = list()
# label.list[[1]] = label
# label.list[[2]] = label.test

# ------------- parameters --------------
nfolds.log = 5 # ordinal logistic: Sl
nfolds.llr = 5 # local linear regression

alpha0 = 0
gamma.vec = exp(rev(seq(-2, 7, length.out = 50)))
# initial point for optimization, first K-1 parameters are thetas, then the first coming p are coefficients the last p are slack variable
initial.x = c(seq(-4, 4, length.out = length(levels(label.list[[1]]))-1), rep(0,p), rep(1,p))

measure.type = "corr"
if (alpha == 0){
  lambda.vec = c(1e6, exp(rev(seq(-2, 7, length.out = 99))))
}else{
  lambda.vec = c(1e6, exp(rev(seq(-7, 2, length.out = 99))))
}
# ----------------------------------------


# # --------------------------------setting parameters------------------------------- #
# nfolds.log = 5 # ordinal logistic: Sl
# nfolds.np = 5 # SlRf
# nfolds.llr = 5 # local linear regression
# 
# # if (alpha0){
# #   gamma.vec = exp(rev(seq(-7, 2, length.out = 50)))
# # }else{
# #   gamma.vec = exp(rev(seq(-2, 7, length.out = 50))) 
# # }
# 
# alpha0 = 0
# gamma.vec = exp(rev(seq(-2, 7, length.out = 50))) 
# 
# initial.x = c(seq(-4, 4, length.out = length(levels(label.list[[1]]))-1), rep(0,p), rep(1,p))
# # generating initial point for optimization, first K-1 parameters are thetas, last p are coefficients 
# 
# measure.type = "corr"
# 
# if (alpha == 1){
#   lambda.vec = exp(rev(seq(-7, 2, length.out = 100)))
# }else if(alpha == 0){
#   lambda.vec = exp(rev(seq(-2, 7, length.out = 100))) 
# }else{
#   lambda.vec = exp(rev(seq(-7, 2, length.out = 100)))
# }

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
# ------------------------------------------------------------------------

# ----------------------------- SlRf LWPR --------------------------------- 
par.list = cv.bothPen.noDb(label.list[[1]], X.list[[1]], Y.list[[1]], lambda.vec, alpha, nfolds.llr, sl.list[[1]], Di.vec)
Di.selected = par.list$Di
lambda.selected = par.list$lambda
id.which = par.list$id.which
print("Finish SlRf cross validation")

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
pom.list = penalized.origin.method(X.list[[1]], Y.list[[1]], X.list[[2]], rwrf.list, lambda.selected, alpha)
Yhat.mymethodPen = pom.list$Yhat
print("Finish local fitting with penalization")
# --------------------------------------------------------------------------

# ------------------------------ Caculating results ----------------------------------------
# -------------------- mae and corr --------------------------
mae.mymethod = mean(abs(Yhat.mymethod - Y.list[[2]]))
corr.mymethod = cor(Yhat.mymethod, Y.list[[2]])
mae.mymethodPen = mean(abs(Yhat.mymethodPen - Y.list[[2]]))
corr.mymethodPen = cor(Yhat.mymethodPen, Y.list[[2]])
# ------------------------------------------------------------
# -------------------------------------------------------------------------------------------

method.names = c("mae.mymethod", "mae.mymethodPen", "corr.mymethod", "corr.mymethodPen", "Di", "id.which", "seed")

file.name = c("mae+corr+sim+alpha0=", as.character(alpha0), "+alpha=", as.character(alpha), ".csv") 
file.name = paste(file.name, collapse ="")

write.table(t(c(mae.mymethod, mae.mymethodPen, corr.mymethod, corr.mymethodPen, Di.selected, id.which, myseed)), file = file.name, sep = ',', append = T, col.names = ifelse(rep(file.exists(file.name), 7), F, method.names), row.names = F)
