# ---------------------- reading shell command --------------------- 
args = (commandArgs(TRUE))
# cat(args, "\n")
for (i in 1:length(args)) {
  eval(parse(text = args[[i]]))
}
# ------------------------------------------------------------------ 

library(dplyr)
library(readr)
library(ltjmm)
library(caret)
library(doParallel)
library(energy)

#setwd("/Users/MonicaW/Documents/GitHub/LWPR/data")
#setwd("C:/Users/peiyao/Documents/GitHub/LWPR/data")
X0 = as.matrix(read_table("/nas/longleaf/home/peiyao/LWPR/data/X1a.txt", col_names = F))
Y0 = as.matrix(read_table("/nas/longleaf/home/peiyao/LWPR/data/Y5T.txt", col_names = F))[, 4+5*(c(1,2,3)-1)]
label0 = as.ordered(read_table("/nas/longleaf/home/peiyao/LWPR/data/label1.txt", col_names = F)$X1)

#load("data.RData")

# change negatives to na in Y
Y0 = apply(Y0, 2, function(x) {x[x<0] = NA 
x})
allnan = apply(Y0, 1, function(row) !prod(is.na(row)))
Y0 = Y0[allnan, ]
Y0 = log(Y0+1)

Xs = X0[allnan, 1:186]
label0 = label0[allnan]
# within group median , each element from the list is the median within the group 
# list of length 3, each element 1 by 186 group median of each feature
Xs.med.list = lapply(levels(label0), function(x) apply(Xs[label0 == x, ], 2, function(y) median(y, na.rm = T)))

# impute with median
Xs.missing = is.na(Xs)
for (i in 1:nrow(Xs)){
  ip.med = Xs.med.list[[as.numeric(label0[i])]]
  Xs[i, Xs.missing[i, ]] = ip.med[Xs.missing[i, ]]
}

n = nrow(Xs)
p = ncol(Xs)
set.seed(myseed)
idtrain = unlist(createDataPartition(label0, times = 1, p = 3/4))
idtest = (1:n)[-idtrain]
id_all = seq(1,n)

X1.mean = apply(Xs[idtrain,], 2, mean)
X1.sd = apply(Xs[idtrain,], 2, sd)
# if the std is really small just subtract the mean in the following step 
X1.sd = sapply(X1.sd, function(x) ifelse(x<1e-5, 1, x)) 
Xs[idtrain,] = t(apply(Xs[idtrain,], 1, function(x) (x - X1.mean)/X1.sd))
Xs[idtest,] = t(apply(Xs[idtest,], 1, function(x) (x - X1.mean)/X1.sd))

# interaction
X.interaction.list = list()
k = 0
for (i in 1:p){
  for (j in 1:p){
    if (i < j){
      k = k+1
      X.interaction.list[[k]] = Xs[,i]*Xs[,j]
    }
  }
}
X.interaction = do.call(cbind, X.interaction.list)
X1.interaction.mean = apply(X.interaction[idtrain,], 2, mean)
X1.interaction.sd = apply(X.interaction[idtrain,], 2, sd)
X1.interaction.sd = sapply(X1.interaction.sd, function(x) ifelse(x<1e-5, 1, x)) 
X.interaction[idtrain, ]= t(apply(X.interaction[idtrain, ], 1, function(x) (x - X1.interaction.mean)/X1.interaction.sd))
X.interaction[idtest, ] = t(apply(X.interaction[idtest, ], 1, function(x) (x - X1.interaction.mean)/X1.interaction.sd))
X.plus.inter = cbind(Xs, X.interaction)

p_dc = 200 
cl = makeCluster(4) # number of cores you can use
registerDoParallel(cl)

dc.vec = foreach(col_ix = 1:ncol(X.plus.inter[idtrain, ]), .packages = "energy", .combine = "c") %dopar% {
  dcor(Y0[idtrain], X.plus.inter[idtrain,][,col_ix])
}
stopCluster(cl)

X.selected.feature.id = order(dc.vec, decreasing = TRUE)[1:p_dc]
X.selected.feature = X.plus.inter[, X.selected.feature.id]

feature.ncol = ncol(X.selected.feature)
colnames(X.selected.feature) = as.character(seq(1, feature.ncol))

X.selected.feature.pca = prcomp(X.selected.feature)
X.selected.feature.pr_var = X.selected.feature.pca$sdev^2
X.selected.feature.prop_varex <- X.selected.feature.pr_var/sum(X.selected.feature.pr_var)
X.selected.feature.prop.explained = cumsum(X.selected.feature.prop_varex)
pc.num = seq(1,200)[X.selected.feature.prop.explained>.95][1]
X.selected.pc = X.selected.feature.pca$x[,1:pc.num]

id_year_Y.list = list()
for (i in 1:n){
  Y.tmp = Y0[i,][!is.na(Y0[i,])]
  year = c(0,1,2)[!is.na(Y0[i,])]
  id_year_Y.list[[i]] = cbind(id = id_all[i], year, Y = Y.tmp)
}
id_year_Y = as.data.frame(do.call(rbind, id_year_Y.list))
id_year_Y_outcome = data.frame(id_year_Y, outcome = 1)
id_X = as.data.frame(cbind(id = id_all, X.selected.pc))

dd = right_join(id_X, id_year_Y_outcome, by = 'id')


# fit ltjmm
setup <- ltjmm(Y ~ year | .-Y-outcome-year-id | id | outcome, data = dd[dd$id %in% idtrain,])
fit <- stan(file = file.path(.libPaths()[1], "ltjmm", "stan", "ltjmm.stan"),
            data = setup$data,
            pars = c('beta', 'delta', 'alpha0', 'alpha1', 'gamma',
                     'sigma_alpha0', 'sigma_alpha1', "sigma_delta", 'sigma_y', 'log_lik'),
            open_progress = FALSE, chains = 2, iter = 2000,
            warmup = 1000, thin = 1, cores = 4, control = list(max_treedepth = 15))
save.image("ltjmm_PCA.RData")







