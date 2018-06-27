library(dplyr)
setwd("/Users/MonicaW/Documents/GitHub/LWPR/data")
X0 = as.matrix(read_table("X1a.txt", col_names = F))
Y0 = as.matrix(read_table("Y5T.txt", col_names = F))[, 4+5*(c(1,2,3)-1)]
label0 = as.ordered(read_table("label1.txt", col_names = F)$X1)

# change negatives to na in Y
Y0 = apply(Y0, 2, function(x) {x[x<0] = NA 
x})
allnan = apply(Y0, 1, function(row) !prod(is.na(row)))
Y0 = Y0[allnan, ]

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



id = sample(805,200)
X = Xs[id,]
Y = Y0[id,]
n = nrow(X)

id_year_Y.list = list()
for (i in 1:n){
  Y.tmp = Y[i,][!is.na(Y[i,])]
  year = c(0,1,2)[!is.na(Y[i,])]
  id_year_Y.list[[i]] = cbind(id = id[i], year, Y = Y.tmp)
}
id_year_Y = as.data.frame(do.call(rbind, id_year_Y.list))
id_year_Y_outcome = data.frame(id_year_Y, outcome = 1)
id_X = as.data.frame(cbind(id, X))

dd = right_join(id_X, id_year_Y_outcome, by = 'id')

setup <- ltjmm(Y ~ year | .-Y-outcome-year-id | id | outcome, data = dd)
fit <- stan(file = file.path(.libPaths()[1], "ltjmm", "stan", "ltjmm.stan"),
            seed = rng_seed,
            data = setup$data,
            pars = c('beta', 'delta', 'alpha0', 'alpha1', 'gamma',
                     'sigma_alpha0', 'sigma_alpha1', "sigma_delta", 'sigma_y', 'log_lik'),
            open_progress = FALSE, chains = 2, iter = 2000,
            warmup = 1000, thin = 1, cores = 2)





