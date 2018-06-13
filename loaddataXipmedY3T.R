# load data 
# X1a.txt; Y5T.txt; label1.txt
# X features imputed by group median

library(readr)
# t = 1:3 given in parallel.R; drop time point 4 and 5 since there are so few AD patients 
# t = 3 
# 4+5*(t-1)
X0 = as.matrix(read_table("/netscr/peiyao/np/X1a.txt", col_names = F))
Y0 = as.matrix(read_table("/netscr/peiyao/np/Y5T.txt", col_names = F))[, 4+5*(t-1)]
label0 = as.ordered(read_table("/netscr/peiyao/np/label1.txt", col_names = F)$X1)

# setwd("/Users/MonicaW/Documents/R/np")
X0 = as.matrix(read_table("~/Documents/R/np/X1a.txt", col_names = F))
Y0 = as.matrix(read_table("~/Documents/R/np/Y5T.txt", col_names = F))[, 4+5*(t-1)]
label0 = as.ordered(read_table("~/Documents/R/np/label1.txt", col_names = F)$X1)

id.cut1 = !is.na(Y0)
X0 = X0[id.cut1, ]
Y0 = Y0[id.cut1]
label0 = label0[id.cut1]

id.cut2 = (Y0 > 0)
X0 = X0[id.cut2,]
Y0 = Y0[id.cut2]
label0 = label0[id.cut2]

# MRI  PET          MRI+PET   MRI+PET+CSF 
# 1:93 94:(94+93-1) 1:186     all 
# range.list = list()
# feature.name = c("MRI", "PET", "MRIPET", "MRIPETCSF")
# range.list[[feature.name[1]]] = 1:93
# range.list[[feature.name[2]]] = 94:186
# range.list[[feature.name[3]]] = 1:186
# range.list[[feature.name[4]]] = 1:189

# only use MRI and PET
Xs = X0[, 1:186]

# within group median , each element from the list is the median within the group 
Xs.med.list = lapply(levels(label0), function(x) apply(Xs[label0 == x, ], 2, function(y) median(y, na.rm = T)))

# impute with median
Xs.missing = is.na(Xs)
for (i in 1:nrow(Xs)){
  ip.med = Xs.med.list[[as.numeric(label0[i])]]
  Xs[i, Xs.missing[i, ]] = ip.med[Xs.missing[i, ]]
}

X = Xs
Y = Y0
label = label0






