# load data 
# X1a.txt; Y5T.txt; label1.txt
# X features imputed by group median

library(readr)
# t = 1:3 given in parallel.R; drop time point 4 and 5 since there are so few AD patients 

# longleaf
X0 = as.matrix(read_table("/nas/longleaf/home/peiyao/LWPR/data/X1a.txt", col_names = F))
Y0 = as.matrix(read_table("/nas/longleaf/home/peiyao/LWPR/data/Y5T.txt", col_names = F))[, 4+5*(t-1)]
label0 = as.ordered(read_table("/nas/longleaf/home/peiyao/LWPR/data/label1.txt", col_names = F)$X1)

# # mac: /Users/MonicaW/Documents/GitHub/LWPR
# X0 = as.matrix(read_table("/Users/MonicaW/Documents/GitHub/LWPR/data/X1a.txt", col_names = F))
# Y0 = as.matrix(read_table("/Users/MonicaW/Documents/GitHub/LWPR/data/Y5T.txt", col_names = F))[, 4+5*(t-1)]
# label0 = as.ordered(read_table("/Users/MonicaW/Documents/GitHub/LWPR/data/label1.txt", col_names = F)$X1)

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



