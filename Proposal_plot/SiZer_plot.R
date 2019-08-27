library(SiZer)
library(readr)
# library(glmnet)

t = 1
# longleaf
X0 = as.matrix(read_table("~/Documents/GitHub/LWPR/data/X1a.txt", col_names = F))
Y0 = as.matrix(read_table("~/Documents/GitHub/LWPR/data/Y5T.txt", col_names = F))[, 4+5*(t-1)]
label0 = as.ordered(read_table("~/Documents/GitHub/LWPR/data/label1.txt", col_names = F)$X1)

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

# only select MRI
X0 = X0[, 1:93]

# all NC and MCI and AD
X = X0
Y = Y0
label = label0

n = dim(X)[1]
p = dim(X)[2]

X.mean = apply(X, 2, mean)
X.sd = apply(X, 2, sd)
# X.sd = sapply(X.sd, function(x) ifelse(x<1e-5, 1, x)) 
X = t(apply(X, 1, function(x) (x - X.mean)/X.sd))

PCA.res = eigen(t(X)%*%X/n)
# PCA.res$values
# dim(PCA.res$vectors)
# t(X)%*%X%*%PCA.res$vectors/n-PCA.res$vectors%*%diag(PCA.res$values)

PC1 = X%*%PCA.res$vectors[,1]
# PC2 = X%*%PCA.res$vectors[,2]
# PC3 = X%*%PCA.res$vectors[,3]
# PC4 = X%*%PCA.res$vectors[,4]

res.SiZer = SiZer(PC1, log(Y+1), x.grid=seq(-10, 10, by = 0.5), h = c(1,15), degree = 1)
res.LWP = locally.weighted.polynomial(PC1, log(Y+1), x.grid=seq(-10, 10, by = 0.5), degree = 1, kernel.type = "Normal", h = 10^0.27)

par(mfcol = c(2,1), cex = 0.7)
plot(res.LWP, alpha = 0.05, use.ess=F, draw.points = F, xlim = c(-10,10), ylim = c(1.5,3.5), ylab = "Clinical Score", xlab = "PC1", main = "Local Polynomial Fit")
plot(res.SiZer, xlab = "PC1", xlim = c(-10, 10), main = "SiZer Plot")
abline(h = 0.27, col = "white")
# plot(PC1, log(Y+1), type = "p")

