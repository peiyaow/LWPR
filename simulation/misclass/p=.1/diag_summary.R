library(readr)
#setwd("C:/Users/peiyao/Dropbox/LWPR_result/simulation/misclass/p=.1")
setwd("C:/Users/peiyao/Documents/GitHub/LWPR/simulation/misclass/p=.1")
setwd("./diag_cov")
misclass.diag.list = list()
for (i in c(1,2,3)){
  setwd(paste0("./2", i))
  myresult.list = list()
  for (alpha in c(0,0.5,1)){
    result.mtx = read.csv(paste0("mae+corr+sim+alpha0=0+alpha=", alpha, ".csv"), header = T)
    mymean = apply(result.mtx, 2, mean)
    mysd = apply(result.mtx, 2, sd)
    myresult.list[[paste0("alpha=", alpha)]] = matrix(rbind(mymean[c(1,2,3,4)], mysd[c(1,2,3,4)]), ncol = 8)
  }
  setwd("..")
  misclass.diag.list[[i]] = do.call(rbind, myresult.list)
}
result.matrix.misclass = round(do.call(rbind, misclass.diag.list),3)[,c(3,4,7,8)]

# setwd("C:/Users/peiyao/Dropbox/LWPR_result/simulation/trueclass/")
setwd("C:/Users/peiyao/Documents/GitHub/LWPR/simulation/trueclass/")
setwd("./diag_cov")
trueclass.diag.list = list()
for (i in c(1,2,3)){
  setwd(paste0("./2", i))
  myresult.list = list()
  for (alpha in c(0,0.5,1)){
    result.mtx = read.csv(paste0("mae+corr+sim+alpha0=0+alpha=", alpha, ".csv"), header = T)
    mymean = apply(result.mtx, 2, mean)
    mysd = apply(result.mtx, 2, sd)
    myresult.list[[paste0("alpha=", alpha)]] = matrix(rbind(mymean[c(1,2,3,4)], mysd[c(1,2,3,4)]), ncol = 8)
  }
  setwd("..")
  trueclass.diag.list[[i]] = do.call(rbind, myresult.list)
}
result.matrix.trueclass = round(do.call(rbind, trueclass.diag.list),3)[,c(3,4,7,8)]


#setwd("C:/Users/peiyao/Dropbox/LWPR_result/simulation/comp_result/")
setwd("C:/Users/peiyao/Documents/GitHub/LWPR/simulation/comp_result/")
setwd("./diag_cov")
comp.diag.list =list()
for (i in c(1,2,3)){
  result.mtx = read.csv(paste0("simulation_results2", i, ".csv"), header = T)
  mymean = apply(result.mtx, 2, function(x) mean(x, na.rm = T))
  mysd = apply(result.mtx, 2, function(x) sd(x, na.rm = T))
  comp.diag.list[[i]] = cbind(t(rbind(mymean, mysd)[,1:4]), t(rbind(mymean, mysd)[,5:8]))

}
result.matrix.comp = round(do.call(rbind,comp.diag.list),3)

result.matrix.raw = rbind(result.matrix.comp, result.matrix.trueclass, result.matrix.misclass)

result.list = list()
for (t in 1:3){
  temp.matrix = result.matrix.raw[c(seq(1,4),seq(13,15),seq(22,24))+c(rep(1,4)*4, rep(1,6)*3)*(t-1),]
  temp.matrix = temp.matrix[c(1,3,4,2,5,6,7,8,9,10),]
  row.names(temp.matrix) = c("rf","ridge", "elast",  "lasso", "true ridge", "true elast", "true lasso", "mis ridge", "mis elast", "mis lasso")
  #row.names(temp.matrix) = c("rf", "lasso", "ridge", "elast", "true ridge", "true elast", "true lasso", "mis ridge", "mis elast", "mis lasso")
  colnames(temp.matrix) = c("mae.mean", "mae.sd", "corr.mean", "corr.sd")
  result.list[[t]] = temp.matrix
}

#setwd("C:/Users/peiyao/Dropbox/LWPR_result/simulation/misclass/p=.1")
setwd("C:/Users/peiyao/Documents/GitHub/LWPR/simulation/misclass/p=.1")
write.table(do.call(rbind, result.list), file = "diag_p=.1.csv", sep = ',', col.names = T, row.names = T)



