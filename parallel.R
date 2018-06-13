nsim = 50

# myseeds = floor(1e4 * runif(nsim))
# save(myseeds, file = "myseed1.RData")
# load("/netscr/peiyao/np/myseed1.RData")
load("/netscr/jialu/myseed1.RData")

for (i in 1:nsim){
   for (a in c(0, 0.5, 1)){
      for (t in 1:3){
        system(paste0('bsub -q week -o SlRfnoDb.txt -M 20 Rscript main_SlRfnoDb.R alpha0=', 0, ' alpha=', a, ' myseed=', myseeds[i], ' t=', t))
      }
   }
}

# 
# 
# for (i in 1:nsim){
#   system(paste0('bsub -q week -o SlRfnoDb.txt -M 20 Rscript main_SlRfnoDb.R myseed=', myseeds[i]))
# }