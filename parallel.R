nsim = 50
# myseeds = floor(1e4 * runif(nsim))
# save(myseeds, file = "myseed1.RData")

load("/nas/longleaf/home/peiyao/LWPR/myseed1.RData")
for (i in 1:nsim){
 for (a in c(0, 0.5, 1)){
   for (t in 1:3){
     system(paste0('sbatch -o main_scaleY_logistic.out -t 10:00:00 -n 4 --mem-per-cpu=4g -N 1-1 --wrap="Rscript main_SlRfnoDb_scaleY_logistic.R alpha0=', 0, ' alpha=', a, ' myseed=', myseeds[i], ' t=', t, '"'))
   }
 }
}

# a=1
# t=2
# for (i in 1:nsim){
#   system(paste0('sbatch -o main_scaleY2.out -t 10:00:00 -n 4 --mem-per-cpu=4g -N 1-1 --wrap="Rscript main_SlRfnoDb_scaleY3.R alpha0=', 0, ' alpha=', a, ' myseed=', myseeds[i], ' t=', t, '"'))
# }


