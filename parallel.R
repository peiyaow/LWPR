nsim = 50

# myseeds = floor(1e4 * runif(nsim))
# save(myseeds, file = "myseed1.RData")
load("/nas/longleaf/home/peiyao/LWPR/myseed1.RData")

# for (i in 1:nsim){
#    for (a in c(0, 0.5, 1)){
#       for (t in 1:3){
#         system(paste0('bsub -q week -o SlRfnoDb.txt -M 20 Rscript test0613.R alpha0=', 0, ' alpha=', a, ' myseed=', myseeds[i], ' t=', t))
#       }
#    }
# }

a = 0
t = 1
for (i in 1:nsim){
#  for (a in c(0, 0.5, 1)){
#    for (t in 1:3){
      system(paste0('sbatch -o test.out -t 10:00:00 -n 1 --wrap="Rscript test0613.R alpha0=', 0, ' alpha=', a, ' myseed=', myseeds[i], ' t=', t, '"'))
#    }
#  }
}