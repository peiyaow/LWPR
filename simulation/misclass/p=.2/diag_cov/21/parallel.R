nsim = 50

load("/nas/longleaf/home/peiyao/LWPR/simulation/sim.myseed.RData")

for (i in 1:nsim){
  for (a in c(0, 0.5, 1)){
#    print(paste0('sbatch -o sim.out -t 10:00:00 -n 1 --mem=4g --wrap="Rscript sim_bothnoDb.R alpha=', a, ' myseed=', myseeds[i], '"'))
    system(paste0('sbatch -o sim.out -t 10:00:00 -n 1 --mem=4g --wrap="Rscript sim_bothnoDb.R alpha=', a, ' myseed=', myseeds[i], '"'))  
  }
}
