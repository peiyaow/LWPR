#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH -o test.out

Rscript parallel.R
