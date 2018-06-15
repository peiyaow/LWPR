#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH -o parallel_main.out

Rscript parallel_main.R
