#!/bin/sh
qsub  -lnodes=1:r662:k20:ppn=48,walltime=10:00:00 ./run.sh 
