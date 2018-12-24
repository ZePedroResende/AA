#!/bin/sh
qsub -qmei -lnodes=1:r662:ppn=48,walltime=30:00 ./run.sh 
