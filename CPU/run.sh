#!/bin/bash
module load gcc/5.3.0
source /share/apps/intel/parallel_studio_xe_2019/compilers_and_libraries_2019/linux/bin/compilervars.sh intel64
module load papi/5.4.1
cd $HOME/AATRABALHO/AA/CPU/
make clean
make
export OMP_NUM_THREADS=24
python testingOMP.py 
