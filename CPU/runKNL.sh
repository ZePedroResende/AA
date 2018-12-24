#!/bin/bash
export PATH=/share/apps/gcc/5.3.0/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/gcc/5.3.0/lib:$LD_LIBRARY_PATH
source /share/apps/intel/parallel_studio_xe_2019/compilers_and_libraries_2019/linux/bin/compilervars.sh intel64
#source /opt/intel/compilers and libraries/linux/bin/compilervars.sh intel64
#module load papi/5.4.1
export PATH=/share/apps/papi/5.4.1/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/papi/5.4.1/lib:$LD_LIBRARY_PATH
cd $HOME/AATRABALHO/AA/CPU/
make clean
make
export OMP_NUM_THREADS=128
python testingKNL.py 
