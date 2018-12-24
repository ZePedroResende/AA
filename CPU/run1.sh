#!/bin/bash
module load gcc/5.3.0
source /share/apps/intel/parallel_studio_xe_2019/compilers_and_libraries_2019/linux/bin/compilervars.sh intel64
cd $HOME/aa/knc/lab3/
make clean
make
export OMP_NUM_THREADS=32
./helloflops3o_xeon

export OMP_NUM_THREADS=64
./helloflops3o_xeon
