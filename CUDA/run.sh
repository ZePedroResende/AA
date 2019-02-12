#!/bin/bash
source /share/apps/intel/parallel_studio_xe_2019/compilers_and_libraries_2019/linux/bin/compilervars.sh intel64
module load papi/5.4.1
module load gcc/4.8.2
module load cuda/7.0.28
export CUDA=yes
cd $HOME/AA/CUDA/
make clean
make
./bin/simple comCUDA 32 time
python testing.py 
