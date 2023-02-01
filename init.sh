export PYTHONPATH=`pwd`
source venv/bin/activate
module load gcc/10.2.0-gcc-4.8.5
module load cuda/11.2.1-gcc-10.2.0
#Xvfb :5 -screen 0 800x600x24 &
#export DISPLAY=:5