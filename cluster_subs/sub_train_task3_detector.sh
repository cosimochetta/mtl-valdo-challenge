#$ -l gpu=true  
#$ -l h_rt=10:0:0                                                                                                       
#$ -l tmem=20.5G                                                                                                        
#$ -S /bin/bash                                                                                                         
#$ -j y                                                                                                                 
#$ -wd /cluster/project0/MAS/cosimo/mtl-valdo-challenge/train                                                                                
#$ -o /cluster/project0/MAS/cosimo/outputs                                                                              
#$ -e /cluster/project0/MAS/cosimo/outputs   
                                                                                                                                                                                                   
source activate /cluster/project0/MAS/cosimo/env                                                                        
source /share/apps/examples/source_files/cuda/cuda-10.0.source   
                                                                                                                                                                               
/cluster/project0/MAS/cosimo/env/bin/python3.6 -u train_task3_detector.py \
    --directory="/cluster/project0/MAS/cosimo/valdo_patches/Task3" \
    --batch_size=128 \
    --epochs=20 \
    --lr=0.00001 \
    --optimizer="Adam" \
    --optim_args='{"weight_decay": 0.0001}' \
    --scheduler="None" \
    --scheduler_args='{}' \
    --log_folder="./runs/Task3/detector" \
    --run_model="resnet" \
    --run_info="" \
    --patch_size=32
