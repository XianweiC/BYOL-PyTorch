# default setting is BYOL with stl10 dataset. see config.py for details.
python train.py

# supervised baseline
#python train.py --model_name='Supervised' --dataset='STL-10' --max_epoch=1000