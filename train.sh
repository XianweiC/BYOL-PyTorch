# default setting is BYOL with stl10 dataset. see config.py for details.
python SSL_train_byol.py --model_name='BYOL' --dataset='STL-10' --max_epoch=40
python linear_probing.py --checkpoint='checkpoints/BYOL/STL-10/xxxx/model-final.pth'

# supervised baseline
#python train.py --model_name='Supervised' --dataset='STL-10' --max_epoch=1000