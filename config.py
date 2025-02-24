import os
import time
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Self-supervised Training.')
    parser.add_argument('--model_name', type=str, default='BYOL', help='The name of models', choices=['Supervised', 'BYOL', 'SimSiam'],
                        # required=True
                        )
    parser.add_argument('--dataset', type=str, default='CIFAR-10', choices=['CIFAR-10', 'STL-10', 'CIFAR-100', 'Tiny-Imagenet-200'],
                        # required=True
                        )
    parser.add_argument('--random_seed', type=int, default=2025, metavar='S', help='Random seed')
    parser.add_argument('--max_epoch', type=int, default=40, help="training epochs")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers')
    parser.add_argument('--stage', type=int, default=0, help="0 train model; 1 linear evaluate model")

    parser.add_argument('--lr', default=0.2, type=float, metavar='LR', help='base learning rate for weights')
    parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR', help='base learning rate for weights')
    parser.add_argument('--learning-rate-biases', default=0.005, type=float, metavar='LR', help='base learning rate for biases and batch norm parameters')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='SGD momentum')
    parser.add_argument('--weight-decay', default=4e-4, type=float, metavar='M', help='SGD weight decay')

    parser.add_argument('--num_classes', type=int, default=10, help="the categories of different labels")
    parser.add_argument('--hidden_dim', type=int, default=2048, help="dimension of contrastive representation")
    parser.add_argument('--proj_dim', type=int, default=256, help="dimension of contrastive representation")
    parser.add_argument('--predictor_dim', type=int, default=256, help="dimension of predictor representation")

    parser.add_argument('--linear_epochs', type=int, default=100, help="training epochs")
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--ema_momentum', default=0.996, type=float, help='momentum of updating key encoder (default: 0.996)')

    parser.add_argument('--data_root_dir', default="/Users/xianweicao/Documents/workspace/Dataset", type=str, help='the directory of train or test dataset')
    parser.add_argument('--saver_dir', default="./checkpoints", type=str, help='the directory of train or test dataset')
    parser.add_argument('--save_freq', default=10, type=int, help='save checkpoint every x epochs')
    args = parser.parse_args()

    args.lr = 0.2 * args.batch_size / 256

    args.save_dir_prefix = f'{args.model_name}/{args.dataset}/{args.random_seed}_{args.max_epoch}_{args.batch_size}_{args.hidden_dim}_{args.proj_dim}_{args.predictor_dim}/{time.strftime("%H%M%S")}'
    args.save_dir = os.path.join(args.saver_dir, args.save_dir_prefix)
    print("save_dir:", args.save_dir)

    args.data_dir = os.path.join(args.data_root_dir, args.dataset)
    print("data dir:", args.data_dir)

    if args.dataset == "CIFAR-10":
        args.num_classes = 10
        args.H, args.W = 32, 32
    elif args.dataset == "CIFAR-100":
        args.num_classes = 100
        args.H, args.W = 32, 32
    elif args.dataset == "STL-10":
        args.num_classes = 10
        args.H, args.W = 96, 96
    elif args.dataset == "Tiny-Imagenet-200":
        args.num_classes = 200
        args.H, args.W = 64, 64

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args
