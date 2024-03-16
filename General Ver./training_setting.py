import os
import nni
import argparse
coco_parser = argparse.ArgumentParser()
coco_parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
coco_parser.add_argument('--cuda_num', type=str, default='3', help='Which GPU')
coco_parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
coco_parser.add_argument('--testmode', type=str, default='test/', help='Export file fold choose')
coco_parser.add_argument('--seed', type=int, default=42, help='Random seed.')
coco_parser.add_argument('--resume_last', action='store_true', default=False)
coco_parser.add_argument('--resume_best', action='store_true', default=False)

coco_parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
coco_parser.add_argument('--patience', type=int, default=150, help='Patience')
coco_parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')  # 0.01
coco_parser.add_argument('--lr_step', type=float, default=1, help='Initial learning rate.')  # 0.99999
coco_parser.add_argument('--lr_lb', type=float, default=0.001, help='Initial learning rate.')
coco_parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')# 5e-4
coco_parser.add_argument('--warmup', type=int, default=1, help='Learning rate warmup')
coco_parser.add_argument('--nbatch', type=int, default=1, help='Number of self defined batch')
coco_parser.add_argument('--dropout', type=float, default=0.0, help='Dropout Probability')

coco_parser.add_argument('--fold_idx', type=int, default=1, help='fold idx for multi-split')
coco_parser.add_argument('--dataset', type=str, default='CHAMELEON', help='Type of dataset')
coco_parser.add_argument('--sampler_type', type=str, default="cluster", help='random, graphsaint')

coco_parser.add_argument('--model_type', type=str, default="feat", help='stage choice: encoder, full')
coco_parser.add_argument('--self_loop', type=int, default=1, help='self loop for normalized adj')

coco_parser.add_argument('--d_model', type=int, default=64, help='Number of hidden units.')
coco_parser.add_argument('--nblock', type=int, default=0, help='Number of coco block in network')
coco_parser.add_argument('--nlayer', type=int, default=4, help='Number of conv layers in network')
coco_parser.add_argument('--nTlayer', type=int, default=0, help='Number of layers in network')
coco_parser.add_argument('--nh', type=int, default=8, help='dimension for perm generation feature')
coco_parser.add_argument('--nk', type=int, default=8, help='dimension for perm generation feature')
coco_parser.add_argument('--filter_size', type=int, default=5, help='Conv filter size')
coco_parser.add_argument('--stride', type=int, default=5, help='Conv stride')
coco_parser.add_argument('--temp', type=float, default=10, help='dimension for perm generation feature')
coco_parser.add_argument('--sparse', action='store_true', default=False)
coco_parser.add_argument('--wo_res', action='store_true', default=False)
coco_parser.add_argument('--app_adj', action='store_true', default=False)

# for model test, will be removed in the final release
coco_parser.add_argument('--base_size', type=int, default=1000, help='dimension for perm generation feature')
args = coco_parser.parse_args()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num