import os
import argparse
cocn_parser = argparse.ArgumentParser()
cocn_parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
cocn_parser.add_argument('--cuda_num', type=str, default='3', help='Which GPU')
cocn_parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
cocn_parser.add_argument('--testmode', type=str, default='test/', help='Export file fold choose')
cocn_parser.add_argument('--seed', type=int, default=42, help='Random seed.')
cocn_parser.add_argument('--resume_last', action='store_true', default=False)
cocn_parser.add_argument('--resume_best', action='store_true', default=False)

cocn_parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
cocn_parser.add_argument('--patience', type=int, default=150, help='Patience')
cocn_parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
cocn_parser.add_argument('--lr_step', type=float, default=1, help='Learning rate dropping rate')  # 0.99999
cocn_parser.add_argument('--lr_lb', type=float, default=0.001, help='Learning rate lowerbound.')
cocn_parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
cocn_parser.add_argument('--warmup', type=int, default=1, help='Learning rate warmup')
cocn_parser.add_argument('--nbatch', type=int, default=1, help='Number of self defined batch')
cocn_parser.add_argument('--dropout', type=float, default=0.5, help='Dropout Probability')
cocn_parser.add_argument('--fold_idx', type=int, default=1, help='Fold idx for multi split dataset')

cocn_parser.add_argument('--d_model', type=int, default=64, help='Hidden size')
cocn_parser.add_argument('--dataset', type=str, default='CORNELL', help='Type of dataset: COLLAB, IMDB-BINARY, IMDB-MULTI, MUTAG, PROTEINS, NCI1, CORNELL, TEXAS, WISCONSIN, SQUIRREL, CHAMELEON, ACTOR')
cocn_parser.add_argument('--nblock', type=int, default=0, help='Number of cocn pooling layers')
cocn_parser.add_argument('--nlayer', type=int, default=1, help='Number of cocn conv layers')
cocn_parser.add_argument('--nTlayer', type=int, default=0, help='Number of layers in permutation module')
cocn_parser.add_argument('--nh', type=int, default=8, help='Number of permutations')
cocn_parser.add_argument('--temp', type=float, default=0.1, help='Relaxation factor for permutation')
cocn_parser.add_argument('--filter_size', type=int, default=5, help='Conv filter size')
cocn_parser.add_argument('--stride', type=int, default=5, help='Conv stride')
args = cocn_parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num