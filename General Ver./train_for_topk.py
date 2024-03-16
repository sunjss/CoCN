from training_setting import args
from utils.utils import CoCNDataLoader
from model.modules import TopKPermGenModule
from model.modules import FeatureBasedPositionGenModule, DistanceBasedPositionGenModule
from model.modules import TopKCoCN, ResCoCNModuleG

import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch
lr_ranking = args.lr * 1
print("running on GPU" + str(args.cuda_num))
print("Tlayer:" + str(args.nTlayer) + "; layer:" + str(args.nlayer) + "; block:" + str(args.nblock))
print("heads:"+str(args.nh) + "; hidden:" + str(args.d_model) + "; filter size:" + str(args.filter_size))
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dataset_list = ['squirrel-directed', 'squirrel-filtered-directed', "squirrel-filtered", "squirrel", 
                'chameleon-directed', 'chameleon-filtered-directed', "chameleon", "chameleon-filtered",
                "AmazonComputers", "AmazonPhoto", "CoauthorCS", "CoauthorPhysics",
                'amazon-ratings', 'minesweeper', 'tolokers', 'questions', "genius",
                "ACTOR", "SQUIRREL", "CHAMELEON", "CORNELL", "TEXAS", "WISCONSIN"]
def lambda_lr(s):
    s += 1
    if s < args.warmup * args.nbatch:
        return float(s) / float(args.warmup * args.nbatch)
    return max(args.lr_lb, args.lr_step ** (s - args.warmup * args.nbatch))


def model_init(dataset):
    global nfeats
    global nclass
    ranker_dict = {"dist": DistanceBasedPositionGenModule,
                   "feat": FeatureBasedPositionGenModule}
    ranking_model = ranker_dict[args.model_type](N_enc=args.nTlayer,
                                                 d_in=ndists if args.model_type == "dist" else nfeats, 
                                                 d_model=args.d_model, 
                                                 h=args.nh,
                                                 dropout=args.dropout)
    perm = TopKPermGenModule
    CoCNMoudle = ResCoCNModuleG
    CoCN = TopKCoCN
    perm_gen_model = perm(d_model=args.d_model, temp=args.temp, k=args.nk)
    cocn = CoCNMoudle(h=args.nh,
                      d_in=nfeats,
                      d_ein=nedgefeats,
                      d_model=args.d_model,
                      nclass=nclass,
                      filter_size=args.filter_size,
                      stride=args.stride,
                      nlayers=args.nlayer,
                      nblocks=args.nblock,
                      app_adj=args.app_adj,
                      dropout=args.dropout)
    model = CoCN(ranking_model, perm_gen_model, cocn, task_type=task_type, self_loop=args.self_loop, app_adj=args.app_adj)
    cocn_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if task_type in ["multi-class", "binary-class"] or "link" in task_type:
        loss_func = nn.BCEWithLogitsLoss()
    else:
        loss_func = nn.NLLLoss()
    if args.cuda:
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model, cocn_optimizer, loss_func


def model_train(model, optimizer, scheduler):
    model.train()
    torch.autograd.set_detect_anomaly(True)
    running_loss = .0
    acc = .0
    out_ls = []
    label_ls = []
    div = 0
    iter_cnt_div = 0
    for it, data in enumerate(train_data):
        node_num = 1 if args.base_size == -1 else data.x.shape[-2]
        # idx = torch.randperm(node_num)
        # if args.cuda:
        #     idx = idx.cuda()
        # node_idx_batch = idx[:args.base_size]
        # j = node_idx_batch.numel()
        # i = 1
        # iter_range = 1      
        iter_range = node_num
        base_size = node_num if args.base_size == -1 else args.base_size
        for i in range(0, node_num, base_size):
            j = node_num - i if node_num - i < base_size else base_size
            node_idx_batch = torch.arange(i, i + j, device=next(model.parameters()).device)
            optimizer.zero_grad()
            if isinstance(data, list):
                labels = [d.y for d in data]
                labels = torch.cat(labels)
                data = Batch.from_data_list(data)
                if args.cuda:
                    labels = labels.cuda()
                    data = data.cuda()
            else:
                if args.cuda:
                    data = data.cuda()
                # labels = data.y.flatten()[node_idx_batch]
                labels = data.y.flatten()[i:i+j]
            if args.dataset in dataset_list:
                if args.dataset in ["SQUIRREL", "CHAMELEON", "CORNELL", "TEXAS", "WISCONSIN", "ACTOR"]:
                    # train_mask = data.train_mask[node_idx_batch, args.fold_idx]
                    train_mask = data.train_mask[i:i+j, args.fold_idx]
                else:
                    train_mask = data.mask * data.train_mask if labels.min() == -1 else data.train_mask
                    # train_mask = train_mask[node_idx_batch]
                    train_mask = train_mask[i:i+j]
                if train_mask.sum() == 0:
                    continue
            div += j
            iter_cnt_div += 1
            # output = model(data, node_idx_batch, j)
            output = model(data, i, j)
            if args.dataset in dataset_list:
                if args.dataset in ['genius', 'minesweeper', 'tolokers', 'questions']:
                    output = output.squeeze(-1)
                    # labels = labels.type(torch.float32)
                    loss_train = loss_func(output[train_mask], labels[train_mask])
                    acc_train = 0
                    out_ls.append(output[train_mask, 1])
                    label_ls.append(labels[train_mask])
                else:
                    loss_train = loss_func(output[train_mask], labels[train_mask])
                    acc_train = metric(output[train_mask], labels[train_mask]).item() * j
            else:
                loss_train = loss_func(output, labels)
                acc_train = metric(output, labels).item() * j
            loss_train.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss_train.data.item()
            acc += acc_train
            tqdm_t.set_description('lr %.4e, %d/%d %.4e' %
                                (scheduler.get_last_lr()[0],
                                i, iter_range,
                                running_loss / iter_cnt_div) +\
                                val_des + sum_des)
            writer.add_scalar('data/train_loss', running_loss / iter_cnt_div, epoch * len(train_data) + it)
            writer.add_scalar('data/train_acc', acc / div, epoch * len(train_data) + it)
    if args.dataset in ['genius', 'minesweeper', 'tolokers', 'questions']:
        acc = div * metric(out_ls, label_ls)
    writer.add_scalar('data/epoch_train_loss', running_loss / iter_cnt_div, epoch)
    writer.add_scalar('data/epoch_train_acc', acc / div, epoch)
    return running_loss / iter_cnt_div, acc / div


def model_val(model):
    # Evaluate validation set performance separately,
    # deactivates dropout during validation run.
    model.eval()
    running_loss = .0
    acc = .0
    out_ls = []
    label_ls = []
    div = 0
    iter_cnt_div = 0
    with torch.no_grad():
        for it, data in enumerate(val_data):
            node_num = data.x.shape[-2]
            base_size = node_num if args.base_size == -1 else args.base_size
            # base_size = node_num if args.base_size == -1 else 20000 # args.base_size
            # iter_range = 3 # node_num
            # for i in range(iter_range):
            #     idx = torch.randperm(node_num)
            #     if args.cuda:
            #         idx = idx.cuda()
            #     node_idx_batch = idx[:args.base_size]
            #     j = node_idx_batch.numel()
            iter_range = node_num
            for i in range(0, node_num, base_size):
                j = node_num - i if node_num - i < base_size else base_size
                node_idx_batch = torch.arange(i, i + j, device=next(model.parameters()).device)
                if isinstance(data, list):
                    labels = [d.y for d in data]
                    labels = torch.cat(labels)
                    data = Batch.from_data_list(data)
                    if args.cuda:
                        labels = labels.cuda()
                        data = data.cuda()
                else:
                    if args.cuda:
                        data = data.cuda()
                    # labels = data.y.flatten()[node_idx_batch]
                    labels = data.y.flatten()[i:i+j]
                if args.dataset in dataset_list:
                    if args.dataset in ["SQUIRREL", "CHAMELEON", "CORNELL", "TEXAS", "WISCONSIN", "ACTOR"]:
                        # val_mask = data.val_mask[node_idx_batch, args.fold_idx]
                        val_mask = data.val_mask[i:i+j, args.fold_idx]
                    else:
                        val_mask = data.mask * data.val_mask if labels.min() == -1 else data.val_mask
                        val_mask = val_mask[i:i+j]
                        # val_mask = val_mask[node_idx_batch]
                    if val_mask.sum() == 0:
                        continue
                div += j
                iter_cnt_div += 1
                # output = model(data, node_idx_batch, j)
                output = model(data, i, j)
                if args.dataset in dataset_list:
                    if args.dataset in ['genius', 'minesweeper', 'tolokers', 'questions']:
                        output = output.squeeze(-1)
                        # labels = labels.type(torch.float32)
                        loss_val = loss_func(output[val_mask], labels[val_mask])
                        acc_val = 0
                        out_ls.append(output[val_mask, 1])
                        label_ls.append(labels[val_mask])
                    else:
                        loss_val = loss_func(output[val_mask], labels[val_mask])
                        acc_val = metric(output[val_mask], labels[val_mask]).item() * j
                else:
                    loss_val = loss_func(output, labels)
                    acc_val = metric(output, labels).item()
                if args.cuda:
                    torch.cuda.empty_cache()
                acc += acc_val
                running_loss += loss_val.data.item()
                tqdm_t.set_description('lr %.4e' % (scheduler.get_last_lr()[0]) + \
                                        train_des + ', val %d/%d %f' % 
                                        (i, iter_range, running_loss / iter_cnt_div)+ \
                                        sum_des)
            writer.add_scalar('data/val_loss', running_loss / iter_cnt_div, epoch * len(val_data) + it)
            writer.add_scalar('data/val_acc', acc / div, epoch * len(val_data) + it)
    writer.add_scalar('data/epoch_val_loss', running_loss / iter_cnt_div, epoch)
    writer.add_scalar('data/epoch_val_acc', acc / len(val_data), epoch)
    if args.dataset in ['genius', 'minesweeper', 'tolokers', 'questions']:
        acc = div * metric(out_ls, label_ls)
    return running_loss / iter_cnt_div, acc / div


def model_test(model):
    model.eval()
    running_loss = .0
    acc = .0
    out_ls = []
    label_ls = []
    div = 0
    iter_cnt_div = 0
    pbar = None
    with torch.no_grad():
        for it, data in enumerate(test_data):
            node_num = data.x.shape[-2]
            if pbar is None:
                pbar = tqdm(desc='Test', unit='it', total=node_num)
            base_size = node_num if args.base_size == -1 else args.base_size # 1000
            for i in range(0, node_num, base_size):
                j = node_num - i if node_num - i < base_size else base_size
                # node_idx_batch = torch.arange(i, i + j, device=next(model.parameters()).device)
                if isinstance(data, list):
                    labels = [d.y for d in data]
                    labels = torch.cat(labels)
                    data = Batch.from_data_list(data)
                    if args.cuda:
                        labels = labels.cuda()
                        data = data.cuda()
                else:
                    if args.cuda:
                        data = data.cuda()
                    # labels = data.y.flatten()[node_idx_batch]
                    labels = data.y.flatten()[i:i+j]
                if args.dataset in dataset_list:
                    if args.dataset in ["SQUIRREL", "CHAMELEON", "CORNELL", "TEXAS", "WISCONSIN", "ACTOR"]:
                        # test_mask = data.test_mask[node_idx_batch, args.fold_idx]
                        test_mask = data.test_mask[i:i+j, args.fold_idx]
                    else:
                        test_mask = data.mask * data.test_mask if labels.min() == -1 else data.test_mask
                        # test_mask = test_mask[node_idx_batch]
                        test_mask = test_mask[i:i+j]
                    if test_mask.sum() == 0:
                        continue
                div += j
                iter_cnt_div += 1
                # output = model(data, node_idx_batch, j)
                output = model(data, i, j)
                # output_hetero(perm, output, labels)
                if args.dataset in dataset_list:
                    if args.dataset in ['genius', 'minesweeper', 'tolokers', 'questions']:
                        output = output.squeeze(-1)
                        # labels = labels.type(torch.float32)
                        loss_test = loss_func(output[test_mask], labels[test_mask])
                        acc_test = 0
                        out_ls.append(output[test_mask, 1])
                        label_ls.append(labels[test_mask])
                    else:
                        loss_test = loss_func(output[test_mask], labels[test_mask])
                        acc_test = metric(output[test_mask], labels[test_mask]).item() * j
                else:
                    loss_test = loss_func(output, labels)
                    acc_test = metric(output, labels).item() * j
                if args.cuda:
                    torch.cuda.empty_cache()
                acc += acc_test
                running_loss += loss_test.data.item()
                pbar.update(j)
    if args.dataset in ['genius', 'minesweeper', 'tolokers', 'questions']:
        acc = div * metric(out_ls, label_ls)
    if type(acc) is dict:
        print("Test set results:",
                "loss= {:.4f}".format(running_loss / iter_cnt_div))
        acc_out = ""
        for k, v in acc.items():
            acc_out += f"{k}={v / div}, "
        acc = acc_out[:-2]
        print(acc)
    else:
        print("Test set results:",
                "loss= {:.4f}".format(running_loss / iter_cnt_div),
                "metric= {:.4f}".format(acc / div))
        acc = round(acc / div, 5)

    return round(running_loss / iter_cnt_div, 5), acc


if __name__ == '__main__':
    # Load data
    data_loader = CoCNDataLoader()
    print("Loading " + args.dataset.title() + "...")
    assert args.dataset in dataset_list, "Only node classification is supported for Segment CoCN"
    flag = (args.dataset in ['squirrel-directed', 'squirrel-filtered-directed', "squirrel-filtered", "squirrel", 
                             'chameleon-directed', 'chameleon-filtered-directed', "chameleon", "chameleon-filtered"])
    graph_iter_range = 10 if flag else 1
    for k in range(graph_iter_range):
        args.fold_idx = k if graph_iter_range > 1 else args.fold_idx
        data_loader.load_data(dataset=args.dataset, spilit_type="public", 
                            nbatch=args.nbatch, fold_idx=args.fold_idx)
        nclass, nfeats, nedgefeats = data_loader.nclass, data_loader.nfeats, data_loader.nedgefeats
        train_data, val_data, test_data = data_loader.train_data, data_loader.val_data, data_loader.test_data
        metric = data_loader.metric
        task_type = data_loader.task_type
        ndists = data_loader.ndists

        flag = (args.dataset in ["SQUIRREL", "CHAMELEON", "CORNELL", "TEXAS", "WISCONSIN", "ACTOR"])
        iter_range = 10 if flag else 1
        for i in range(iter_range):
            save_path = 'export/' + args.testmode + args.dataset + '/'
            model_config =  '%.2f_%d_%d_%d_%d_%d_%d_%d_%d_%d/' % (args.dropout, args.base_size, args.nk, args.nTlayer, args.nblock, args.nlayer, args.filter_size, args.stride, args.nh, args.d_model)
            args.fold_idx = i if iter_range > 1 else args.fold_idx
            if args.dataset in ['squirrel-directed', 'squirrel-filtered-directed', "squirrel-filtered", "squirrel", 
                                'chameleon-directed', 'chameleon-filtered-directed', "chameleon", "chameleon-filtered", 
                                "NCI1", "IMDB-BINARY", "IMDB-MULTI", "MUTAG", "PROTEINS", "ENZYMES", "D&D", "COLLAB", "REDDIT-BINARY", "REDDIT-MULTI-12K", 
                                "SQUIRREL", "CHAMELEON", "CORNELL", "TEXAS", "WISCONSIN", "ACTOR"]:
                model_config = model_config + '%d/' % (args.fold_idx)
                print("Running on split " + str(args.fold_idx))
            save_path = save_path + model_config
            is_exists = os.path.exists(save_path)
            if not is_exists:
                save_path = save_path + "1/"
                os.makedirs(save_path)
            else:
                files = os.listdir(save_path)
                if args.resume_last or args.resume_best:
                    file_idx = str(len(files))
                    save_path = save_path + file_idx + "/"
                else:
                    file_idx = str(len(files) + 1)
                    save_path = save_path + file_idx + "/"
                    os.makedirs(save_path)
            model, optimizer, loss_func = model_init(args.dataset)
            name = "CoCN"
            logfile = "compress_convolution"
            scheduler = lr_scheduler.LambdaLR(optimizer, lambda_lr)
            continue_flag = True
            bad_counter = 0
            start_epoch = 0
            best = args.epochs + 1
            writer = SummaryWriter(log_dir=os.path.join(save_path, name))
            if args.resume_last or args.resume_best:
                if args.resume_last:
                    fname = save_path + logfile + '_last.pth'
                else:
                    fname = save_path + logfile + '_best.pth'

                if os.path.exists(fname):
                    data = torch.load(fname)
                    torch.set_rng_state(data['torch_rng_state'])
                    torch.cuda.set_rng_state(data['cuda_rng_state'])
                    np.random.set_state(data['numpy_rng_state'])
                    random.setstate(data['random_rng_state'])
                    model.load_state_dict(data['state_dict'], strict=False)
                    optimizer.load_state_dict(data['cocn_optimizer'])
                    scheduler.load_state_dict(data['cocn_scheduler'])
                    start_epoch = data['epoch'] + 1
                    bad_counter = data['patience']
                    best = data['best_val_loss']
                    # snd_flag = data['snd_flag']
                    # train_cnt = data['train_cnt']
                    print('Resuming from epoch %d, best validation loss %f' % (
                        data['epoch'], data['best_val_loss']))
            val_des = ', val NaN'
            sum_des = ', bad 0, best NaN'
            l_bad_cnt = 0
            print("Start training...")
            with tqdm(unit='it', total=args.epochs) as tqdm_t:
                for epoch in range(start_epoch, start_epoch + args.epochs):
                    if not continue_flag:
                        break
                    loss_train_value, acc_train_value = model_train(model, optimizer, scheduler)
                    train_des = ', %d/%d %.4e' % (len(train_data), len(train_data), loss_train_value)
                    torch.save({
                        'torch_rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state(),
                        'numpy_rng_state': np.random.get_state(),
                        'random_rng_state': random.getstate(),
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'cocn_optimizer': optimizer.state_dict(),
                        'cocn_scheduler': scheduler.state_dict(),
                        'patience': bad_counter,
                        'best_val_loss': best,
                    }, save_path + logfile + "_last.pth")
                    if not args.fastmode:
                        loss_val_value, acc_val_value = model_val(model)
                        if loss_val_value < best:
                            best = loss_val_value
                            torch.save({
                                'torch_rng_state': torch.get_rng_state(),
                                'cuda_rng_state': torch.cuda.get_rng_state(),
                                'numpy_rng_state': np.random.get_state(),
                                'random_rng_state': random.getstate(),
                                'epoch': epoch,
                                'state_dict': model.state_dict(),
                                'cocn_optimizer': optimizer.state_dict(),
                                'cocn_scheduler': scheduler.state_dict(),
                                'patience': bad_counter,
                                'best_val_loss': best,
                            }, save_path + logfile + "_best.pth")
                            if bad_counter > l_bad_cnt:
                                l_bad_cnt = bad_counter
                            bad_counter = 0
                        else:
                            bad_counter += 1
                        val_des = ', val %d/%d %f' % (len(val_data), len(val_data), loss_val_value)
                        sum_des = ', bad %i, best %.4f' % (bad_counter, best)
                        tqdm_t.set_description('lr %.4e' % (scheduler.get_last_lr()[0]) + \
                                                train_des + val_des + sum_des)
                        continue_flag = (bad_counter <= args.patience) or (l_bad_cnt == 0)
                    tqdm_t.update(1)
            if not args.fastmode:
                data = torch.load(save_path + logfile + "_best.pth")
                model.load_state_dict(data['state_dict'])
                print("Testing model..." + str(l_bad_cnt))
                test_loss, test_acc = model_test(model)
                

            print('Exporting data......')
            write_text = "-------training arg-------" + '\n'
            for k, v in vars(args).items():
                if k == "fold_idx":
                    write_text += f"-------dataset arg-------\ntask_type = {task_type}\n"
                elif k == "model_type":
                    write_text += "-------model arg-------\n"
                write_text += f'{k} = {v}\n'
            write_text = write_text + "-------results-------\n"
            write_text = write_text + 'loss = ' + str(test_loss) + '\n' + 'acc = ' + str(test_acc) + '\n'
            write_text = write_text + 'best val loss = ' + str(best) + '\n'
            fname = '/' + args.model_type + '_' + args.dataset + '_result.txt'
            with open(save_path + fname, "w") as f:
                f.write(write_text)
            f.close()
    print("Done!")