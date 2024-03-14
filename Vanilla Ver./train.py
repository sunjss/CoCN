from __future__ import division
from __future__ import print_function

from training_setting import args
from utils import CoCNDataLoader
from model.modules import PermGenModule, Ranker
from model.modules import CoCN, MiniBatchCoCN, CoCNModuleG, CoCNModuleN

import os
import random
import numpy as np
from tqdm import tqdm
from shutil import copyfile

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch
print("running on GPU" + str(args.cuda_num))
print("Tlayer:" + str(args.nTlayer) + "; layer:" + str(args.nlayer) + "; block:" + str(args.nblock))
print("heads:"+str(args.nh) + "; hidden:" + str(args.d_model) + "; filter size:" + str(args.filter_size))
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def lambda_lr(s):
    s += 1
    if s < args.warmup * args.nbatch:
        return float(s) / float(args.warmup * args.nbatch)
    return max(args.lr_lb, args.lr_step ** (s - args.warmup * args.nbatch))


def model_init(dataset):
    ranking_model = Ranker(N_enc=args.nTlayer,
                        d_in=nfeats, 
                        d_model=args.d_model, 
                        h=args.nh,
                        dropout=args.dropout)
    perm_gen_model = PermGenModule(d_model=args.d_model, temp=args.temp)
    if dataset in ["ACTOR", "CORNELL", "TEXAS", "WISCONSIN", "SQUIRREL", "CHAMELEON"]:
        conv = CoCNModuleN(h=args.nh,
                            d_in=nfeats,
                            d_ein=nedgefeats,
                            d_model=args.d_model,
                            nclass=nclass,
                            filter_size=args.filter_size,
                            stride=args.stride,
                            nlayers=args.nlayer,
                            nblocks=args.nblock,
                            dropout=args.dropout)
        model = CoCN(ranking_model, perm_gen_model, conv)
    else:
        conv = CoCNModuleG(h=args.nh,
                            d_in=nfeats,
                            d_ein=nedgefeats,
                            d_model=args.d_model,
                            nclass=nclass,
                            filter_size=args.filter_size,
                            stride=args.stride,
                            nlayers=args.nlayer,
                            nblocks=args.nblock,
                            dropout=args.dropout)
        model = MiniBatchCoCN(ranking_model, perm_gen_model, conv)
    cocn_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_func = nn.NLLLoss()
    if args.cuda:
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model, cocn_optimizer, loss_func


def model_train(model, optimizer, scheduler):
    model.train()
    running_loss = .0
    acc = .0
    for it, data in enumerate(train_data):
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
            labels = data.y.flatten()
        output, _ = model(data)
        if args.dataset in ["ACTOR", "SQUIRREL", "CHAMELEON", "CORNELL", "TEXAS", "WISCONSIN"]:
            train_mask = data.train_mask[:, args.fold_idx]
            loss_train = loss_func(output[train_mask], labels[train_mask])
            acc_train = accuracy(output[train_mask], labels[train_mask])
        else:
            loss_train = loss_func(output, labels)
            acc_train = accuracy(output, labels)
        loss_train.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss_train.data.item()
        acc += acc_train.item()
        tqdm_t.set_description('lr %.4e, %d/%d %.4e' %
                               (scheduler.get_last_lr()[0],
                               it, len(train_data),
                               running_loss / (it + 1)) +\
                               val_des + sum_des)
        writer.add_scalar('data/train_loss', running_loss / (it + 1), epoch * len(train_data) + it)
        writer.add_scalar('data/train_acc', acc / (it + 1), epoch * len(train_data) + it)
    writer.add_scalar('data/epoch_train_loss', running_loss / len(train_data), epoch)
    writer.add_scalar('data/epoch_train_acc', acc / len(train_data), epoch)
    return running_loss / len(train_data), acc / len(train_data)


def model_val(model):
    # Evaluate validation set performance separately,
    # deactivates dropout during validation run.
    model.eval()
    running_loss = .0
    acc = .0
    with torch.no_grad():
        for it, data in enumerate(val_data):
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
                labels = data.y.flatten()
            output, _ = model(data)
            if args.dataset in ["ACTOR", "SQUIRREL", "CHAMELEON", "CORNELL", "TEXAS", "WISCONSIN"]:
                val_mask = data.val_mask[:, args.fold_idx]
                loss_val = loss_func(output[val_mask], labels[val_mask])
                acc_val = accuracy(output[val_mask], labels[val_mask])
            else:
                loss_val = loss_func(output, labels)
                acc_val = accuracy(output, labels)
            if args.cuda:
                torch.cuda.empty_cache()
            acc += acc_val.item()
            running_loss += loss_val.data.item()
            tqdm_t.set_description('lr %.4e' % (scheduler.get_last_lr()[0]) + \
                                    train_des + ', val %d/%d %f' % 
                                    (it, len(val_data), running_loss / (it + 1))+ \
                                    sum_des)
            writer.add_scalar('data/val_loss', running_loss / (it + 1), epoch * len(val_data) + it)
            writer.add_scalar('data/val_acc', acc / (it + 1), epoch * len(val_data) + it)
    writer.add_scalar('data/epoch_val_loss', running_loss / len(val_data), epoch)
    writer.add_scalar('data/epoch_val_acc', acc / len(val_data), epoch)

    return running_loss / len(val_data), acc / len(val_data)

def model_test(model):
    model.eval()
    running_loss = .0
    acc = .0
    with tqdm(desc='Test', unit='it', total=len(test_data)) as pbar, torch.no_grad():
        for it, data in enumerate(test_data):
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
                labels = data.y.flatten()
            output, _ = model(data)
            if args.dataset in ["ACTOR", "SQUIRREL", "CHAMELEON", "CORNELL", "TEXAS", "WISCONSIN"]:
                test_mask = data.test_mask[:, args.fold_idx]
                loss_test = loss_func(output[test_mask], labels[test_mask])
                acc_test = accuracy(output[test_mask], labels[test_mask])
            else:
                loss_test = loss_func(output, labels)
                acc_test = accuracy(output, labels)
            if args.cuda:
                torch.cuda.empty_cache()
            acc += acc_test.item()
            running_loss += loss_test.data.item()
            pbar.update()
    print("Test set results:",
            "loss= {:.4f}".format(running_loss / len(test_data)),
            "accuracy= {:.4f}".format(acc / len(test_data)))

    return round(running_loss / len(test_data), 5), round(acc / len(test_data), 5)


if __name__ == '__main__':
    # Load data
    data_loader = CoCNDataLoader()
    print("Loading " + args.dataset.title() + "...")
    flag = (args.dataset in ["NCI1", "IMDB-BINARY", "IMDB-MULTI", "MUTAG", "PROTEINS", "COLLAB"])
    graph_iter_range = 10 if flag else 1
    for k in range(graph_iter_range):
        args.fold_idx = k if graph_iter_range > 1 else args.fold_idx
        data_loader.load_data(dataset=args.dataset, nbatch=args.nbatch, fold_idx=args.fold_idx)
        nclass, nfeats, nedgefeats = data_loader.nclass, data_loader.nfeats, data_loader.nedgefeats
        train_data, val_data, test_data = data_loader.train_data, data_loader.val_data, data_loader.test_data
        accuracy = data_loader.accuracy

        flag = (args.dataset in ["SQUIRREL", "CHAMELEON", "CORNELL", "TEXAS", "WISCONSIN", "ACTOR"])
        iter_range = 10 if flag else 1
        for i in range(iter_range):
            save_path = 'export/' + args.testmode + args.dataset + '/'
            model_config =  '%d_%d_%d_%d_%d_%d_%d/' % (args.nTlayer, args.nblock, args.nlayer, args.filter_size, args.stride, args.nh, args.d_model)
            args.fold_idx = i if iter_range > 1 else args.fold_idx
            if args.dataset in ["NCI1", "IMDB-BINARY", "IMDB-MULTI", "MUTAG", "PROTEINS", "COLLAB", "SQUIRREL", "CHAMELEON", "CORNELL", "TEXAS", "WISCONSIN", "ACTOR"]:
                model_config = model_config + '%d/' % (args.fold_idx)
                print("Running on split " + str(args.fold_idx))
            save_path = save_path + model_config
            is_exists = os.path.exists(save_path)
            if not is_exists:
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
                            if bad_counter > l_bad_cnt:
                                l_bad_cnt = bad_counter
                            bad_counter = 0
                            copyfile(save_path + logfile + "_last.pth", save_path + logfile + "_best.pth")
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
            write_text = write_text + 'epochs = ' + str(args.epochs) + '\n'
            write_text = write_text + 'lr = ' + str(args.lr) + '\n'
            write_text = write_text + 'weight_decay = ' + str(args.weight_decay) + '\n'
            write_text = write_text + 'batch nums = ' + str(args.nbatch) + '\n'
            write_text = write_text + 'dropout = ' + str(args.dropout) + '\n'
            write_text = write_text + "-------model arg-------" + '\n'
            write_text = write_text + 'd_model = ' + str(args.d_model) + '\n'
            write_text = write_text + 'nblocks = ' + str(args.nblock) + '\n'
            write_text = write_text + 'nlayers = ' + str(args.nlayer) + '\n'
            write_text = write_text + 'nTlayers = ' + str(args.nTlayer) + '\n'
            write_text = write_text + 'nh = ' + str(args.nh) + '\n'
            write_text = write_text + 'filter_size = ' + str(args.filter_size) + '\n'
            write_text = write_text + 'stride = ' + str(args.stride) + '\n'
            write_text = write_text + "-------results-------" + '\n'
            write_text = write_text + 'loss = ' + str(test_loss) + '\n' + 'acc = ' + str(test_acc) + '\n'
            write_text = write_text + 'best val loss = ' + str(best) + '\n'
            fname = '/' + args.dataset + '_result.txt'
            with open(save_path + fname, "w") as f:
                f.write(write_text)
            f.close()