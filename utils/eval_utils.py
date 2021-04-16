import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM
from models.model_attention_mil import MIL_Attention_fc
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import EarlyStopping,  Accuracy_Logger
from utils.file_utils import save_pkl, load_pkl
from sklearn.metrics import roc_auc_score, roc_curve, auc
import h5py
from models.resnet_custom import resnet50_baseline
import math
from sklearn.preprocessing import label_binarize

def initiate_model(args, ckpt_path=None):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type in ['clam', 'attention_mil', 'clam_new']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam':
        model = CLAM(**model_dict)
    elif args.model_type == 'attention_mil':
        model = MIL_Attention_fc(**model_dict)    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    model.relocate()
    #print_network(model)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, aucs, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    for cls_idx in range(len(aucs)):
        print('class {} auc: {}'.format(cls_idx, aucs[cls_idx]))
    return model, patient_results, test_error, auc, aucs, df

def infer(dataset, args, ckpt_path, class_labels):
    model = initiate_model(args, ckpt_path)
    df = infer_dataset(model, dataset, args, class_labels)
    return model, df

# Code taken from pytorch/examples for evaluating topk classification on on ImageNet
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)
    if args.n_classes > 2:
        acc1, acc3 = accuracy(torch.from_numpy(all_probs), torch.from_numpy(all_labels), topk=(1, 3))
        print('top1 acc: {:.3f}, top3 acc: {:.3f}'.format(acc1.item(), acc3.item()))
        
    if len(np.unique(all_labels)) == 1:
        auc_score = -1
    else:
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
            aucs = []
        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, aucs, df, acc_logger

def infer_dataset(model, dataset, args, class_labels, k=3):
    model.eval()
    all_probs = np.zeros((len(dataset), k))
    all_preds = np.zeros((len(dataset), k))
    all_preds_str = np.full((len(dataset), k), ' ', dtype=object)
    slide_ids = dataset.slide_data
    for batch_idx, data in enumerate(dataset):
        data = data.to(device)
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        probs, ids = torch.topk(Y_prob, k)
        probs = probs.cpu().numpy()
        ids = ids.cpu().numpy()
        all_probs[batch_idx] = probs
        all_preds[batch_idx] = ids
        all_preds_str[batch_idx] = np.array(class_labels)[ids]
    del data
    results_dict = {'slide_id': slide_ids}
    for c in range(k):
        results_dict.update({'Pred_{}'.format(c): all_preds_str[:, c]})
        results_dict.update({'p_{}'.format(c): all_probs[:, c]})
    df = pd.DataFrame(results_dict)
    return df

# def infer_dataset(model, dataset, args, class_labels, k=3):
#     model.eval()

#     all_probs = np.zeros((len(dataset), args.n_classes))
#     all_preds = np.zeros(len(dataset))
#     all_str_preds = np.full(len(dataset), ' ', dtype=object)

#     slide_ids = dataset.slide_data
#     for batch_idx, data in enumerate(dataset):
#         data = data.to(device)
#         with torch.no_grad():
#             logits, Y_prob, Y_hat, _, results_dict = model(data)
        
#         probs = Y_prob.cpu().numpy()
#         all_probs[batch_idx] = probs
#         all_preds[batch_idx] = Y_hat.item()
#         all_str_preds[batch_idx] = class_labels[Y_hat.item()]
#     del data

#     results_dict = {'slide_id': slide_ids, 'Prediction': all_str_preds, 'Y_hat': all_preds}
#     for c in range(args.n_classes):
#         results_dict.update({'p_{}_{}'.format(c, class_labels[c]): all_probs[:,c]})
#     df = pd.DataFrame(results_dict)
#     return df

def compute_features(dataset, args, ckpt_path, save_dir, model=None, feature_dim=512):
    if model is None:
        model = initiate_model(args, ckpt_path)

    names = dataset.get_list(np.arange(len(dataset))).values
    file_path = os.path.join(save_dir, 'features.h5')

    initialize_features_hdf5_file(file_path, len(dataset), feature_dim=feature_dim, names=names)
    for i in range(len(dataset)):
        print("Progress: {}/{}".format(i, len(dataset)))
        save_features(dataset, i, model, args, file_path)

def save_features(dataset, idx, model, args, save_file_path):
    name = dataset.get_list(idx)
    print(name)
    features, label = dataset[idx]
    features = features.to(device)
    with torch.no_grad():
        if type(model) == CLAM:
            _, Y_prob, Y_hat, _, results_dict = model(features, instance_eval=False, return_features=True)
            bag_feat = results_dict['features'][Y_hat.item()]
        else:
            _, Y_prob, Y_hat, _, results_dict = model(features, return_features=True)
            bag_feat = results_dict['features']
    del features
    Y_hat = Y_hat.item()
    Y_prob = Y_prob.view(-1).cpu().numpy()
    bag_feat = bag_feat.view(1, -1).cpu().numpy()

    with h5py.File(save_file_path, 'r+') as file:
        print('label', label)
        file['features'][idx, :] = bag_feat
        file['label'][idx] = label
        file['Y_hat'][idx] = Y_hat
        file['Y_prob'][idx] = Y_prob[Y_hat]

def initialize_features_hdf5_file(file_path, length, feature_dim=512, names = None):
    
    file = h5py.File(file_path, "w")

    dset = file.create_dataset('features', 
                                shape=(length, feature_dim), chunks=(1, feature_dim), dtype=np.float32)

    # if names is not None:
    #     names = np.array(names, dtype='S')
    #     dset.attrs['names'] = names
    if names is not None:
        dt = h5py.string_dtype()
        label_dset = file.create_dataset('names', 
                                        shape=(length, ), chunks=(1, ), dtype=dt)
    
    label_dset = file.create_dataset('label', 
                                        shape=(length, ), chunks=(1, ), dtype=np.int32)

    pred_dset = file.create_dataset('Y_hat', 
                                        shape=(length, ), chunks=(1, ), dtype=np.int32)

    prob_dset = file.create_dataset('Y_prob', 
                                        shape=(length, ), chunks=(1, ), dtype=np.float32)

    file.close()
    return file_path



def eval2(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    if args.pretrain_VAE:
        print("Initializing VAE")
        VAE = GenomicVAE(input_dim=args.omic_input_dim, hidden=[1024, 256, 128])
        ckpt = torch.load('./VAE/logs/tcga_base/000-all/%d/%d/%d_best.ckpt' % (cur, cur, cur))
        state_dict = ckpt['state_dict']
        state_dict = OrderedDict((k[6:], v) for k, v in state_dict.items())
        VAE.load_state_dict(state_dict)
        args.omic_input_dim = 128
        VAE.relocate()
        dfs_freeze(VAE)
        VAE.eval()
    else:
        VAE = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'cox_surv':
            loss_fn = CoxSurvLoss()
        else:
            raise NotImplementedError
    else:
        if args.bag_loss == 'svm':
            from topk import SmoothTop1SVM
            loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
            if device.type == 'cuda':
                loss_fn = loss_fn.cuda()
        elif args.bag_loss == 'ce':
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

    if args.reg_type == 'omic':
        reg_fn = l1_reg_all
    elif args.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type in ['clam', 'clam_simple'] and args.subtyping:
        model_dict.update({'subtyping': True})
    
        if args.model_size is not None:
            model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam', 'clam_simple']:
        if args.task_type == 'survival':
            raise NotImplementedError
        else:
            if args.inst_loss == 'svm':
                from topk import SmoothTop1SVM
                instance_loss_fn = SmoothTop1SVM(n_classes = 2)
                if device.type == 'cuda':
                    instance_loss_fn = instance_loss_fn.cuda()
            else:
                instance_loss_fn = nn.CrossEntropyLoss()
            
            if args.model_type =='clam':
                model = CLAM(**model_dict, instance_loss_fn=instance_loss_fn)
            else:
                model = CLAM_Simple(**model_dict, instance_loss_fn=instance_loss_fn)

    elif args.model_type =='attention_mil':
        if args.task_type == 'survival':
            model = MIL_Attention_fc_surv(**model_dict)
            # model.alpha.requires_grad = False
        else:
            model = MIL_Attention_fc(**model_dict)

    elif args.model_type =='mm_attention_mil':
        model_dict.update({'input_dim': args.omic_input_dim, 'meta_dim': args.meta_dim, 
            'fusion': args.fusion, 'model_size_wsi':args.model_size_wsi, 'model_size_omic':args.model_size_omic,
            'gate_path': args.gate_path, 'gate_omic': args.gate_omic, 'n_classes': args.n_classes, 
            'pretrain': args.pretrain, 'tcga_proj': '_'.join(args.task.split('_')[:2]), 'split_idx': cur})

        if args.task_type == 'survival':
            model = MM_MIL_Attention_fc_surv(**model_dict)
            # model.alpha.requires_grad = False
        else:
            model = MM_MIL_Attention_fc(**model_dict)

    elif args.model_type =='max_net':
        model_dict = {'input_dim': args.omic_input_dim, 'meta_dim': args.meta_dim, 'model_size_omic': args.model_size_omic, 'n_classes': args.n_classes}
        if args.task_type == 'survival':
            model = MaxNet(**model_dict)
            # model.alpha.requires_grad = False
        else:
            raise NotImplementedError

    else: # args.model_type == 'mil'
        if args.task_type == 'survival':
            raise NotImplementedError
        else:
            if args.n_classes > 2:
                model = MIL_fc_mc(**model_dict)
            else:
                model = MIL_fc(**model_dict)

    model.relocate()
    print('Done!')
    print_network(model)
    ckpt = torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, 
                                    weighted = args.weighted_sample, task_type=args.task_type, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split,  testing = args.testing, task_type=args.task_type, batch_size=args.batch_size)
    test_loader = get_split_loader(test_split, testing = args.testing, task_type=args.task_type, batch_size=args.batch_size)
    print('Done!')


    if args.task_type == 'survival':
        results_val_dict, val_c_index = summary_survival(model, val_loader, args.n_classes, VAE)
        print('Val c-index: {:.4f}'.format(val_c_index))

        results_test_dict, test_c_index = summary_survival(model, test_loader, args.n_classes, VAE)
        print('Test c-index: {:.4f}'.format(test_c_index))

        if writer:
            writer.add_scalar('final/val_c_index', val_c_index, 0)
            writer.add_scalar('final/test_c_index', test_c_index, 0)
        
        writer.close()
        return results_val_dict, results_test_dict, val_c_index, test_c_index

    elif args.task_type == 'classification':
        pass
