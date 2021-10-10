import numpy as np
import torch
import pickle 
from utils.utils import *
import os
from collections import OrderedDict

from argparse import Namespace
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored


def train_loop_survival_cluster(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16, VAE=None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    
    for batch_idx, (data_WSI, cluster_id, data_omic, meta, label, event_time, c) in enumerate(loader):
        
        if isinstance(data_WSI, torch_geometric.data.Batch):
            if data_WSI.x.shape[0] > 150000:
                continue
        else:
            if data_WSI.shape[0] > 150000:
                continue

        data_WSI, cluster_id, data_omic, meta = data_WSI.to(device), cluster_id, data_omic.to(device), meta.to(device)
        label = label.to(device)
        c = c.to(device)

        hazards, S, Y_hat, _, _ =  model(x_path=data_WSI, cluster_id=cluster_id, x_omic=data_omic, meta=meta) # return hazards, S, Y_hat, A_raw, results_dict
        
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, label.item(), float(event_time), float(risk), data_WSI.size(0)))
        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)

def validate_survival_cluster(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, VAE=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    # loader.dataset.update_mode(True)
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    # model.alpha.requires_grad=True

    for batch_idx, (data_WSI, cluster_id, data_omic, meta, label, event_time, c) in enumerate(loader):
        if isinstance(data_WSI, torch_geometric.data.Batch):
            if data_WSI.x.shape[0] > 150000:
                continue
        else:
            if data_WSI.shape[0] > 150000:
                continue
        data_WSI, cluster_id, data_omic, meta = data_WSI.to(device), cluster_id, data_omic.to(device), meta.to(device)
        label = label.to(device)
        c = c.to(device)

        with torch.no_grad():
            hazards, S, Y_hat, _, _ = model(x_path=data_WSI, cluster_id=cluster_id, x_omic=data_omic, meta=meta) # return hazards, S, Y_hat, A_raw, results_dict
        
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg


        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)

    #c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)


    #monitor_cindex(c_index, model, ckpt_name=os.path.join(results_dir, "s_{}_maxval_checkpoint.pt".format(cur)))
    if epoch == 10:
        torch.save(model.state_dict(), os.path.join(results_dir, 's_%d_mid_checkpoint.pt' % cur))
    print('\nVal Set, val_loss_surv: {:.4f}, val_loss: {:.4f}, val c-index: {:.4f}'.format(val_loss_surv, val_loss, c_index))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    # model.alpha.requires_grad=False
    return False


def summary_survival_cluster(model, loader, n_classes, VAE):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, cluster_id, data_omic, meta, label, event_time, c) in enumerate(loader):
        if isinstance(data_WSI, torch_geometric.data.Batch):
            if data_WSI.x.shape[0] > 150000:
                continue
        else:
            if data_WSI.shape[0] > 150000:
                continue
        data_WSI, cluster_id, data_omic, meta = data_WSI.to(device), cluster_id, data_omic.to(device), meta.to(device)
        label = label.to(device)
        c = c.to(device)

        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            hazards, survival, Y_hat, _, _ = model(x_path=data_WSI, cluster_id=cluster_id, x_omic=data_omic, meta=meta) # return hazards, S, Y_hat, A_raw, results_dict


        risk = np.asscalar(-torch.sum(survival, dim=1).cpu().numpy())
        event_time = np.asscalar(event_time)
        c = np.asscalar(c)
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time, 'censorship': c}})

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return patient_results, c_index