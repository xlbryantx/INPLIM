import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def train_eval(args, net, criterion, optimizer, loader, epoch, phase='Train'):
    assert phase in ['Train', 'Valid', 'Test']
    if phase == 'Train':
        net.train()
    else:
        net.eval()

    nll_cum, out_cum, lab_cum = [], [], []
    for i, batch in enumerate(tqdm(loader)):
        seqs, labels, days = batch
        if args.devices != 'cpu':
            seqs, labels, days = seqs.cuda(), labels.cuda(), days.cuda()

        output, _, _, _ = net(seqs, days)
        nll = criterion(output, labels)

        if phase == 'Train':
            optimizer.zero_grad()
            nll.backward()
            optimizer.step()

        nll_cum.append(nll.cpu().data.numpy())
        out_cum.append(F.softmax(output[:, :2], dim=1).cpu().detach().numpy())
        lab_cum.append(labels.cpu().detach().numpy())

    avg_nll = np.mean(nll_cum)
    auc_roc = roc_auc_score(np.concatenate(lab_cum, axis=0), np.concatenate(out_cum, axis=0)[:, 1])

    print('[{}]---[Epoch:{}]--[NLL:{:.4f}\tAUROC:{:.4f}]'.format(phase, epoch, avg_nll, auc_roc))
    return avg_nll, auc_roc
