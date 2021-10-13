import os
import pickle
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from Model.INPLIM import Doctor
from data_utils import CodeLevelDataset
from utils import train_eval


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='',
                        help='Set the root path of the dataset')
    parser.add_argument('--devices', type=str, default='cpu',
                        help='Setting the IDs of GPU devices.')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Setting the number of epochs to run.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Setting the mini-batch size.')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Setting weight decay')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Setting the learning rate.')
    parser.add_argument('--dim', type=int, default=128,
                        help='Setting the inner dim of the model.')
    parser.add_argument('--max_len', type=int, default=200,
                        help='Setting the maximum number of code to use for a patient.')
    parser.add_argument('--drop_context', type=float, default=0.3,
                        help='Setting drop rate of the context-aware branch.')
    parser.add_argument('--drop_time', type=float, default=0.3,
                        help='Setting drop rate of the time-aware branch.')
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save the parameters of the trained model.',
                        default=True)
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                        help='Setting the dir of saving trained model.')
    return parser


def main(opts):
    if opts.devices != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = opts.devices

    dataset = pickle.load(open(opts.data_root, 'rb'))
    train_set = CodeLevelDataset(dataset=dataset, max_len=opts.max_len, phase='train')
    valid_set = CodeLevelDataset(dataset=dataset, max_len=opts.max_len, phase='val')
    test_set = CodeLevelDataset(dataset=dataset, max_len=opts.max_len, phase='test')

    train_loader = DataLoader(train_set, batch_size=opts.batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(valid_set, batch_size=1, num_workers=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=1, shuffle=False)

    net = Doctor(features=dataset['features'], out_dim=2, emb_dim=opts.dim, dropout_context=opts.drop_context,
                 dropout_time=opts.drop_time)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay, eps=0)

    if opts.devices != 'cpu':
        net = torch.nn.DataParallel(net).cuda()
        criterion = criterion.cuda()

    best_auc, best_epoch, best_test_nll, best_test_auc_roc = 0, 0, 0, 0
    model_dict = {}
    for epoch in range(opts.epochs):
        train_eval(opts, net, criterion, optimizer, train_loader, epoch, phase='Train')
        _, auc = train_eval(opts, net, criterion, optimizer, val_loader, epoch, phase='Valid')

        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_test_nll, best_test_auc_roc = train_eval(opts, net, criterion, optimizer, test_loader, epoch,
                                                          phase='Test')
            model_dict['opts'] = opts
            model_dict['states'] = net.state_dict()
    print('Best Test NLL:{:.4f}\t Best AUROC:{:.4f}'.format(best_test_nll, best_test_auc_roc))

    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    if opts.save_model:
        torch.save(model_dict, os.path.join(opts.save_dir, 'Model-AUC-{:.4f}.pth'.format(best_test_auc_roc)))


if __name__ == '__main__':
    opts = args().parse_args()
    main(opts)
