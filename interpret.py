import os
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from data_utils import CodeLevelDataset
from torch.utils.data import DataLoader
from Model.INPLIM import Doctor


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./datasets/heart/dataset_1',
                        help='Setting the root path of the dataset')
    parser.add_argument('--model_path', type=str, default='./saved_models/Model-AUC-0.7384.pth',
                        help='Setting the model path.')
    parser.add_argument('--save_path', type=str, default='./saved_explanation')
    return parser


def get_contribution(seqs, alpha, beta, feat_emb, w_fc, w_ft, w_out):
    seqs = seqs[seqs != 0]
    contri_a, contri_c, contri_t = [], [], []
    for i, code in enumerate(seqs):
        c_c = np.matmul(np.matmul(feat_emb[i] * alpha[i], w_fc), w_out)
        c_t = np.matmul(np.matmul(feat_emb[i] * beta[i], w_ft), w_out)
        c_a = c_c + c_t
        contri_a.append(c_a)
        contri_c.append(c_c)
        contri_t.append(c_t)

    return seqs, contri_c, contri_t, contri_a


def main(args):
    model_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
    opts, model_states = model_dict['opts'], model_dict['states']

    dataset = pickle.load(open(args.data_root, 'rb'))
    test_set = CodeLevelDataset(dataset=dataset, max_len=opts.max_len, phase='test')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=1, shuffle=False)

    net = Doctor(features=dataset['features'], out_dim=2, emb_dim=opts.dim, dropout_context=opts.drop_context,
                 dropout_time=opts.drop_time)

    w_fuse, w_out = model_states['w_fuse.weight'].numpy(), model_states['w_out.weight'].numpy()[1]
    w_fc, w_ft = w_fuse[:, :opts.dim].T, w_fuse[:, opts.dim:].T
    net.eval()

    interpretations = {}
    for b, batch in enumerate(tqdm(test_loader)):
        seqs, labels, days = batch
        output, alpha, beta, feat_emb = net(seqs, days)
        seqs, a, c, t = get_contribution(seqs.squeeze().detach().numpy(), alpha.squeeze().detach().numpy(),
                                         beta.squeeze().detach().numpy(), feat_emb.squeeze().detach().numpy(),
                                         w_fc, w_ft, w_out)
        inp = {
            'Code': seqs,
            'Contribution_all': a,
            'Contribution_context': c,
            'Contribution_time': t,
            'Out': output.squeeze().detach().numpy()
        }
        interpretations[b] = inp

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    pickle.dump(interpretations, open(os.path.join(args.save_path,
                                                   args.model_path.split('/')[-1] + '-exp.pkl'), 'wb'))


if __name__ == '__main__':
    opts = args().parse_args()
    main(opts)
