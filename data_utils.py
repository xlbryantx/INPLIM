import numpy as np
from torch.utils.data import Dataset


def reconstruct_features(seqs, days, max_len=200):
    seqs_, days_ = [], []
    for p, patient in enumerate(seqs):
        patient_codes, patient_times = [], []
        for v, visit in enumerate(patient):
            for c, code in enumerate(visit):
                patient_codes.append(code)
                patient_times.append(days[p][v])

        if len(patient_codes) > max_len:
            patient_codes = patient_codes[:-(max_len - 1)]
            patient_times = patient_times[:-(max_len - 1)]

        seqs_.append(np.concatenate((np.array(patient_codes), np.zeros([max_len - len(patient_codes)]))).
                     astype(np.int))
        patient_times = np.array(patient_times)
        patient_times = np.abs(patient_times - np.max(patient_times)) / 1000.0
        days_.append(np.concatenate((patient_times, np.ones([max_len - patient_times.shape[0]]) * -1)).
                     astype(np.float32))
    return seqs_, days_


class CodeLevelDataset(Dataset):
    def __init__(self, dataset, max_len, phase='train'):

        seqs, self.labels, days = dataset['{}_seqs'.format(phase)], dataset['{}_labels'.format(phase)], \
                             dataset['{}_days'.format(phase)]

        self.seqs, self.days = reconstruct_features(seqs, days, max_len=max_len)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, item):
        return self.seqs[item], self.labels[item], self.days[item]
