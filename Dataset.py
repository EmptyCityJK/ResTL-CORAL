import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from scipy.signal import butter, lfilter, resample

class OpenBMI_RS_MI_Dataset(Dataset):
    def __init__(self, root_dir, subjects, size=750, ch_num=62, is_training=True):
        super().__init__()
        self.samples = []
        self.size = size  # 每段 3s, 对应 750 个点（250Hz）
        self.ch_num = ch_num
        self.is_training = is_training
        self.session_num = 2

        for subj_id in subjects:
            for sess_id in range(1, self.session_num + 1):
                file_name = f"sess{sess_id:02d}_subj{subj_id:02d}_EEG_MI.mat"
                file_path = os.path.join(root_dir, file_name)

                if not os.path.exists(file_path):
                    print(f"[Warning] File not found: {file_path}")
                    continue

                mat_data = sio.loadmat(file_path)
                print(f"[\u2714] Loaded {file_name}")

                for phase in ['EEG_MI_train', 'EEG_MI_test']:
                    if phase not in mat_data:
                        continue
                    eeg_struct = mat_data[phase][0, 0]

                    raw_eeg = eeg_struct['x']       # (timepoints, channels)
                    t_list = eeg_struct['t'][0]     # (n_trials,)
                    y_dec = eeg_struct['y_dec'][0]  # (n_trials,)
                    fs = int(eeg_struct['fs'][0, 0])

                    for i, start in enumerate(t_list):
                        end = start + fs * 6  # 6s trial
                        if end > raw_eeg.shape[0]:
                            print(f"[Warning] Truncated trial at index {i} in {file_name}")
                            continue  # skip truncated trials

                        trial = raw_eeg[start:end, :]        # (6000, 62)
                        trial = resample(trial, 1500, axis=0)  # (1500, 62)
                        trial = self.bandpass(trial, 250, 0.5, 40)
                        trial = self.normalize(trial)

                        rs = trial[:750, :].T  # (62, 750)
                        ts = trial[750:, :].T  # (62, 750)

                        label = int(y_dec[i]) - 1  # 0=right, 1=left

                        self.samples.append({
                            'rs': rs,
                            'ts': ts,
                            'label': label,
                            'subject': subj_id
                        })

        self.labels = sorted(list(set([s['label'] for s in self.samples])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        anchor = self.samples[index]
        anchor_label = anchor['label']
        anchor_subj = anchor['subject']

        pos_candidates = [s for s in self.samples if s['label'] == anchor_label and s['subject'] == anchor_subj and s is not anchor]
        pos = random.choice(pos_candidates) if pos_candidates else anchor

        neg_candidates = [s for s in self.samples if s['label'] != anchor_label and s['subject'] == anchor_subj and s is not anchor]
        # print(neg_candidates.__len__(), anchor_label, anchor_subj)
        neg = random.choice(neg_candidates)
        assert isinstance(neg, dict) and 'rs' in neg and isinstance(neg['rs'], np.ndarray), f"Bad neg sample: {type(neg)}"
        # print("NEG TYPE", type(neg), isinstance(neg, dict), neg.keys() if isinstance(neg, dict) else "Not dict")
        # print("neg['rs'] shape", neg['rs'].shape)

        # print(neg['rs'].shape, neg['ts'].shape, neg['label'], neg['subject'])

        def to_tensor(x):
            return torch.FloatTensor(x).unsqueeze(0)  # (1, ch, size)

        return {
            'x_anc': to_tensor(anchor['ts']),
            'x_pos': to_tensor(pos['ts']),
            'x_neg': to_tensor(neg['ts']),
            'label': torch.nn.functional.one_hot(torch.tensor(anchor_label), num_classes=len(self.labels)).float(),
            'x_anc_rest': to_tensor(anchor['rs']),
            'x_pos_rest': to_tensor(pos['rs']),
            'x_neg_rest': to_tensor(neg['rs'])
        }

    def bandpass(self, data, fs, lowcut, highcut, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data, axis=0)

    def normalize(self, data):
        return (data - data.mean(axis=0, keepdims=True)) / (data.std(axis=0, keepdims=True) + 1e-5)

class OpenBMI_RSOnly_Dataset(Dataset):
    def __init__(self, root_dir, subject_id, size=750, ch_num=62, session=1):
        """
        加载指定被试某一会话的 RS 信号，供生成伪TS使用。
        注意：初始化阶段仅加载 RS 样本；TS 是由模型生成后缓存进 testset 的。
        """
        super().__init__()
        self.size = size
        self.ch_num = ch_num
        self.samples = []
        self.is_training = True  # 初始阶段为 RS-only，用于生成伪TS
        self.current_session = session  # 0-based 索引

        file_name = f"sess{session:02d}_subj{subject_id:02d}_EEG_MI.mat"
        file_path = os.path.join(root_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[!] Missing RS file: {file_path}")
        mat_data = sio.loadmat(file_path)
        print(f"[✔] Loaded RS data: {file_name}")

        if 'EEG_MI_train' not in mat_data:
            raise KeyError(f"[!] EEG_MI_train missing in {file_name}")

        eeg_struct = mat_data['EEG_MI_train'][0, 0]
        raw_eeg = eeg_struct['x']
        t_list = eeg_struct['t'][0]
        fs = int(eeg_struct['fs'][0, 0])

        for i, start in enumerate(t_list):
            end = start + fs * 6
            if end > raw_eeg.shape[0]:
                print(f"[!] Truncated RS trial {i}, skipping.")
                continue

            trial = raw_eeg[start:end, :]
            trial = resample(trial, 1500, axis=0)
            trial = self.bandpass(trial, 250, 0.5, 40)
            trial = self.normalize(trial)
            rs = trial[:750, :].T  # (62, 750)
            self.samples.append(torch.FloatTensor(rs).unsqueeze(0))  # (1, 62, 750)

        # 用于阶段2生成伪TS后存储数据
        self.testset = [{} for _ in range(2)]  # 对应 sess01 / sess02

    def __len__(self):
        if self.is_training:
            return len(self.samples)
        else:
            return self.testset[self.current_session]['x'].shape[0]

    def __getitem__(self, index):
        if self.is_training:
            # 阶段2：生成伪TS时，仅提供 x_anc_rest
            return self.samples[index]
        else:
            # 阶段3：利用 testset 生成完整三元组结构
            session_data = self.testset[self.current_session]
            x_pair = session_data['x'][index]         # (1, 62, 1500)
            label = session_data['y'][index].float()  # (num_class,) one-hot
            assert isinstance(x_pair, torch.Tensor)
            assert x_pair.shape == (1, self.ch_num, self.size * 2) 

            x_rest = x_pair[:, :, :750]  # (1, 62, 750)
            x_ts = x_pair[:, :, 750:]    # (1, 62, 750)

            return {
                'x_anc': x_ts,
                'x_pos': x_ts,
                'x_neg': x_ts,
                'label': label,
                'x_anc_rest': x_rest,
                'x_pos_rest': x_rest,
                'x_neg_rest': x_rest
            }

    def bandpass(self, data, fs, lowcut, highcut, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data, axis=0)

    def normalize(self, data):
        return (data - data.mean(axis=0, keepdims=True)) / (data.std(axis=0, keepdims=True) + 1e-5)