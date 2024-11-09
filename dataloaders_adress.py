from torch.utils.data.dataset import Dataset, ConcatDataset
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import re
import itertools as it

train_path = 'ADReSS-IS2020-data/train'
test_path = 'ADReSS-IS2020-data/test'
test_label_path = 'ADReSS-IS2020-data/test/meta_data_test.txt'

train_21_path = 'ADReSSo21/diagnosis/train'
train_21_progression_path = 'ADReSSo21/progression/train'
test_21_path = 'ADReSSo21/diagnosis/test-dist'
test_21_label_task_1_path = 'ADReSSo21/diagnosis/test-dist/task1.csv'
test_21_label_task_2_path = 'ADReSSo21/diagnosis/test-dist/task2.csv'
asr_text_dir = 'asr_text_whisper'


def load_dataset_easy(dataset_str):
    if 'train' in dataset_str:
        return load_train_dataset(dataset_str)
    elif 'test' in dataset_str:
        return load_test_dataset(dataset_str)
    elif '-all' in dataset_str:
        return load_test_dataset(dataset_str.replace('-all', ''))
    else:
        raise ValueError('Check your dataset_str: train/test')


def load_train_dataset(train_dataset_name):
    train_dataset = None
    if train_dataset_name == 'ADReSSo21-train':
        train_dataset = ADReSSo21TextTrainDataset(
            train_21_path
        )
    elif train_dataset_name == 'ADReSS20-train':
        train_dataset = ADReSSTextTrainDataset(
            train_path
        )
    elif train_dataset_name == 'ADReSS20-train-transcript':
        train_dataset = ADReSSTextTranscriptTrainDataset(
            train_path
        )
    elif train_dataset_name == 'ADReSSo21-progression-train':
        train_dataset = ADReSSo21TextProgressionTrainDataset(
            train_21_progression_path
        )
    return train_dataset


def load_test_dataset(test_dataset_name):
    test_dataset = None
    if test_dataset_name == 'ADReSS20-train':
        test_dataset = ADReSSTextTrainDataset(
            train_path
        )
    elif test_dataset_name == 'ADReSS20-test':
        test_dataset = ADReSSTextTestDataset(
            test_path, test_label_path
        )
    elif test_dataset_name == 'ADReSS20':
        test_dataset_list = [ADReSSTextTrainDataset(
            train_path, ),
            ADReSSTextTestDataset(
                test_path, test_label_path, )]
        test_dataset = ConcatDataset(test_dataset_list)
    elif test_dataset_name == 'ADReSS20-transcript':
        test_dataset_list = [ADReSSTextTranscriptTrainDataset(
            train_path),
            ADReSSTextTranscriptTestDataset(
                test_path, test_label_path)]
        test_dataset = ConcatDataset(test_dataset_list)
    elif test_dataset_name == 'ADReSS20-test-transcript':
        test_dataset = ADReSSTextTranscriptTestDataset(
            test_path, test_label_path)
    elif test_dataset_name == 'ADReSSo21-progression-train':
        test_dataset = ADReSSo21TextProgressionTrainDataset(
            train_21_progression_path
        )
    elif test_dataset_name == 'ADReSSo21-test':
        test_dataset = ADReSSo21TextTestDataset(
            test_21_path, test_21_label_task_1_path, test_21_label_task_2_path,
        )
    return test_dataset


def get_file_text(file_path):
    text_dict = dict()
    with open(file_path, 'r') as f:
        text_dict['text'] = f.read()
    return text_dict


class ADReSSTextTrainDataset(Dataset):
    def __init__(self, dir_path):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        self.file_path_list = []
        mmse_labels = self.load_mmse(dir_path)
        for folder, sentiment in (('cc', 0), ('cd', 1)):
            folder = os.path.join(dir_path, asr_text_dir, folder)
            for name in tqdm(sorted(os.listdir(folder))):
                file_path = os.path.join(folder, name)
                text_file = get_file_text(file_path)
                self.X.append(text_file)
                self.Y.append(sentiment)
                self.Y_mmse.append(mmse_labels[name.split('.')[0]])
                self.file_idx.append(name.split('.')[0])
                self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
            'file_path': self.file_path_list[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_mmse(self, dir_path):
        labels = {}
        with open(os.path.join(dir_path, 'cc_meta_data.txt'), 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = lines[i].split(';')[3].strip()
                if file_label == 'NA':
                    file_label = 29
                else:
                    file_label = int(file_label)
                labels[file_id] = file_label
        with open(os.path.join(dir_path, 'cd_meta_data.txt'), 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = int(lines[i].split(';')[3].strip())
                labels[file_id] = file_label
        # print(labels)
        return labels


class ADReSSo21TextTrainDataset(Dataset):
    def __init__(self, dir_path):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        self.file_path_list = []
        mmse_labels = self.load_mmse(dir_path)
        for folder, sentiment in (('cn', 0), ('ad', 1)):
            folder = os.path.join(dir_path, asr_text_dir, folder)
            for name in tqdm(sorted(os.listdir(folder))):
                file_path = os.path.join(folder, name)
                text_file = get_file_text(file_path)
                self.X.append(text_file)
                self.Y.append(sentiment)
                self.Y_mmse.append(mmse_labels[name.split('.')[0]])
                self.file_idx.append(name.split('.')[0])
                self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
            'file_path': self.file_path_list[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_mmse(self, dir_path):
        labels = {}
        data = pd.read_csv(os.path.join(dir_path, 'adresso-train-mmse-scores.csv'))
        df = pd.DataFrame(data)

        for index, row in df.iterrows():
            # print(row[0], row['mmse'])
            labels[row['adressfname']] = int(row['mmse'])
        # print(labels)
        return labels


class ADReSSTextTestDataset(Dataset):
    def __init__(self, dir_path, label_path):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        self.file_path_list = []
        labels, mmse_labels = self.load_test_label(label_path)
        for name in tqdm(sorted(os.listdir(os.path.join(dir_path, asr_text_dir)))):
            file_path = os.path.join(dir_path, asr_text_dir, name)
            text_file = get_file_text(file_path)
            self.X.append(text_file)
            self.Y.append(labels[name.split('.')[0]])
            self.Y_mmse.append(mmse_labels[name.split('.')[0]])
            self.file_idx.append(name.split('.')[0])
            self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
            'file_path': self.file_path_list[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_test_label(self, label_path):
        labels = {}
        mmse_labels = {}
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = int(lines[i].split(';')[3].strip())
                file_mmse_label = int(lines[i].split(';')[4].strip())
                labels[file_id] = file_label
                mmse_labels[file_id] = file_mmse_label
        # print(labels)
        # print(mmse_labels)
        return labels, mmse_labels


class ADReSSTextTranscriptDataset(Dataset):
    def get_file_text(self, file_path):
        text_file = ''
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                text = line.strip().replace('	', ' ')
                if line.startswith('*'):
                    text = text.split(':', maxsplit=1)[1] + ' '
                    temp_idx = idx
                    while not '' in lines[temp_idx]:
                        temp_idx += 1
                        if temp_idx >= len(lines) or not lines[temp_idx].startswith('\t'):
                            break
                        text += lines[temp_idx].strip() + ' '
                    text = text.split('')[0]
                    text_file += text

        # print(text_file)
        text_file = text_file.replace('_', ' ')
        text_file = re.sub(r'\[[^\]]+\]', '', text_file)
        text_file = re.sub('[^0-9a-zA-Z,. \'?]+', '', text_file)
        text_file = text_file.replace('...', '').replace('..', '')
        # print(text_file)
        return text_file


class ADReSSTextTranscriptTrainDataset(ADReSSTextTranscriptDataset):
    def __init__(self, dir_path):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        self.file_path_list = []
        mmse_labels = self.load_mmse(dir_path)
        for folder, ad_label in (('cc', 0), ('cd', 1)):
            folder = os.path.join(dir_path, 'transcription', folder)
            for name in tqdm(sorted(os.listdir(folder))):
                file_path = os.path.join(folder, name)
                text_file = self.get_file_text(file_path)
                self.X.append(text_file)
                self.Y.append(ad_label)
                self.Y_mmse.append(mmse_labels[name.split('.')[0]])
                self.file_idx.append(name.split('.')[0])
                self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
            'file_path': self.file_path_list[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_mmse(self, dir_path):
        labels = {}
        with open(os.path.join(dir_path, 'cc_meta_data.txt'), 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = lines[i].split(';')[3].strip()
                if file_label == 'NA':
                    file_label = 29
                else:
                    file_label = int(file_label)
                labels[file_id] = file_label
        with open(os.path.join(dir_path, 'cd_meta_data.txt'), 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = int(lines[i].split(';')[3].strip())
                labels[file_id] = file_label
        # print(labels)
        return labels


class ADReSSTextTranscriptTestDataset(ADReSSTextTranscriptDataset):
    def __init__(self, dir_path, label_path):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        self.file_path_list = []
        labels, mmse_labels = self.load_test_label(label_path)
        for name in tqdm(sorted(os.listdir(os.path.join(dir_path, 'transcription')))):
            file_path = os.path.join(dir_path, 'transcription', name)
            text_file = self.get_file_text(file_path)
            self.X.append(text_file)
            self.Y.append(labels[name.split('.')[0]])
            self.Y_mmse.append(mmse_labels[name.split('.')[0]])
            self.file_idx.append(name.split('.')[0])
            self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
            'file_path': self.file_path_list[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_test_label(self, label_path):
        labels = {}
        mmse_labels = {}
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = int(lines[i].split(';')[3].strip())
                file_mmse_label = int(lines[i].split(';')[4].strip())
                labels[file_id] = file_label
                mmse_labels[file_id] = file_mmse_label
        # print(labels)
        # print(mmse_labels)
        return labels, mmse_labels


class ADReSSo21TextProgressionTrainDataset(Dataset):
    def __init__(self, dir_path):
        self.X, self.Y = [], []
        self.file_idx = []
        self.file_path_list = []
        for folder, sentiment in (('no_decline', 0), ('decline', 1)):
            folder = os.path.join(dir_path, asr_text_dir, folder)
            for name in tqdm(sorted(os.listdir(folder))):
                file_path = os.path.join(folder, name)
                text_file = get_file_text(file_path)
                self.X.append(text_file)
                self.Y.append(sentiment)
                self.file_idx.append(name.split('.')[0])
                self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            'label': self.Y[idx],
            'label_mmse': 0.0,
            'file_path': self.file_path_list[idx],
        }

    def __len__(self):
        return len(self.X)


class ADReSSo21TextTestDataset(Dataset):
    def __init__(self, dir_path, label_ad_path, label_mmse_path):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        self.file_path_list = []
        labels, mmse_labels = self.load_test_label(label_ad_path, label_mmse_path)
        for name in tqdm(sorted(os.listdir(os.path.join(dir_path, asr_text_dir)))):
            file_path = os.path.join(dir_path, asr_text_dir, name)
            text_file = get_file_text(file_path)
            self.X.append(text_file)
            self.Y.append(labels[name.split('.')[0]])
            self.Y_mmse.append(mmse_labels[name.split('.')[0]])
            self.file_idx.append(name.split('.')[0])
            self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
            'file_path': self.file_path_list[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_test_label(self, label_ad_path, label_mmse_path):
        labels = {}
        mmse_labels = {}

        data = pd.read_csv(label_ad_path)
        df = pd.DataFrame(data)
        for index, row in df.iterrows():
            # print(row[0], row['mmse'])
            labels[row['ID']] = 0 if row['Dx'] == "Control" else 1

        data = pd.read_csv(label_mmse_path)
        df = pd.DataFrame(data)
        for index, row in df.iterrows():
            # print(row[0], row['mmse'])
            mmse_labels[row['ID']] = int(row['MMSE'])

        # print(labels)
        # print(mmse_labels)
        return labels, mmse_labels


if __name__ == '__main__':
    train_data_id = 'ADReSSo21-train'
    # train_data_id = 'ADReSSo21-test'
    train_dataset = [data for data in load_dataset_easy(train_data_id)]

    for data in train_dataset:
        print(data['file_idx'], data['label'])
        print(data['text'])
        print()
