import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_provider.m4 import M4Dataset, M4Meta
from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe
import warnings
from PIL import Image
from torchvision import transforms
import math # Import math

warnings.filterwarnings('ignore')

    
class Dataset_Satellite(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=False, chunksize=10000, cols_data=None):
        """
        data_paths: List of CSV file names in the root_path directory
        cols_data: List of column names or indices to read from each CSV
        """
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        self.chunksize = chunksize
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        data_paths=[]
        for file_name in os.listdir(self.root_path):
            if file_name.endswith(".csv"):
                data_paths.append(file_name)
        self.data_paths = data_paths  # List of CSV file paths
        self.data_x_list = []  # To store processed data from all CSVs
        self.data_y_list = []  # To store labels from all CSVs
        self.data_stamp_list = []  # To store timestamps from all CSVs
        self.tot_len_list = []  # To store lengths of data from all CSVs

        # Accept columns to read as argument
        self.cols_data = cols_data

        self.__read_data__()
        self.enc_in = self.data_x_list[0].shape[-1]
        self.tot_len = sum(self.tot_len_list)  # Total length across all CSVs

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.total_files_no=0
        for data_path in self.data_paths:
            # Load the CSV file
            df_raw = pd.read_csv(os.path.join(self.root_path, data_path))

            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            # Use columns from argument if provided, else fallback to previous behavior
            if self.cols_data is not None:
                # Always flatten to a list of strings
                if isinstance(self.cols_data, str):
                    cols_data = [self.cols_data]
                elif isinstance(self.cols_data, (list, tuple, set)):
                    # flatten if it's a single tuple/list/set inside a list
                    if len(self.cols_data) == 1 and isinstance(self.cols_data[0], (tuple, list, set)):
                        cols_data = list(self.cols_data[0])
                    else:
                        cols_data = list(self.cols_data)
                else:
                    raise ValueError("cols_data must be a string or a list/tuple/set of strings")
            else:
                cols_data = df_raw.columns[-3]  # Default: previous behavior

            df_data = df_raw[cols_data]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            # Load the corresponding .pt file
            data_name = data_path.split('.')[0]
            pt_file_path = os.path.join(self.root_path, f'{data_path}.pt')
            if os.path.exists(pt_file_path):
                data_stamp = torch.load(pt_file_path)
                data_stamp = data_stamp[border1:border2]
            else:
                raise FileNotFoundError(f"Missing .pt file for {data_name}. Expected at: {pt_file_path}")

            # Store the processed data
            data_x = data[border1:border2]
            data_y = data[border1:border2]

            self.data_x_list.append(data_x)
            self.data_y_list.append(data_y)
            self.data_stamp_list.append(data_stamp)
            self.tot_len_list.append(len(data_x)) # Track the length of each dataset
            self.total_files_no += 1
         
    def __getitem__(self, index):
        # Determine which CSV the index belongs to
        cumulative_len = 0
        for i, length in enumerate(self.tot_len_list*self.enc_in):
            #print(f'index:{index},cumulative_len + length:{cumulative_len + length}')
            if index < cumulative_len + length:
                #print('ffffffffffffffffff')
                dataset_idx = i % self.total_files_no
                local_index = index - cumulative_len
                break
            cumulative_len += length

        data_x = self.data_x_list[dataset_idx]
        data_y = self.data_y_list[dataset_idx]
        data_stamp = self.data_stamp_list[dataset_idx]

        tot_len = len(data_x) - self.seq_len - self.pred_len + 1

        # Debugging log
        #print(f"Index: {index}, dataset_idx: {dataset_idx}, local_index: {local_index}, tot_len: {tot_len}")

        feat_id = index // self.tot_len
        s_begin = local_index % tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        # Debugging log for slicing
        #print(f"Index: {index},feat_id:{feat_id}, s_begin: {s_begin}, s_end: {s_end}, r_begin: {r_begin}, r_end: {r_end}")
        #print(f"data_x.shape: {data_x.shape}, data_y.shape: {data_y.shape}")

        # Validate boundaries
        if s_end > len(data_x) or r_end > len(data_y):
            raise IndexError(
                f"Invalid slicing: s_begin={s_begin}, s_end={s_end}, r_begin={r_begin}, r_end={r_end}, "
                f"len(data_x)={len(data_x)}, len(data_y)={len(data_y)},{self.data_paths[dataset_idx]}"
            )

        seq_x = data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = data_stamp[s_end:r_end:self.token_len]

        # Debugging log for tensor shapes
        #print(f"seq_x shape: {seq_x.shape}, seq_y shape: {seq_y.shape}")

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        #print(f'self.total_files_no={self.total_files_no}')
        #return self.tot_len
        total_len=(self.tot_len - self.total_files_no*(self.seq_len + self.pred_len - 1)) * self.enc_in
        #print(f'totallen={total_len}')
        return total_len

    
    
class Dataset_Preprocess_Sat(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTh1.csv', scale=True, seasonal_patterns=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        self.data_set_type = data_path.split('.')[0]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.tot_len = len(self.data_stamp)

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_stamp = df_raw[['epoch']]
        df_stamp['epoch'] = pd.to_datetime(df_stamp.epoch).apply(str)
        self.data_stamp = df_stamp['epoch'].values
        self.data_stamp = [str(x) for x in self.data_stamp]
        df_rcs = df_raw[['rcs']]
        self.data_rcs = df_rcs['rcs'].values[0]

    def __getitem__(self, index):
        s_begin = index % self.tot_len
        s_end = s_begin + self.token_len
        try:
            start = datetime.datetime.strptime(self.data_stamp[s_begin], "%Y-%m-%d %H:%M:%S")
        except ValueError as v:
            ulr = len(v.args[0].partition('unconverted data remains: ')[2])
            if ulr:
                start = datetime.datetime.strptime(self.data_stamp[s_begin][:-ulr], "%Y-%m-%d %H:%M:%S")
            else:
                raise ValueError
        #start = datetime.datetime.strptime(self.data_stamp[s_begin], "%Y-%m-%d %H:%M:%S")
        end = (start + datetime.timedelta(hours=self.token_len-1)).strftime("%Y-%m-%d %H:%M:%S")
        seq_x_mark = f"This is Time Series from {start} to {end}. The RCS is {self.data_rcs}."
        return seq_x_mark

    def __len__(self):
        return len(self.data_stamp)


