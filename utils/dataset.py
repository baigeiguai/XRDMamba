from tqdm import tqdm
from torch import multiprocessing 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch 



class XRDData(Dataset):
    def __init__(self,data_file_path:str,mode:str,dataset_type:str):
        data = np.load(data_file_path,allow_pickle=True,encoding='latin1')
        raw_features = data.item().get('features')
        raw_labels230 = data.item().get('labels230')
        angle_range = data.item().get('angle_range')
        if dataset_type =='generated':
            self.features = []
            self.labels230 = []
            self.atomic_number = []
            for i in range(len(raw_features)):
                for j in raw_features[i]:
                    self.features.append(j)
                    self.labels230.append(raw_labels230[i])
                    self.atomic_number.append(angle_range[i])
                    if mode == 'test':
                        break
        elif dataset_type == 'pymatgen':
            self.features = raw_features
            self.labels230 = raw_labels230
            self.atomic_number = angle_range

    def __getitem__(self, index) :
        return [self.features[index],self.atomic_number[index],self.labels230[index]]
    
    def __len__(self):
        return len(self.features)


if __name__ == '__main__':
    t = XRDData('/home/ylh/code/MyExps/MOF/data/Pymatgen_Wrapped/3/train_0.npy',"train",'pymatgen')    
    dataloader = DataLoader(t,16,True)
    for data in dataloader :
        x,y,i = data[0],data[1],data[2]
        print(x.shape)
        break
    