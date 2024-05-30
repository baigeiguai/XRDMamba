import os 
import numpy as np 

def data_path2file_paths(file_path:str):
    # dirs = os.listdir(os.path.join(file_path,'Wrapped'))
    # max_dir = str(max([int(d) for d in dirs]))
    # base_path = os.path.join(os.path.join(file_path,'Wrapped'),max_dir)
    file_paths = os.listdir(file_path)
    return [os.path.join(file_path,f)  for f in file_paths if f.startswith('train')],[os.path.join(file_path,f) for f in file_paths if f.startswith('test')]


def lengthen_xrd(source_x,source_y,to_length,angle_start,angle_end):
    insert_num = to_length-source_x.shape[0]
    if insert_num == 0 :
        return np.add(source_x.reshape(1,-1),source_y.reshape(1,-1),0) # type: ignore
    insert_x = np.linspace(angle_start,angle_end,insert_num)
    insert_y = np.zeros(insert_num)
    x = np.append(source_x,insert_x,0)
    y = np.append(source_y,insert_y,0)
    sort_idx = np.argsort(x,0)    
    return np.append(x[sort_idx].reshape(1,-1),y[sort_idx].reshape(1,-1),0)
    
def per_class_acc2sum_acc(class_cnt,class_acc,class_idx):
    acc_n,n = 0.0,0.0
    for i in class_idx:
        acc_n += class_cnt[i]*class_acc[i]  
        n+= class_cnt[i]
    return  acc_n/n

if  __name__ == '__main__':
    x = np.random.rand(5)
    y = np.random.rand(5)
    res = lengthen_xrd(x,y,10,0,1)
    print("res:",res,"res.shape",res.shape)
    
