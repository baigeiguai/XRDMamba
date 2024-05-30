from torch.utils.data import DataLoader
import json,os,numpy as np ,torch,argparse,logging,datetime
from utils.convert import data_path2file_paths
from utils.init import seed_torch
from utils.dataset import XRDData
from torch.utils.tensorboard.writer import SummaryWriter
from utils.log import print_info_before_epoch,print_info_after_epoch
from utils.writer import add_scalars
from torchmetrics.classification import MulticlassAccuracy
from models.XRDMamba import XRDMamba 

parser = argparse.ArgumentParser(description='主函数')
parser.add_argument('--data_path',type=str,required=True)
parser.add_argument('--model',type=str,required=True)
parser.add_argument('--encoder_path',type=str)
parser.add_argument('--input_channel',type=int,default=1)
parser.add_argument('--dataset',type=str,choices=['generated','pymatgen'],default='pymatgen')
parser.add_argument('--learning_rate',type=float,default=0.01)
parser.add_argument('--min_learning_rate',type=float,default=0.002)
parser.add_argument('--start_scheduler_step',type=int)
parser.add_argument('--weight_decay',type=float,default=1e-4)
parser.add_argument('--batch_size',type=int,default=256)
parser.add_argument('--class_number',type=int,default=230)
parser.add_argument('--epoch_number',type=int,default=200)
parser.add_argument('--model_save_path',type=str,required=True)
parser.add_argument('--device',type=str,default="2")
parser.add_argument("--p_dropout",type=float,default=0.2)
parser.add_argument("--scheduler_T",type=int)
parser.add_argument("--embeding_size",type=int,default=16)
parser.add_argument("--m_layers",type=int,default=8)
parser.add_argument("--m_d_state",type=int,default=32)
args = parser.parse_args()

logging.getLogger().setLevel(logging.DEBUG)
args.log_file_name = './loggs/%s_%s.log'%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),args.model)
logging.basicConfig(filename=args.log_file_name)

seed_torch(3407)

device_list = [int(i) for i in args.device.split(',')]
device = torch.device("cuda:%d"%device_list[0] if torch.cuda.is_available() else 'cpu')


if args.encoder_path is None:
    encoder = XRDMamba(embeding_size=args.embeding_size,p_dropout=args.p_dropout,m_layers=args.m_layer,m_d_state=args.m_d_state).to(device)
else :
    encoder = torch.load(args.encoder_path,map_location=device)

if len(device_list) > 1 :
    encoder = torch.nn.DataParallel(encoder,device_list).to(device)
    
lossfn = torch.nn.CrossEntropyLoss().to(device)

if not os.path.exists(args.model_save_path):
    os.mkdir(args.model_save_path)
with open(os.path.join(args.model_save_path,'config.json'),'w') as json_file:
    json_file.write(json.dumps(vars(args)))
model_save_path = os.path.join(args.model_save_path,args.model)
optimizer = torch.optim.AdamW(params=encoder.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
start_scheduler_step = args.epoch_number//2 if args.start_scheduler_step is None else args.start_scheduler_step
scheduler_T = args.epoch_number-start_scheduler_step if args.scheduler_T is None else args.scheduler_T
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,scheduler_T,args.min_learning_rate)
logging.info('start task,args: %s,device: %s,encoder: %s,loss:%s,optimizer:%s'%(str(args),str(device),encoder,str(lossfn),str(optimizer)))
train_files,test_files = data_path2file_paths(args.data_path)
writer = SummaryWriter(log_dir='./board_dir/train_%s_%s'%(args.model,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

def train():
    max_acc  = 0 
    mini_err = 1e9
    for epoch_idx in range(args.epoch_number):
        print_info_before_epoch(epoch_idx,lr_scheduler.get_lr())
        total_num = 0.0
        total_err = 0.0
        batch_cnt = 0
        for  file in train_files:
            xrd_dataset = XRDData(file,'train',args.dataset)
            dataloader = DataLoader(xrd_dataset,batch_size=args.batch_size,shuffle=True,num_workers=10)
            for data in dataloader:
                optimizer.zero_grad()
                intensity,angle_range,space_group = data[0],data[1],data[2]
                intensity = intensity.type(torch.float).to(device)
                space_group =  space_group.to(device)
                angle_range = angle_range.to(device)
                logits = encoder(intensity,angle_range)
                error = lossfn(logits,space_group)
                error.backward()
                optimizer.step()
                total_num += angle_range.shape[0]
                batch_cnt += 1 
                total_err += error.item()
        logging.info("[training]total_num:%s,total_err:%s"%(total_num,total_err/batch_cnt))
        test_acc,test_err = test()
        add_scalars(writer,epoch_idx,{"train/acc":test_acc,"train/err":test_err})    
        if epoch_idx >= start_scheduler_step:
            lr_scheduler.step()
        if mini_err > test_err:
            mini_err = test_err 
            max_acc = max(max_acc,test_acc)
            torch.save(encoder if not len(device_list)>1  else encoder.module,model_save_path+'_encoder_%d'%(epoch_idx+1)+'.pth')
        elif max_acc < test_acc :
            max_acc = test_acc
            torch.save(encoder if not len(device_list)>1  else encoder.module,model_save_path+'_encoder_%d'%(epoch_idx+1)+'.pth')
        print_info_after_epoch(epoch_idx)
        
def test():
    encoder.eval()
    total_acc = MulticlassAccuracy(args.class_number,average='micro').to(device)
    total_num = 0 
    total_err = 0.0 
    batch_cnt = 0 
    with torch.no_grad():
        for file in train_files:
            xrd_dataset = XRDData(file,'test',args.dataset)
            dataloader = DataLoader(xrd_dataset,args.batch_size)
            for data in dataloader:
                intensity,angle_range,space_group = data[0],data[1],data[2]
                angle_range = angle_range.to(device)
                intensity = intensity.type(torch.float).to(device)
                space_group = space_group.to(device)
                raw_logits = encoder(intensity,angle_range)
                err = lossfn(raw_logits,space_group)

                total_err += err.item()
                logits = raw_logits.softmax(dim=1)
                total_num += intensity.shape[0]
                total_acc(logits,space_group)
                batch_cnt += 1
    total_acc_val = total_acc.compute().cpu().item()
    logging.info('[testing]total_number:%d,error:%f,total_acc:%s'%(total_num,total_err/batch_cnt,total_acc_val))
    return total_acc_val,total_err/batch_cnt

train()
writer.close()




