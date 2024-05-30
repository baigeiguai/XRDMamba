import torch 
from models.ResConv import ResConv
from models.Mamba import MambaConfig,Mamba


class XRDMamba(torch.nn.Module):
    def __init__(self,embeding_size=8,p_dropout=0.15,embed_nums=5000,m_layers=4,m_d_state=16,m_expand_factor=2,m_d_conv=3) -> None:
        super(XRDMamba,self).__init__()
        self.embed = torch.nn.Embedding(embed_nums,embeding_size)
        self.conv = ResConv(embeding_size,p_dropout)
        config = MambaConfig(d_model=embeding_size,n_layers=m_layers,d_state=m_d_state,expand_factor=m_expand_factor,d_conv=m_d_conv)
        self.mamba_embed = Mamba(config)
        
    def forward(self,x,idx):
        x = x.view(x.shape[0],-1,1)
        embedding = self.embed(idx.repeat(x.shape[0],1))
        features = embedding*x
        features = self.mamba_embed(features)
        features = features.transpose(1,2)
        cls = self.conv(features)
        return cls
    
    
if __name__ == '__main__':
    device = torch.device("cuda:4")
    idx = torch.tensor(list(range(5000)),requires_grad=False).to(device)
    inp = torch.rand(128,1,5000).to(device)
    model = XRDMamba(embeding_size=16).to(device)
    out = model(inp,idx)
    print(out.shape)
    