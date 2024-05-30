import torch 

class ResBlock1D(torch.nn.Module):
    def __init__(self,in_channel,out_channel) -> None:
        super(ResBlock1D,self).__init__()
        self.pre = torch.nn.Identity() if in_channel == out_channel else torch.nn.Conv1d(in_channel,out_channel,1,bias=False)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(out_channel,out_channel,3,1,1,bias=False),
            torch.nn.BatchNorm1d(out_channel),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(out_channel,out_channel,3,1,1,bias=False),
            torch.nn.BatchNorm1d(out_channel),
        )
        self.relu = torch.nn.LeakyReLU()

    def forward(self,x):
        x = self.pre(x)
        out = self.conv(x)
        return self.relu(x+out)


class ResConv(torch.nn.Module):
    def __init__(self,in_c,p_dropout):
        super(ResConv,self).__init__()
        # self.down_size = torch.nn.MaxPool1d(10,10)
        # self.down_size = torch.nn.Conv1d(1,1,10,10,0)
        self.in_c = in_c
        self.conv = torch.nn.Sequential(

            ResBlock1D(in_c,32),
            torch.nn.MaxPool1d(2,2),

            ResBlock1D(32,32),
            torch.nn.MaxPool1d(2,2),

            ResBlock1D(32,64),
            torch.nn.MaxPool1d(2,2),

            ResBlock1D(64,64),
            torch.nn.AvgPool1d(2,2,1),
            
            ResBlock1D(64,128),
            torch.nn.AvgPool1d(2,2,1),
            
            ResBlock1D(128,128),
            torch.nn.AvgPool1d(2,2,1),

            ResBlock1D(128,256),
            torch.nn.AvgPool1d(2,2,1),

            ResBlock1D(256,256),
            torch.nn.AvgPool1d(2,2),
            torch.nn.Dropout(p_dropout),
            
            ResBlock1D(256,256),
            torch.nn.AvgPool1d(2,2),
            torch.nn.Dropout(p_dropout),

            ResBlock1D(256,512),
            torch.nn.AvgPool1d(2,2),
            torch.nn.Dropout(p_dropout),
            
            ResBlock1D(512,512),
            torch.nn.AvgPool1d(2,2,1),
            torch.nn.Dropout(p_dropout),

            ResBlock1D(512,1024),
            torch.nn.AvgPool1d(2,2,1),
            torch.nn.Dropout(p_dropout),

            ResBlock1D(1024,1024),
            torch.nn.AvgPool1d(2,2),
            # torch.nn.Dropout(p_dropout),

        )
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1024,512),
            torch.nn.Linear(512,230),
        )

    def forward(self,x,idx=None):
        x = x.view(x.shape[0],self.in_c,-1)
        x = self.conv(x)
        x = x.view(x.shape[0],-1)
        x = self.cls(x)
        return x 


if __name__ == '__main__':
    inp = torch.rand(128,1,5000)
    model = ResConv(1,0.15)
    out = model(inp)
    print(out.shape)