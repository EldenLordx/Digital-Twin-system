import torch
from torch.autograd import Variable
import torch.nn.functional as F
from CNet import CNet


class NumRec():
    def init(self):
        # model = torch.load('./model.pkl')
        self.model = CNet().cuda()
        self.model.load_state_dict(torch.load('./num.pkl'))
        self.model.eval()

    def rec(self, img):
        img = torch.from_numpy(img)
        # img = Variable(torch.unsqueeze(torch.unsqueeze(img, dim=0), dim=0).float(), requires_grad=False).cuda()
        img = Variable(torch.unsqueeze(torch.unsqueeze(img, dim=0), dim=0).float(), requires_grad=False).cuda()
        output = self.model(img)
        num = torch.argmax(F.softmax(output, dim=1)).item()

        return num
