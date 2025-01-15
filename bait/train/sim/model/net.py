import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self, backbone):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        flatten = self.convnet(x).flatten(start_dim=1)
        output = flatten.view(flatten.size()[0], -1)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(nn.Module):
    def __init__(self, backbone):
        super(EmbeddingNetL2, self).__init__()
        self.convnet = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        output = self.convnet(x).flatten(start_dim=1)
        backbone = output.view(output.size()[0], -1)
        return F.normalize(backbone, p=2, dim=1)

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
