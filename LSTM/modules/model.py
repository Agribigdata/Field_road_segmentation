from torch import nn
import torch.nn.functional as F
import os
import torch
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch_geometric.nn import GCNConv, SAGEConv, GATConv,TAGConv,GENConv,GINConv,GatedGraphConv,GINEConv,SGConv,GCNConv

DEVICE = "cuda:1"
class GCN_NET(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(GCN_NET, self).__init__()
        self.gcn1 = GCNConv(feature, hidden)
        self.gcn2 = GCNConv(hidden, classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.to(DEVICE)
        x = x.to(DEVICE)
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gcn2(x, edge_index)
        return F.log_softmax(x, dim=1)
class GraphSAGE_NET(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(GraphSAGE_NET, self).__init__()
        self.sage1 = SAGEConv(feature, hidden)  # 定义两层GraphSAGE层
        self.sage2 = SAGEConv(hidden, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.to(DEVICE)
        x = x.to(DEVICE)
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)
        return F.log_softmax(x, dim=1)
class GAT_NET(torch.nn.Module):
    def __init__(self, feature, hidden, classes, heads=4):
        super(GAT_NET, self).__init__()
        self.gat1 = GATConv(feature, hidden, heads=4)  # 定义GAT层，使用多头注意力机制
        self.gat2 = GATConv(hidden*heads, classes)  # 因为多头注意力是将向量拼接，所以维度乘以头数。

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.to(DEVICE)
        x = x.to(DEVICE)
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1)
class TAG_NET(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(TAG_NET, self).__init__()
        self.tag1 = TAGConv(feature, hidden)  # 定义GAT层，使用多头注意力机制
        self.tag2 = TAGConv(hidden, classes)  # 因为多头注意力是将向量拼接，所以维度乘以头数。
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.to(DEVICE)
        x = x.to(DEVICE)
        x = self.tag1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.tag2(x, edge_index)
        return F.log_softmax(x, dim=1)
class GEN_NET(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(GEN_NET, self).__init__()
        self.GEN1 = GENConv(feature, hidden)  # 定义GAT层，使用多头注意力机制
        self.GEN2 = GENConv(hidden, classes)  # 因为多头注意力是将向量拼接，所以维度乘以头数。

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.to(DEVICE)
        x = x.to(DEVICE)
        x = self.GEN1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.GEN2(x, edge_index)

        return F.log_softmax(x, dim=1)
class GIN_NET(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(GIN_NET, self).__init__()
        self.GIN1 = GINConv(feature, hidden)  # 定义GAT层，使用多头注意力机制
        self.GIN2 = GINConv(hidden, classes)  # 因为多头注意力是将向量拼接，所以维度乘以头数。
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.to(DEVICE)
        x = x.to(DEVICE)
        x = self.GIN1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.GIN2(x, edge_index)
        return F.log_softmax(x, dim=1)
class GatedGraph_NET(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(GatedGraph_NET, self).__init__()
        self.GatedGraph1 = GatedGraphConv(feature, hidden)  # 定义GAT层，使用多头注意力机制
        self.GatedGraph2 = GatedGraphConv(hidden, classes)  # 因为多头注意力是将向量拼接，所以维度乘以头数。
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.to(DEVICE)
        x = x.to(DEVICE)
        x = self.GatedGraph1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.GatedGraph2(x, edge_index)
        return F.log_softmax(x, dim=1)
class SG_NET(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(SG_NET, self).__init__()
        self.SG1 = SGConv(feature, hidden)  # 定义GAT层，使用多头注意力机制
        self.SG2 = SGConv(hidden, classes)  # 因为多头注意力是将向量拼接，所以维度乘以头数。
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.to(DEVICE)
        x = x.to(DEVICE)
        x = self.SG1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.SG2(x, edge_index)
        return F.log_softmax(x, dim=1)
class GINE_NET(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(GINE_NET, self).__init__()
        self.GINE1 = GINEConv(feature, hidden)  # 定义GAT层，使用多头注意力机制
        self.GINE2 = GINEConv(hidden, classes)  # 因为多头注意力是将向量拼接，所以维度乘以头数。
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.to(DEVICE)
        x = x.to(DEVICE)
        x = self.GINE1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.GINE2(x, edge_index)
        return F.log_softmax(x, dim=1)
class Transformer(nn.Module):
    def __init__(self, squ_len,dim_model=8,num_head=8,num_classes=2,dim_feedforward=2048,dropout=0.3,activation='relu'):
        super(Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(dim_model, num_head, dim_feedforward, dropout, activation)
        #self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout, config.device)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=6)
        self.fc1 = nn.Linear(dim_model, num_classes)

    def forward(self,x):
        print(x.shape)
        out = self.encoder(x)
        out = self.fc1(out)
        out = out.view(-1, 2)
        return out
class Transformer_TAG_NET(nn.Module):
    def __init__(self,hidden,classes,dim_model=8,num_head=8,num_classes=2,dim_feedforward=2048,dropout=0.3,activation='relu'):
        super(Transformer_TAG_NET, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(dim_model, num_head, dim_feedforward, dropout, activation)
        #self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout, config.device)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=6)
        #self.fc1 = nn.Linear(dim_model, num_classes)
        self.tag1 = TAGConv(dim_model, hidden)  # 定义GAT层，使用多头注意力机制
        self.tag2 = TAGConv(hidden, classes)  # 因为多头注意力是将向量拼接，所以维度乘以头数。
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.to(DEVICE)
        x = x.unsqueeze(dim=0)
        print(x.shape)
        x = x.to(DEVICE)
        x = self.encoder(x)
        x = self.tag1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.tag2(x, edge_index)
        #print(x.shape)
        x=x.squeeze(dim=0)
        return F.log_softmax(x, dim=1)
class GRU(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers,target_size=2,  drop_out=0.3):
        super(GRU, self).__init__()
        self.gru=nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=drop_out if num_layers > 1 else 0,
            bidirectional=True
        )


        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(hidden_size * 2, target_size)
    def forward(self,x,labels=None):
        x, hidden = self.gru(x)  # [batch, interval, hidden_size*2],  x只支持3维输入
        x = self.dropout(x)
        logits = self.fc(x)  # [batch, interval, 2]
        outputs = (logits,)
        # softmax
        loss_fct = nn.CrossEntropyLoss()
        loss_mask = labels.gt(-1)  # 筛选出有效标签做loss, labels.gt(-1)
        active_loss = loss_mask.view(-1) == 1
        active_labels = labels.view(-1)[active_loss]  # [batch*max_batch_len]
        active_logits = logits.view(-1, 2)[active_loss]  # [batch*max_batch_len, 2]
        loss = loss_fct(active_logits, active_labels)
        outputs = (loss,) + outputs
        return outputs  # logits, outputs
class LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, target_size=2, drop_out=0.3):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size = embedding_size,
            hidden_size = hidden_size,
            batch_first = True,
            num_layers = num_layers,
            dropout = drop_out if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(hidden_size*2, target_size)          #hidden_size * 12

    def forward(self, x, labels=None):              #x.shape = [batch, interval], labels.shape = [batch, interval]
        # print(x.shape)
        x, (hidden, cell) = self.lstm(x)          #[batch, interval, hidden_size*2],  x只支持3维输入
        # print(x.shape)
        x = self.dropout(x)
        logits = self.fc(x)                         #[batch, interval, 2]
        # print(logits.shape)
        output = (logits,)
        #softmax
        loss_fct = nn.CrossEntropyLoss()
        loss_mask = labels.gt(-1)
        active_loss = loss_mask.view(-1) == 1
        active_labels = labels.view(-1)[active_loss]          #[batch*max_batch_len]
        active_logits = logits.view(-1, 2)[active_loss]       #[batch*max_batch_len, 2]
        loss = loss_fct(active_logits, active_labels)
        outputs = (loss,) + output
        return outputs                       #logits, outputs
class single_LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, target_size=2, drop_out=0.3):
        super(single_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size = embedding_size,
            hidden_size = hidden_size,
            batch_first = True,
            num_layers = num_layers,
            dropout = drop_out if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(hidden_size, target_size)          #hidden_size * 12

    def forward(self, x, labels=None):              #x.shape = [batch, interval], labels.shape = [batch, interval]
        # print(x.shape)
        x, (hidden, cell) = self.lstm(x)          #[batch, interval, hidden_size*2],  x只支持3维输入
        # print(x.shape)
        x = self.dropout(x)
        logits = self.fc(x)                         #[batch, interval, 2]
        # print(logits.shape)
        output = (logits,)
        #softmax
        loss_fct = nn.CrossEntropyLoss()
        loss_mask = labels.gt(-1)
        active_loss = loss_mask.view(-1) == 1
        active_labels = labels.view(-1)[active_loss]          #[batch*max_batch_len]
        active_logits = logits.view(-1, 2)[active_loss]       #[batch*max_batch_len, 2]
        loss = loss_fct(active_logits, active_labels)
        outputs = (loss,) + output
        return outputs
class linear(nn.Module):
    def __init__(self, embedding_size, target_size=2, drop_out=0.3):
        super(linear, self).__init__()
        self.fc = nn.Linear(embedding_size, target_size)          #hidden_size * 12
    def forward(self, x, labels=None):              #x.shape = [batch, interval], labels.shape = [batch, interval]
        logits = self.fc(x)                         #[batch, interval, 2]
        outputs = (logits,)
        #softmax
        loss_fct = nn.CrossEntropyLoss()
        loss_mask = labels.gt(-1)
        active_loss = loss_mask.view(-1) == 1
        active_labels = labels.view(-1)[active_loss]          #[batch*max_batch_len]
        active_logits = logits.view(-1, 2)[active_loss]       #[batch*max_batch_len, 2]
        loss = loss_fct(active_logits, active_labels.long())
        outputs = (loss,) + outputs
        return outputs                       #logits, outputs
class LSTM_GCN_NET(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, target_size=2, drop_out=0.3):
        super(LSTM_GCN_NET, self).__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=drop_out if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(drop_out)
        #self.fc = nn.Linear(hidden_size * 2, target_size)  # hidden_size * 12
        self.tag1 = GCNConv(hidden_size * 2, hidden_size * 4)  # 定义GAT层，使用多头注意力机制
        self.tag2 = GCNConv(hidden_size * 4, target_size)  # 因为多头注意力是将向量拼接，所以维度乘以头数。
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.to(DEVICE)
        x = x.unsqueeze(dim=0)
        x = x.to(DEVICE)
        x, (hidden, cell) = self.lstm(x)  # [batch, interval, hidden_size*2],  x只支持3维输入
        x = self.tag1(x, edge_index)
        #x = F.relu(x)
        x = self.dropout(x)
        x = self.tag2(x, edge_index)
        x=x.squeeze(dim=0)
        return F.log_softmax(x, dim=1)
class LSTM_TAG_NET(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, target_size=2, drop_out=0.3):
        super(LSTM_TAG_NET, self).__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=drop_out if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(drop_out)
        #self.fc = nn.Linear(hidden_size * 2, target_size)  # hidden_size * 12
        self.tag1 = TAGConv(hidden_size * 2, hidden_size * 4)  # 定义GAT层，使用多头注意力机制
        self.tag2 = TAGConv(hidden_size * 4, target_size)  # 因为多头注意力是将向量拼接，所以维度乘以头数。
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.to(DEVICE)
        x = x.unsqueeze(dim=0)
        x = x.to(DEVICE)
        x, (hidden, cell) = self.lstm(x)  # [batch, interval, hidden_size*2],  x只支持3维输入
        x = self.tag1(x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.dropout(x)
        x = self.tag2(x, edge_index)
        #print(x.shape)
        x=x.squeeze(dim=0)
        #return F.log_softmax(x, dim=1)
        return F.log_softmax(x, dim=1)
class TAG_LSTM_NET(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, target_size=2, drop_out=0.3):
        super(TAG_LSTM_NET, self).__init__()
        self.tag1 = TAGConv(embedding_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size*2,
            batch_first=True,
            num_layers=num_layers,
            dropout=drop_out if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(hidden_size * 4, target_size)  # hidden_size * 12

    def forward(self, data, labels=None):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.to(DEVICE)
        x = x.to(DEVICE)
        x = self.tag1(x, edge_index)
        #x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = x.unsqueeze(dim=0)
        x, (hidden, cell) = self.lstm(x)
        logits = self.fc(x)  # [batch, interval, 2]
        logits = logits.squeeze(dim=0)
        return F.log_softmax(logits, dim=1)
class SelfAttention(nn.Module):

    def __init__(self, input_dim, att_type='general'):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.att_type = att_type
        self.scalar = nn.Linear(self.input_dim, 1, bias=True)

    def forward(self, M, x=None):
        """
        now M -> (batch, seq_len, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        if self.att_type == 'general':
            scale = self.scalar(M)  # seq_len, batch, 1
            #            scale = torch.tanh(scale)
            alpha = F.softmax(scale, dim=0).permute(0, 2, 1)  # batch, 1, seq_len
            attn_pool = torch.bmm(alpha, M)[:, 0, :]  # batch, vector/input_dim
        if self.att_type == 'general2':
            scale = self.scalar(M)  # seq_len, batch, 1
            #            scale = F.tanh(scale)
            alpha = F.softmax(scale, dim=0).permute(0, 2, 1)  # batch, 1, seq_len
            #            print ('alpha',alpha.size())
            att_vec_bag = []
            for i in range(M.size()[1]):
                alp = alpha[:, :, i]
                #                print ('alp',alp.size())
                vec = M[:, i, :]
                #                print ('vec',vec.size())
                alp = alp.repeat(1, self.input_dim)
                #                print ('alp',alp.size())
                att_vec = torch.mul(alp, vec)  # batch, vector/input_dim
                att_vec = att_vec + vec
                #                att_vec = torch.bmm(alp, vec)[:,0,:] # batch, vector/input_dim
                att_vec_bag.append(att_vec)
            attn_pool = torch.cat(att_vec_bag, -1)

        return attn_pool, alpha
