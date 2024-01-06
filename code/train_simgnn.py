import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from layers import GraphConvolution, GraphAggregation, MultiGraphConvolutionLayers, MultiDenseLayer
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from rdkit.Chem import AllChem
from matplotlib import pyplot as plt
from focal_loss.focal_loss import FocalLoss
import pandas as pd
import json
import random
import matplotlib
from rdkit.Chem import MACCSkeys



class SimGNN(nn.Module):
    """VAE encoder sharing part."""
    def __init__(self, conv_dim, m_dim, b_dim, z_dim, with_features=False, f_dim=0, dropout_rate=0.):
        super(SimGNN, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, with_features, f_dim, dropout_rate)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1]+m_dim, aux_dim, torch.nn.Tanh(), with_features, f_dim,
                                          dropout_rate)
        self.multi_dense_layer = MultiDenseLayer(aux_dim, linear_dim, torch.nn.Tanh(), dropout_rate=dropout_rate)
        self.linear1 = nn.Linear(64, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, adj, hidden, node, adj2, hidden2, node2,  activation=None):
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h = self.gcn_layer(node, adj, None)
        h = self.agg_layer(h, node, None)
        h = self.multi_dense_layer(h)

        adj2 = adj2[:, :, :, 1:].permute(0, 3, 1, 2)
        h2 = self.gcn_layer(node2, adj2, None)
        h2 = self.agg_layer(h2, node2, None)
        h2 = self.multi_dense_layer(h2)

        h3 = h * h2

        h3 = self.linear1(h3)
        h3 = self.linear2(h3)
        h3 = self.linear3(h3)
        h3 = self.sigmoid(h3)


        return h3


class FPGNN(nn.Module):
    """VAE encoder sharing part."""
    def __init__(self, conv_dim, m_dim, b_dim, with_features=False, f_dim=0, dropout_rate=0.):
        super(FPGNN, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, with_features, f_dim, dropout_rate)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1]+m_dim, aux_dim, torch.nn.Tanh(), with_features, f_dim,
                                          dropout_rate)
        self.multi_dense_layer = MultiDenseLayer(aux_dim, linear_dim, torch.nn.Tanh(), dropout_rate=dropout_rate)
        # self.linear1 = nn.Linear(64, 128)        
        # self.act1 = nn.ReLU()
        # self.linear2 = nn.Linear(128, 256)
        # self.act2 = nn.ReLU()
        # self.linear3 = nn.Linear(256, 256)
        # self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(128, 256)        
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 256)
        # self.act2 = nn.ReLU()
        # self.linear3 = nn.Linear(256, 256)
        self.sigmoid = nn.Sigmoid()
        # self.emb_mean = nn.Linear(linear_dim[-1], z_dim)
        # self.emb_logvar = nn.Linear(linear_dim[-1], z_dim)

    # @staticmethod
    # def reparameterize(mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

    def forward(self, adj, hidden, node, activation=None):
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h = self.gcn_layer(node, adj, None)
        h = self.agg_layer(h, node, None)
        h = self.multi_dense_layer(h)

        # h3 = self.linear1(h)
        # h3 = self.act1(h3)
        # h3 = self.linear2(h3)
        # h3 = self.act2(h3)
        # h3 = self.linear3(h3)
        h3 = self.sigmoid(h)


        return h3


def valid(sim_model):

    mols, _, _, a, x, _, f, _, _ = data.next_validation_batch()
    half_batch = len(mols) // 2
    a = torch.from_numpy(a).to(device).long()  # Adjacency.
    x = torch.from_numpy(x).to(device).long()  # Nodes.
    a_tensor = label2onehot(a, b_dim)
    x_tensor = label2onehot(x, m_dim)
    f = torch.from_numpy(f).to(device).float()

    ## Shuffling the batch
    # idx = torch.randperm(x.size(0))
    # a_tensor = a_tensor[idx]
    # x_tensor = x_tensor[idx]
    # f = f[idx]
    # mols = mols[idx]

    out = sim_model(a_tensor[:half_batch], f[:half_batch], x_tensor[:half_batch], a_tensor[half_batch:], f[half_batch:], x_tensor[half_batch:])

    # fps = [Chem.RDKFingerprint(mol, fpSize=128) for mol in mols]
    # MACCS
    # fps = [MACCSkeys.GenMACCSKeys(mol, ) for mol in mols]
    # Morgan
    # fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) for mol in mols]
    # Topological Pairs
    fps = [AllChem.GetAtomPairFingerprint(mol) for mol in mols]

    tan_sim_li = []
    for i in range(half_batch):
        tan_sim = DataStructs.DiceSimilarity(fps[i],fps[i+half_batch])
        tan_sim = (tan_sim)
        tan_sim_li.append(tan_sim)
    

    # plt.hist(tan_sim_li)
    # plt.title('Validation Tanimoto Similarity Distribution')
    # plt.savefig('val_tansim_dist.png')
    orig = torch.Tensor(tan_sim_li).to(device)
    # loss = criterion(orig, out.squeeze())
    loss = criterion(orig, out.squeeze())
    out = out.squeeze()
    print('Original Scores: {}'.format(orig[:15]))
    print('=======================================')
    print('Predicted Scores: {}'.format(out[:15]))
    return loss.item()

    # out1 =  sim_model(a_tensor[:1], f[:1], x_tensor[:1], a_tensor[:1], f[:1], x_tensor[:1],)
    # out2 =  sim_model(a_tensor[1:2], f[1:2], x_tensor[1:2], a_tensor[1:2], f[1:2], x_tensor[1:2],)

    # print('Same Graphs Similarity G1 : {}'.format(out1.squeeze().detach().cpu().numpy()))
    # print('Same Graphs Similarity G2 : {}'.format(out2.squeeze().detach().cpu().numpy()))

    # print('Final MSE:{}'.format(loss.item()))
def valid_fp(sim_model):

    mols, _, _, a, x, _, f, _, _ = data.next_validation_batch()
    half_batch = len(mols) // 2
    a = torch.from_numpy(a).to(device).long()  # Adjacency.
    x = torch.from_numpy(x).to(device).long()  # Nodes.
    a_tensor = label2onehot(a, b_dim)
    x_tensor = label2onehot(x, m_dim)
    f = torch.from_numpy(f).to(device).float()

    ## Shuffling the batch
    # idx = torch.randperm(x.size(0))
    # a_tensor = a_tensor[idx]
    # x_tensor = x_tensor[idx]
    # f = f[idx]
    # mols = mols[idx]

    out = sim_model(a_tensor, f, x_tensor)

    fps = [Chem.RDKFingerprint(mol, fpSize=256) for mol in mols]
    # MACCS
    # fps = [MACCSkeys.GenMACCSKeys(mol, ) for mol in mols]
    # Morgan
    # fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) for mol in mols]
    # Topological Pairs
    # fps = [AllChem.GetAtomPairFingerprint(mol) for mol in mols]
    

    # plt.hist(tan_sim_li)
    # plt.title('Validation Tanimoto Similarity Distribution')
    # plt.savefig('val_tansim_dist.png')
    orig = torch.Tensor(fps).to(device)
    # loss = criterion(orig, out.squeeze())
    loss = criterion(orig, out.squeeze())
    out = out.squeeze()
    print('Original Scores: {}'.format(orig[:15]))
    print('=======================================')
    print('Predicted Scores: {}'.format(out[:15]))
    return loss.item()

    # out1 =  sim_model(a_tensor[:1], f[:1], x_tensor[:1], a_tensor[:1], f[:1], x_tensor[:1],)
    # out2 =  sim_model(a_tensor[1:2], f[1:2], x_tensor[1:2], a_tensor[1:2], f[1:2], x_tensor[1:2],)

    # print('Same Graphs Similarity G1 : {}'.format(out1.squeeze().detach().cpu().numpy()))
    # print('Same Graphs Similarity G2 : {}'.format(out2.squeeze().detach().cpu().numpy()))

    # print('Final MSE:{}'.format(loss.item()))

def get_distribution():
    mols, _, _, a, x, _, f, _, _ = data.next_train_batch()
    for fpsize in [128, 256, 512, 1024]:
        for i in range(1):
            tan_sim_li = []
            for j in range(i+1, len(mols)):
                fps_1 = Chem.RDKFingerprint(mols[i], fpSize=fpsize)
                fps_2 = Chem.RDKFingerprint(mols[j], fpSize=fpsize)
                tan_sim = DataStructs.TanimotoSimilarity(fps_1, fps_2)
                tan_sim_li.append(tan_sim)

                # tan_sim = int (round(tan_sim, 1) * 10)
                tan_sim_li.append(tan_sim)
            counts, bins = np.histogram(tan_sim_li)
            print((counts, bins))
            plt.hist(tan_sim_li)
            plt.title('Tanimoto Similarity Distribution using {} bit fingerprint'.format(fpsize))
            plt.savefig('results/train_tansim_dist_'+str(fpsize)+'_.png')
            plt.clf()
        # break            


def create_simpairs():
    mols, _, _, a, x, _, f, _, _ = data.next_train_batch()
    dic = {z: [] for z in range(1, 10)}
    print(dic)
    for i in range(100):
        tan_sim_li = []
        dic2 = {z: 0 for z in range(1, 10)}
        for j in range(i+1, len(mols)):
            fps_1 = Chem.RDKFingerprint(mols[i], fpSize=256)
            fps_2 = Chem.RDKFingerprint(mols[j], fpSize=256)
            tan_sim = DataStructs.TanimotoSimilarity(fps_1, fps_2)
            tan_sim_li.append(tan_sim)

            tan_sim = int (round(tan_sim, 1) * 10)

            if tan_sim in dic.keys():
                dic[tan_sim].append((i, j))
            # else:
            #     dic[tan_sim] = [(i, j)]
            
            if tan_sim in dic2.keys():
                dic2[tan_sim] += 1
            # else:
            #     dic2[tan_sim] = 1
            
            zero_count = list(dic2.values()).count(0)
            if(zero_count<3):
                break

            # flag = True
            # for v in dic2.values():
            #     if v < 10:
            #         flag = False
            #         break
            
            # if(flag):
            #     break
            
            # for k, v in dic.items():
            #     print('Key: {}, Length: {}'.format(k, len(v)))
        counts, bins = np.histogram(tan_sim_li)
        print((counts, bins))
        # plt.hist(tan_sim_li)
        # plt.title('Train Tanimoto Similarity Distribution')
        # plt.savefig('train_sample_tansim_dist.png')
        # break
    
    for k, v in dic.items():
        print('Key: {}, Length: {}'.format(k, len(v)))
    
    with open('graph_pairs_fp256.json', 'w') as f:
        f.write(json.dumps(dic))


import plotly.express as px 
from sklearn.metrics import confusion_matrix


class ArgMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        
        print('Calling Forward')
        idx = torch.argmax(input, 1)

        # output = torch.zeros_like(input)
        # output.scatter_(1, idx, 1)
        output = torch.nn.functional.one_hot(idx)
        ctx.save_for_backward(input)
        # output = input * 2

        return output
	

    @staticmethod
    def backward(ctx, grad_output):
        print('Called Backward')
        input = ctx.saved_tensors
        return grad_output * input
    
from data.sparse_molecular_dataset import SparseMolecularDataset
data = SparseMolecularDataset()
data.load(".\data\qm9_full.sparsedataset")
b_dim = data.bond_num_types

m_dim = data.atom_num_types
b_dim = data.bond_num_types
f_dim = 0
z_dim = 32
dropout_rate = 0.2
# d_conv_dim = [[128, 64], 128, [128, 64]]
d_conv_dim = [[128, 256, 512], 512, [512, 256]]
batch_size = 64
fpSize = 128
lr = 0.00001
epochs = 50



def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    out = torch.zeros(list(labels.size()) + [dim]).to(device)
    out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
    return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:{}'.format(device))

sim_model = FPGNN(d_conv_dim, m_dim, b_dim - 1, with_features=False, dropout_rate=dropout_rate).to(device)


def my_loss(output, target):
    loss = torch.mean((output - target)**3)
    return loss

criterion = nn.MSELoss()
# criterion = FocalLoss(gamma=2)

opt = torch.optim.Adam(sim_model.parameters(), lr=lr)

m = torch.nn.Softmax(dim=-1)

num_steps = (data.next_train_batch()[0].shape[0] // batch_size)

# fpgen = AllChem.GetRDKitFPGenerator()

def iou(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return (intersection / union).item()

def train_simgnn():
    best_l = 100
    train_losses, val_losses = [], []
    for e in range(epochs):
        print(e)
        tan_sim_orig = []
        tan_sim_pred = []
        max_len = 0
        for a_step in range(num_steps):
            mols, s, _, a, x, _, f, _, _ = data.next_train_batch(batch_size)
            max_l = max([len(s) for each in s])
            if(max_l>max_len):
                max_len = max_l
            half_batch = batch_size // 2
            a = torch.from_numpy(a).to(device).long()  # Adjacency.
            x = torch.from_numpy(x).to(device).long()  # Nodes.
            a_tensor = label2onehot(a, b_dim)
            x_tensor = label2onehot(x, m_dim)
            f = torch.from_numpy(f).to(device).float()

            ## Shuffling the batch
            idx = torch.randperm(x.size(0))
            a_tensor = a_tensor[idx]
            x_tensor = x_tensor[idx]
            f = f[idx]

            # rand_num = torch.rand(1)
            # if(rand_num<0.3):
            #     a_tensor[:half_batch] = a_tensor[half_batch:]
            #     x_tensor[:half_batch] = x_tensor[half_batch:]


            out = sim_model(a_tensor[:half_batch], f[:half_batch], x_tensor[:half_batch], a_tensor[half_batch:], f[half_batch:], x_tensor[half_batch:])

            # fps = [AllChem.GetMorganFingerprintAsBitVect(mol) for mol in mols]
            fps = [Chem.RDKFingerprint(mol, fpSize=128) for mol in mols]

            tan_sim_li = []
            for i in range(half_batch):
                tan_sim = DataStructs.TanimotoSimilarity(fps[i],fps[i+half_batch])
                tan_sim = (tan_sim)
                tan_sim_li.append(tan_sim)
            tan_sim_orig.extend(tan_sim_li)
            
            orig = torch.Tensor(tan_sim_li).to(device)
            # orig = torch.unsqueeze(orig, 1)
            # print(out.shape, orig.shape, type(orig))

            loss = criterion(orig, out.squeeze())
            # print(m(out))
            # print(orig)
            # loss = criterion(m(out), orig)
            # out = torch.argmax(m(out), dim=-1) 
            # tan_sim_pred.extend(out.squeeze().detach().cpu().numpy().tolist())


            # if(loss.item()< best_l):
            #     best_l = loss.item()
            #     torch.save(sim_model.state_dict(), 'results/sim_model_rdkf.pth')

            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # counts, bins = np.histogram(tan_sim_orig)
        # print('Max Smiles length: {}'.format(max_len))
        # print((counts, bins))
        # plt.hist(tan_sim_orig, range=(0, 11))
        # plt.title('Train Orig Tanimoto Similarity Distribution')
        # plt.savefig('train_orig_tansim_dist.png')
        # plt.close()

        # plt.hist(tan_sim_pred, range=(0, 11))
        # plt.title('Train Pred Tanimoto Similarity Distribution')
        # plt.savefig('train_pred_tansim_dist.png')
        # plt.close()

        val_loss = valid(sim_model)
        if(val_loss< best_l):
            best_l = val_loss
            torch.save(sim_model.state_dict(), 'results/sim_model_rdkf.pth')

        print('Epoch: {}, Train Loss:{}, Val Loss:{}'.format(e, loss.item(), val_loss))
        train_losses.append(loss.item())
        val_losses.append(val_loss)
    
    plt.plot(list(range(epochs)), train_losses, label='Train Loss')
    plt.plot(list(range(epochs)), val_losses, label='Val Loss')
    plt.savefig('results/loss_curves_rdkf.png')

def train_simgnn_2():
    with open('graph_pairs_rdkf.json','r') as f:
        data_pair = json.load(f)
    # min_len = min([len(val) for val in data_pair.values()])
    min_len = 2000
    mols, s, _, a, x, _, f, _, _ = data.next_train_batch()

    # data_pair['10'] = [[w, w] for w in range(min_len)]

    a_l, a_r = np.empty((0, 9, 9)), np.empty((0, 9, 9))
    x_l, x_r = np.empty((0, 9,)), np.empty((0, 9,))
    f_l, f_r = np.empty((0, 9, 54)), np.empty((0, 9, 54))
    mol_l, mol_r = np.empty((0, )), np.empty((0,))

    for k, v in data_pair.items():
        # new_v = random.sample(v, min_len)
        # left_id = [w[0] for w in new_v]
        # right_id = [w[1] for w in new_v]
        left_id = [w[0] for w in v[:min_len]]
        right_id = [w[1] for w in v[:min_len]]        

        a_l = np.append(a_l, a[left_id], 0)
        a_r = np.append(a_r, a[right_id], 0)

        x_l = np.append(x_l, x[left_id], 0)
        x_r = np.append(x_r, x[right_id], 0)

        f_l = np.append(f_l, f[left_id], 0)
        f_r = np.append(f_r, f[right_id], 0)

        mol_l = np.append(mol_l, mols[left_id], 0)
        mol_r = np.append(mol_r, mols[right_id], 0)

    ## Shuffling the batch
    idx = torch.randperm(len(a_l))
    a_l, a_r = a_l[idx], a_r[idx]
    x_l, x_r = x_l[idx], x_r[idx]
    f_l, f_r = f_l[idx], f_r[idx]
    mol_l, mol_r = mol_l[idx], mol_r[idx]  

    best_l = 100
    train_losses, val_losses = [], []
    batch_size = 64
    num_steps = len(a_l) // batch_size
    for e in range(epochs):
        tan_sim_orig = []
        tan_sim_pred = []
        max_len = 0
        counter = 0
        iter_loss = []
        for a_step in range(num_steps):
            # mols, s, _, a, x, _, f, _, _ = data.next_train_batch(batch_size)
            start, end = a_step*batch_size, (a_step+1)*batch_size
            a1 = torch.from_numpy(a_l[start:end]).to(device).long()  # Adjacency.
            x1 = torch.from_numpy(x_l[start:end]).to(device).long()  # Nodes.
            a_tensor1 = label2onehot(a1, b_dim)
            x_tensor1 = label2onehot(x1, m_dim)
            f1 = torch.from_numpy(f_l[start:end]).to(device).float()
            mols1 = mol_l[start:end]


            a2 = torch.from_numpy(a_r[start:end]).to(device).long()  # Adjacency.
            x2 = torch.from_numpy(x_r[start:end]).to(device).long()  # Nodes.
            a_tensor2 = label2onehot(a2, b_dim)
            x_tensor2 = label2onehot(x2, m_dim)
            f2 = torch.from_numpy(f_r[start:end]).to(device).float()
            mols2 = mol_r[start:end]



            out = sim_model(a_tensor1, f1, x_tensor1, a_tensor2, f2, x_tensor2)

            # fps = [AllChem.GetMorganFingerprintAsBitVect(mol) for mol in mols]
            # fps1 = [Chem.RDKFingerprint(mol, fpSize=128) for mol in mols1]
            # fps2 = [Chem.RDKFingerprint(mol, fpSize=128) for mol in mols2]
            # MACCS
            # fps1 = [MACCSkeys.GenMACCSKeys(mol, ) for mol in mols1]
            # fps2 = [MACCSkeys.GenMACCSKeys(mol, ) for mol in mols2]
            # Morgan
            # fps1 = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) for mol in mols1]
            # fps2 = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) for mol in mols2]
            # Topological Pairs
            # fps1 = [AllChem.GetTopologicalTorsionFingerprint(mol) for mol in mols1]
            # fps2 = [AllChem.GetTopologicalTorsionFingerprint(mol) for mol in mols2]
            # Atom Pairs
            fps1 = [AllChem.GetAtomPairFingerprint(mol) for mol in mols1]
            fps2 = [AllChem.GetAtomPairFingerprint(mol) for mol in mols2]            

            tan_sim_li = []
            for i in range(len(mols1)):
                tan_sim = DataStructs.DiceSimilarity(fps1[i], fps2[i])
                tan_sim = (tan_sim)
                tan_sim_li.append(tan_sim)
            tan_sim_orig.extend(tan_sim_li)
            
            orig = torch.Tensor(tan_sim_li).to(device)
            # orig = torch.unsqueeze(orig, 1)
            # print(out.shape, orig.shape, type(orig))

            loss = criterion(orig, out.squeeze())
            iter_loss.append(loss.item())
            # print(m(out))
            # print(orig)
            # loss = criterion(m(out), orig)
            # out = torch.argmax(m(out), dim=-1) 
            # tan_sim_pred.extend(out.squeeze().detach().cpu().numpy().tolist())


            # if(loss.item()< best_l):
            #     best_l = loss.item()
            #     torch.save(sim_model.state_dict(), 'results/sim_model_rdkf.pth')

            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # counts, bins = np.histogram(tan_sim_orig)
        # print('Max Smiles length: {}'.format(max_len))
        # print((counts, bins))
        # plt.hist(tan_sim_orig, range=(0, 11))
        # plt.title('Train Orig Tanimoto Similarity Distribution')
        # plt.savefig('train_orig_tansim_dist.png')
        # plt.close()

        # plt.hist(tan_sim_pred, range=(0, 11))
        # plt.title('Train Pred Tanimoto Similarity Distribution')
        # plt.savefig('train_pred_tansim_dist.png')
        # plt.close()
        train_loss = np.mean(iter_loss)
        val_loss = valid(sim_model)
        if(val_loss< best_l):
            best_l = val_loss
            torch.save(sim_model.state_dict(), 'results/test/sim_model_ap.pth')

        print('Epoch: {}, Train Loss:{}, Val Loss:{}, Best Val Loss:{}'.format(e, train_loss, val_loss, best_l))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    plt.plot(list(range(epochs)), train_losses, label='Train Loss')
    plt.plot(list(range(epochs)), val_losses, label='Val Loss')
    plt.legend()
    plt.title('Atom Pair Fingerprint')
    plt.xlabel('Epochs')
    plt.xticks(list(range(epochs))[0::5])
    plt.ylabel('Loss')
    plt.savefig('results/test/loss_curves_ap.png')

    print('Train Losses',train_losses)
    print('Val Losses', val_losses)

def train_simgnn_fp():
    best_l = 100
    train_losses, val_losses = [], []
    tani_sim = []
    for e in range(epochs):
        print(e)
        tan_sim_orig = []
        tan_sim_pred = []
        max_len = 0
        for a_step in range(num_steps):
            mols, s, _, a, x, _, f, _, _ = data.next_train_batch(batch_size)
            max_l = max([len(s) for each in s])
            if(max_l>max_len):
                max_len = max_l
            # half_batch = batch_size // 2
            a = torch.from_numpy(a).to(device).long()  # Adjacency.
            x = torch.from_numpy(x).to(device).long()  # Nodes.
            a_tensor = label2onehot(a, b_dim)
            x_tensor = label2onehot(x, m_dim)
            f = torch.from_numpy(f).to(device).float()


            out = sim_model(a_tensor, f, x_tensor)

            # fps = [AllChem.GetMorganFingerprintAsBitVect(mol) for mol in mols]
            fps = [Chem.RDKFingerprint(mol, fpSize=256) for mol in mols]
            # print(type(fps[0]))

            
            orig = torch.Tensor(fps).to(device)
            # orig = torch.unsqueeze(orig, 1)
            # print(out.shape, orig.shape, type(orig))
            # print(out[0], orig[0])
            
            loss = criterion(orig, out.squeeze())
            out = (out>0.5).float()
            # print(out.shape, orig.shape, type(orig))
            # convert tensor on gpu to numpy array on cpu
            # out = out.cpu().numpy()


            tan_sim_li = []
            for i in range(len(mols)):
                # print(type(out[i].tolist()), type(orig[i]))
                # print(out[i].tolist(), orig[i].tolist())
                tan_sim = iou(out[i], orig[i])
                tan_sim_li.append(tan_sim)
            # print(len(tan_sim_li), tan_sim_li[0], np.mean(tan_sim_li))
            tani_sim.append(np.mean(tan_sim_li))       

            # print(m(out))
            # print(orig)
            # loss = criterion(m(out), orig)
            # out = torch.argmax(m(out), dim=-1) 
            # tan_sim_pred.extend(out.squeeze().detach().cpu().numpy().tolist())


            # if(loss.item()< best_l):
            #     best_l = loss.item()
            #     torch.save(sim_model.state_dict(), 'results/sim_model_rdkf.pth')

            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # counts, bins = np.histogram(tan_sim_orig)
        # print('Max Smiles length: {}'.format(max_len))
        # print((counts, bins))
        # plt.hist(tan_sim_orig, range=(0, 11))
        # plt.title('Train Orig Tanimoto Similarity Distribution')
        # plt.savefig('train_orig_tansim_dist.png')
        # plt.close()

        # plt.hist(tan_sim_pred, range=(0, 11))
        # plt.title('Train Pred Tanimoto Similarity Distribution')
        # plt.savefig('train_pred_tansim_dist.png')
        # plt.close()

        val_loss = valid_fp(sim_model)
        if(val_loss< best_l):
            best_l = val_loss
            torch.save(sim_model.state_dict(), 'results/models/sim_model_fp.pth')

        print('Epoch: {}, Train Loss:{}, Val Loss:{}, TanSim: {}'.format(e, loss.item(), val_loss, tani_sim[-1]))
        train_losses.append(loss.item())
        val_losses.append(val_loss)
    
    plt.plot(list(range(epochs)), train_losses, label='Train Loss')
    plt.plot(list(range(epochs)), val_losses, label='Val Loss')
    plt.savefig('results/loss_curves_fp.png')

def convert_to_csv():
    data = SparseMolecularDataset()
    data.load("data\qm9_full.sparsedataset")
    mols, s1, _, a, x, _, f, _, _ = data.next_train_batch()
    s1 = s1.tolist()
    s1 = ['"'+str(s)+'"' for s in s1]
    wopc_train = ((MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=False))).tolist()
    print(len(wopc_train),'wopc train')
    print(len(s1), s1[:10],'S1')


    mols2, s2, _, a, x, _, f, _, _ = data.next_validation_batch()
    s2 = s2.tolist()
    s2 = ['"'+str(s)+'"' for s in s2]
    wopc_val = ((MolecularMetrics.water_octanol_partition_coefficient_scores(mols2, norm=False))).tolist()
    print(len(wopc_val),'wopc val')
    print(len(s2),'S2')

    mols3, s3, _, a, x, _, f, _, _ = data.next_test_batch()
    s3 = s3.tolist()
    s3 = ['"'+str(s)+'"' for s in s3]

    wopc_test = ((MolecularMetrics.water_octanol_partition_coefficient_scores(mols3, norm=False))).tolist()


    s1.extend(s2)
    s1.extend(s3)
    wopc_train.extend(wopc_val)
    wopc_train.extend(wopc_test)

    print(len(s1), len(wopc_train))
    df = pd.DataFrame({'smiles':s1, 'logP' :wopc_train})
    df.to_csv('data/qm9_full_logp.csv', index=None, )
    print(df.head(), df.shape)











