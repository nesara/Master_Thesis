from collections import defaultdict

import numpy as np
import os
import time
import datetime

import rdkit
import torch
import torch.nn.functional as F
from pysmiles import read_smiles
from torch.autograd import Variable
from torchvision.utils import save_image

from util_dir.utils_io import random_string
from utils import *
from models_vae import Generator, Discriminator, EncoderVAE, PropertyMLP
from data.sparse_molecular_dataset import SparseMolecularDataset

from rdkit import Chem
from rdkit import DataStructs
import geomloss

# from simgnn import SimGNN
from train_simgnn import SimGNN
from param_parser import parameter_parser


class Solver1(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, config, log=None):
        """Initialize configurations."""

        # Log
        self.log = log

        # Data loader.
        self.data = SparseMolecularDataset()
        self.data.load(config.mol_data_dir)

        # Model configurations.
        self.z_dim = config.z_dim
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.f_dim = self.data.features
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.lambda_wgan = config.lambda_wgan
        self.lambda_rec = config.lambda_rec
        self.post_method = config.post_method

        self.metric = 'validity,qed'

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.num_steps = (len(self.data) // self.batch_size)
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.p_lr = config.p_lr
        self.dropout_rate = config.dropout
        self.n_critic = config.n_critic
        self.resume_epoch = config.resume_epoch

        # Training or testing.
        self.mode = config.mode

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device: ', self.device)

        # Directories.
        self.log_dir_path = config.log_dir_path
        self.model_dir_path = config.model_dir_path
        self.img_dir_path = config.img_dir_path

        # Step size.
        self.model_save_step = config.model_save_step

        # VAE KL weight.
        self.kl_la = 1.

        # Build the model.
        self.build_model()

    def build_model(self):
        """Create an encoder and a decoder."""
        self.encoder = EncoderVAE(self.d_conv_dim, self.m_dim, self.b_dim - 1, self.z_dim,
                                  with_features=True, f_dim=self.f_dim, dropout_rate=self.dropout_rate).to(self.device)
        self.decoder = Generator(self.g_conv_dim, self.z_dim, self.data.vertexes, self.data.bond_num_types,
                                 self.data.atom_num_types, self.dropout_rate).to(self.device)
        self.property_mlp = PropertyMLP().to(self.device)
        # self.V = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim - 1, self.dropout_rate).to(self.device)

        self.vae_optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                                 list(self.decoder.parameters()), self.d)
        self.property_optimizer = torch.optim.Adam(list(self.property_mlp.parameters()), self.p_lr)
        # self.v_optimizer = torch.optim.RMSprop(self.V.parameters(), self.d_lr)

        self.print_network(self.encoder, 'Encoder', self.log)
        self.print_network(self.decoder, 'Decoder', self.log)
        self.print_network(self.property_mlp, 'Property MLP', self.log)
        # self.print_network(self.V, 'Value', self.log)

    @staticmethod
    def print_network(model, name, log=None):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        if log is not None:
            log.info(model)
            log.info(name)
            log.info("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        print(self.model_dir_path)
        enc_path = os.path.join(self.model_dir_path, '{}-encoder.ckpt'.format(resume_iters))
        dec_path = os.path.join(self.model_dir_path, '{}-decoder.ckpt'.format(resume_iters))
        property_path = os.path.join(self.model_dir_path, '{}-property-mlp.ckpt'.format(resume_iters))
        # V_path = os.path.join(self.model_dir_path, '{}-V.ckpt'.format(resume_iters))
        self.encoder.load_state_dict(torch.load(enc_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(dec_path, map_location=lambda storage, loc: storage))
        self.property_mlp.load_state_dict(torch.load(property_path, map_location=lambda storage, loc: storage))
        # self.V.load_state_dict(torch.load(V_path, map_location=lambda storage, loc: storage))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.vae_optimizer.zero_grad()
        self.property_optimizer.zero_grad()
        # self.v_optimizer.zero_grad()

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size()) + [dim]).to(self.device)
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
        return out

    def sample_z(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    @staticmethod
    def postprocess_logits(inputs, method, temperature=1.):
        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]

    def reward(self, mols):
        rr = 1.
        for m in ('logp,sas,qed,unique' if self.metric == 'all' else self.metric).split(','):

            if m == 'np':
                rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            elif m == 'logp':
                rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
            elif m == 'sas':
                rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            elif m == 'qed':
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
            elif m == 'novelty':
                rr *= MolecularMetrics.novel_scores(mols, self.data)
            elif m == 'dc':
                rr *= MolecularMetrics.drugcandidate_scores(mols, self.data)
            elif m == 'unique':
                rr *= MolecularMetrics.unique_scores(mols)
            elif m == 'diversity':
                rr *= MolecularMetrics.diversity_scores(mols, self.data)
            elif m == 'validity':
                rr *= MolecularMetrics.valid_scores(mols)
            else:
                raise RuntimeError('{} is not defined as a metric'.format(m))

        return rr.reshape(-1, 1)
    
    def restore_tanimoto_model(self, ):
        # self.tanimoto_model = SimGNN(args, 5)
        # self.tanimoto_model.load_state_dict(torch.load(r"D:\Masters\Nesara Thesis\SimGNN\results\model_save_bk.pth"))
        self.tanimoto_model = SimGNN([[128, 64], 128, [128, 64]], self.m_dim, self.b_dim - 1, 16, with_features=False, dropout_rate=0.).to(self.device)
        self.tanimoto_model.load_state_dict(torch.load(r"results/morgan/sim_model_morgan.pth"))
        self.tanimoto_model.eval()
        for param in self.tanimoto_model.parameters():
            param.requires_grad = False


    def train_and_validate(self):
        self.start_time = time.time()

        # Start training from scratch or resume training.
        start_epoch = 0
        if self.resume_epoch:
            start_epoch = self.resume_epoch
            self.restore_model(self.resume_epoch)

        self.restore_tanimoto_model()

        # Start training.
        if self.mode == 'train':
            print('Start training...')
            for i in range(start_epoch, self.num_epochs):
                self.train_or_valid(epoch_i=i, train_val_test='train')
                self.train_or_valid(epoch_i=i, train_val_test='val')
                # self.train_or_valid(epoch_i=i, train_val_test='sample')
        elif self.mode == 'test':
            assert self.resume_epoch is not None
            # self.train_or_valid(epoch_i=start_epoch, train_val_test='sample')
            # self.train_or_valid(epoch_i=start_epoch, train_val_test='val')
            # self.test_property()
            self.gradient_ascent()
        else:
            raise NotImplementedError
    
    def test_property(self,):
        mols, _, _, a, x, _, f, _, _ = self.data.next_validation_batch()
        ## Water Octanol Partition Scores
        wopc = torch.from_numpy(MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)).to(self.device).float()
        print('Max WOPC:{}, Min WOPC:{}'.format(wopc.min(), wopc.max()))
        a = torch.from_numpy(a).to(self.device).long()  # Adjacency.
        x = torch.from_numpy(x).to(self.device).long()  # Nodes.
        a_tensor = self.label2onehot(a, self.b_dim)
        x_tensor = self.label2onehot(x, self.m_dim)
        f = torch.from_numpy(f).to(self.device).float()

        z, z_mu, z_logvar = self.encoder(a_tensor, f, x_tensor)
        wopc_pred = self.property_mlp(z)
        # print(wopc[:10])
        # print(wopc_pred[:10])

    def freeze_weights(self, no_freeze=True):
        for param in self.encoder.parameters():
            param.requires_grad = no_freeze
        for param in self.property_mlp.parameters():
            param.requires_grad = no_freeze
        for param in self.decoder.parameters():
            param.requires_grad = no_freeze

    def gradient_ascent(self,):
        mols, _, _, a, x, _, f, _, _ = self.data.next_validation_batch(batch_size=1)
        wopc = torch.from_numpy(MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=False)).to(self.device).float()

        ## Get Fingerprint
        mols_fp = Chem.RDKFingerprint(mols[0])

        
        mol_f_name = os.path.join(self.img_dir_path, 'orig-mol-{}.png'.format(str(wopc[0].data.cpu().numpy()).replace('.','!')))
        save_mol_img(mols, mol_f_name, is_test=self.mode == 'train')
        self.freeze_weights(no_freeze=False)

        a = torch.from_numpy(a).to(self.device).long()  # Adjacency.
        x = torch.from_numpy(x).to(self.device).long()  # Nodes.
        a_tensor = self.label2onehot(a, self.b_dim)
        x_tensor = self.label2onehot(x, self.m_dim)
        f = torch.from_numpy(f).to(self.device).float()

        z, z_mu, z_logvar = self.encoder(a_tensor, f, x_tensor)
        z.requires_grad = True
        # z.retain_grad()

        optim = torch.optim.SGD([z], lr=1e-3, momentum=0.9)
        
        for i in range(25):
            wopc_pred = self.property_mlp(z)
            loss =  (3.6 - wopc_pred.squeeze())
            # loss = torch.nn.MSELoss()(torch.tensor([3.61]).cuda(), wopc_pred)
            print('Step ', i,  wopc_pred.data, z)
            loss.backward()
            optim.step()
            edges_logits, nodes_logits = self.decoder(z)

            mols_pred = self.get_gen_mols(nodes_logits, edges_logits, 'hard_gumbel')
            if(mols_pred[0]):
                mols_pred_fp = Chem.RDKFingerprint(mols_pred[0])
                sim = DataStructs.FingerprintSimilarity(mols_fp, mols_pred_fp)
            else:
                sim = '#'
            wopc = torch.from_numpy(MolecularMetrics.water_octanol_partition_coefficient_scores(mols_pred, norm=False)).to(self.device).float()
            print('Actual Solubility: {}, Predicted Solubility:{}, Similarity '.format(wopc.data, wopc_pred.data, sim ))

            wopc_str = str(wopc[0].data.cpu().numpy()).replace('.','!')

            # Saving molecule images.
            mol_f_name = os.path.join(self.img_dir_path, 'sample-mol-{}--{}--[{}].png'.format(i, wopc_str, str(sim)[:4]))
            save_mol_img(mols_pred, mol_f_name, is_test=self.mode == 'train')
    
    def train_property():
        mols, _, _, a, x, _, f, _, _ = self.data.next_validation_batch(batch_size=1)




    def get_reconstruction_loss(self, n_hat, n, e_hat, e):
        # This loss cares about the imbalance between nodes and edges.
        # However, in practice, they don't work well.
        # n_loss = torch.nn.CrossEntropyLoss(reduction='none')(n_hat.view(-1, self.m_dim), n.view(-1))
        # n_loss_ = n_loss.view(n.shape)
        # e_loss = torch.nn.CrossEntropyLoss(reduction='none')(e_hat.reshape((-1, self.b_dim)), e.view(-1))
        # e_loss_ = e_loss.view(e.shape)
        # loss_ = e_loss_ + n_loss_.unsqueeze(-1)
        # reconstruction_loss = torch.mean(loss_)
        # return reconstruction_loss

        n_loss = torch.nn.CrossEntropyLoss(reduction='mean')(n_hat.view(-1, self.m_dim), n.view(-1))
        e_loss = torch.nn.CrossEntropyLoss(reduction='mean')(e_hat.reshape((-1, self.b_dim)), e.view(-1))
        reconstruction_loss = n_loss + e_loss
        return reconstruction_loss
    
    def sinkhorn_distance(self,n_hat, n, e_hat, e, epsilon=0.1, n_iters=100):

        loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=epsilon, scaling=0.99, reach=1.0, diameter=None)

        node_loss = loss(n, n_hat).mean()

        e_hat = e_hat.reshape(-1, e_hat.shape[-1])
        e = e.reshape(-1, e.shape[-1])

        loss1 = geomloss.SamplesLoss("sinkhorn", p=2, blur=epsilon, scaling=0.99, reach=1.0, diameter=None)

        edge_loss = loss1(e, e_hat)
        # print('Node Loss: {}, Edge Loss:{}'.format(node_loss, edge_loss))

       

        sinkhorn_loss= node_loss + edge_loss

        return sinkhorn_loss


    # def get_tanimoto_sim(self, n_hat, n, e_hat, e):
    #     mols_orig = self.get_gen_mols(n, e, 'hard_gumbel')
    #     mols_pred = self.get_gen_mols(n_hat, e_hat, 'hard_gumbel')

    def get_tanimoto_sim(self, e, n, e_hat, n_hat):
        # n_hat = torch.nn.functional.gumbel_softmax(n_hat, dim=-1, hard=True)
        # e_hat = torch.nn.functional.gumbel_softmax(e_hat, dim=-1, hard=True)

        # print(e.shape, e_hat.shape, n.shape,)
        out = self.tanimoto_model(e_hat, None, n_hat, e, None, n)

        tani_loss = torch.mean(out)

        return 1-tani_loss




    def get_tanimoto_sim_old(self, e, n, e_hat, n_hat):

        # print(e_hat[0].shape)
        # print(n_hat[0])
        # print(e.shape, n.shape, e_hat.shape, n_hat.shape)
        n_hat = torch.argmax(n_hat, dim=-1)
        e_hat = torch.argmax(e_hat, dim=-1)



        n = n.detach().cpu().numpy()
        n_hat = n_hat.detach().cpu().numpy()

        scores = []
        for id in range(e.shape[0]):
            a1 = np.transpose(np.nonzero(e[id].detach().cpu().numpy())).tolist()
            a2 = np.transpose(np.nonzero(e_hat[id].detach().cpu().numpy())).tolist()

            new_data = dict()

            edges_1 = torch.from_numpy(np.array(a1, dtype=np.int64).T).type(torch.long)
            edges_2 = torch.from_numpy(np.array(a2, dtype=np.int64).T).type(torch.long)

            features_1, features_2 = [], []
            global_labels = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
            

            for j in n[id].tolist():
                features_1.append([1.0 if global_labels[j] == i else 0.0 for i in global_labels.values()])

            for j in n_hat[id].tolist():
                features_2.append([1.0 if global_labels[j] == i else 0.0 for i in global_labels.values()])

            features_1 = torch.FloatTensor(np.array(features_1))
            features_2 = torch.FloatTensor(np.array(features_2))

            new_data["edge_index_1"] = edges_1
            new_data["edge_index_2"] = edges_1

            new_data["features_1"] = features_1
            new_data["features_2"] = features_1

            out = self.tanimoto_model(new_data)
            scores.append(out[0].item())
            # print(out[0].item())

        # print(scores)        

        # print(np.mean(scores))
        tani_loss = torch.tensor(1 - np.mean(scores), requires_grad=True)

        return tani_loss


        
        

    @staticmethod
    def get_kl_loss(mu, logvar):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        return kld_loss
    
    def get_mse_loss(self, w_hat, w):
        # print(w_hat.shape, type(w_hat), w.shape, type(w), 'W Shapes')
        ws_loss = torch.nn.MSELoss()(w_hat.squeeze(), w)
        return ws_loss

    def get_gen_mols(self, n_hat, e_hat, method):
        (edges_hard, nodes_hard) = self.postprocess_logits((e_hat, n_hat), method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]

        # ## If Training comment the following two lines
        if(self.mode == 'test'):
            edges_hard = torch.unsqueeze(edges_hard, 0)
            nodes_hard = torch.unsqueeze(nodes_hard, 0)
        
        mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]
        return mols

    def get_reward(self, n_hat, e_hat, method):
        (edges_hard, nodes_hard) = self.postprocess_logits((e_hat, n_hat), method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]
        reward = torch.from_numpy(self.reward(mols)).to(self.device)
        return reward

    def save_checkpoints(self, epoch_i):
        enc_path = os.path.join(self.model_dir_path, '{}-encoder.ckpt'.format(epoch_i + 1))
        dec_path = os.path.join(self.model_dir_path, '{}-decoder.ckpt'.format(epoch_i + 1))
        property_path = os.path.join(self.model_dir_path, '{}-property-mlp.ckpt'.format(epoch_i + 1))
        # V_path = os.path.join(self.model_dir_path, '{}-V.ckpt'.format(epoch_i + 1))
        torch.save(self.encoder.state_dict(), enc_path)
        torch.save(self.decoder.state_dict(), dec_path)
        torch.save(self.property_mlp.state_dict(), property_path)
        # torch.save(self.V.state_dict(), V_path)
        print('Saved model checkpoints into {}...'.format(self.model_dir_path))
        if self.log is not None:
            self.log.info('Saved model checkpoints into {}...'.format(self.model_dir_path))

    def get_scores(self, mols, to_print=False):
        scores = defaultdict(list)
        m0, m1 = all_scores(mols, self.data, norm=True)  # 'mols' is output of Fake Reward
        for k, v in m1.items():
            scores[k].append(v)
        for k, v in m0.items():
            scores[k].append(np.array(v)[np.nonzero(v)].mean())

        if to_print:
            log = ""
            is_first = True
            for tag, value in scores.items():
                if is_first:
                    log += "{}: {:.2f}".format(tag, np.mean(value))
                    is_first = False
                else:
                    log += ", {}: {:.2f}".format(tag, np.mean(value))
            print(log)
            return scores, log

        return scores

    def train_or_valid(self, epoch_i, train_val_test='val'):
        # Recordings
        losses = defaultdict(list)

        the_step = self.num_steps
        if train_val_test == 'val':
            if self.mode == 'train':
                the_step = 1
            print('[Validating]')

        if train_val_test == 'sample':
            if self.mode == 'train':
                the_step = 1
            print('[Sampling]')

        for a_step in range(the_step):
            if(a_step==1):
                break
                # print('Step: {} / {}'.format(a_step, the_step))
            z = None
            if train_val_test == 'val':
                mols, _, _, a, x, _, f, _, _ = self.data.next_validation_batch()
                ## Water Octanol Partition Scores
                wopc = torch.from_numpy(MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=False)).to(self.device).float()
                # print(mols.shape, a.shape, x.shape, f.shape, 'Mols, A, X, F Shapes')
            elif train_val_test == 'train':
                mols, _, _, a, x, _, f, _, _ = self.data.next_train_batch(self.batch_size)
                wopc = torch.from_numpy(MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=False)).to(self.device).float()
            elif train_val_test == 'sample':
                z = self.sample_z(self.batch_size)
                z = torch.from_numpy(z).to(self.device).float()
            else:
                raise NotImplementedError

            if train_val_test == 'train' or train_val_test == 'val':
                a = torch.from_numpy(a).to(self.device).long()  # Adjacency.
                x = torch.from_numpy(x).to(self.device).long()  # Nodes.
                a_tensor = self.label2onehot(a, self.b_dim)
                x_tensor = self.label2onehot(x, self.m_dim)
                f = torch.from_numpy(f).to(self.device).float()
                # print(a_tensor.shape, f.shape, x_tensor.shape, 'A, F, X Shapes')

            if train_val_test == 'train' or train_val_test == 'val':
                z, z_mu, z_logvar = self.encoder(a_tensor, f, x_tensor)
            wopc_hat = self.property_mlp(z)
            # print(z.shape,'Z Shape')
            edges_logits, nodes_logits = self.decoder(z)
            (edges_hat, nodes_hat) = self.postprocess_logits((edges_logits, nodes_logits), self.post_method)
            # (edges_hat, nodes_hat) = self.postprocess_logits((edges_logits, nodes_logits), "hard_gumbel")

            if train_val_test == 'train' or train_val_test == 'val':
                recon_loss = self.get_reconstruction_loss(nodes_logits, x, edges_logits, a)
                # sink_loss= self.sinkhorn_distance(nodes_logits, x_tensor, edges_logits, a_tensor)
                tanimoto_loss = self.get_tanimoto_sim(a_tensor, x_tensor, edges_logits, nodes_logits)

                kl_loss = self.get_kl_loss(z_mu, z_logvar)
                property_loss = self.get_mse_loss(wopc_hat, wopc)

                # print(wopc_hat[:10], wopc[:10])
                # loss_vae =  tanimoto_loss  + self.kl_la * kl_loss
                loss_vae =  tanimoto_loss

                # vae_loss_train = self.lambda_wgan * loss_vae + (1 - self.lambda_wgan) * loss_rl
                vae_loss_train =  loss_vae 
                # vae_loss_train = loss_vae * 0.5 + tanimoto_loss * 0.5
                losses['l_Rec'].append(recon_loss.item())
                # losses['l_Sink'].append(sink_loss.item())
                losses['l_KL'].append(kl_loss.item())
                losses['l_VAE'].append(loss_vae.item())
                losses['l_property'].append(property_loss.item())
                losses['l_tanimoto'].append(tanimoto_loss.item())
                print('Step: {}, Recon Loss:{} Tanimoto Loss:{}'.format(a_step, recon_loss, tanimoto_loss, property_loss))

                if train_val_test == 'train':
                    self.reset_grad()
                    vae_loss_train.backward(retain_graph=True)
                    self.vae_optimizer.step()

            if train_val_test == 'sample':
                mols_pred = self.get_gen_mols(nodes_logits, edges_logits, 'hard_gumbel')
                scores, mol_log = self.get_scores(mols_pred, to_print=True)

                # Saving molecule images.
                mol_f_name = os.path.join(self.img_dir_path, 'sample-mol-{}.png'.format(epoch_i))
                save_mol_img(mols_pred, mol_f_name, is_test=self.mode == 'test')

                if self.log is not None:
                    self.log.info(mol_log)

            if train_val_test == 'val':
                mols_pred = self.get_gen_mols(nodes_logits, edges_logits, 'hard_gumbel')
                print('Validation Size:', len(mols_pred))
                scores = self.get_scores(mols_pred)

                # Save checkpoints.
                if self.mode == 'train':
                    if (epoch_i + 1) % self.model_save_step == 0:
                        self.save_checkpoints(epoch_i=epoch_i)

                # Saving molecule images.
                mol_f_name = os.path.join(self.img_dir_path, 'mol-{}.png'.format(epoch_i))
                id = save_mol_img(mols_pred, mol_f_name, is_test=self.mode == 'test')
                print('Valid Id', id)
                ## Saving Original Molecule Image
                mol_f_name = os.path.join(self.img_dir_path, 'mol-orig-{}.png'.format(epoch_i))
                save_mol_img([mols[id]], mol_f_name, is_test=self.mode == 'test')                

                # Print out training information.
                et = time.time() - self.start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]:".format(et, epoch_i + 1, self.num_epochs)

                is_first = True
                for tag, value in losses.items():
                    if is_first:
                        log += "\n{}: {:.2f}".format(tag, np.mean(value))
                        is_first = False
                    else:
                        log += ", {}: {:.2f}".format(tag, np.mean(value))
                is_first = True
                for tag, value in scores.items():
                    if is_first:
                        log += "\n{}: {:.2f}".format(tag, np.mean(value))
                        is_first = False
                    else:
                        log += ", {}: {:.2f}".format(tag, np.mean(value))
                print(log)

                if self.log is not None:
                    self.log.info(log)


