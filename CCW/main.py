import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import torch.optim as optim
import wandb
import os
import sys
import datetime
import time

import math
from Models import *

from utility.helper import *
from utility.batch_test import *


class Model_Wrapper(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = args.model_type
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.scc = args.scc
        
        self.mess_dropout = eval(args.mess_dropout)

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        
            
        if self.scc == 2 :
            self.norm_adj_list = data_config['norm_adj_list']
            self.idx_list = data_config['idx_list']
            self.incd_mat_list = data_config['incd_mat_list']
            
            self.s_norm_adj_list=[]
            for norm_adj in self.norm_adj_list:
                s_norm_adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj).float()
                self.s_norm_adj_list.append(s_norm_adj.cuda())

            if len(self.idx_list[0]) != len(self.idx_list[1]):
                print('Warning : different number of clusters between Users & Items')
            else:
                num_iter = len(self.idx_list[0]) # number of local-clusters
                self.u_dict_map = {i:[] for i in range(self.n_users)}
                self.i_dict_map = {i:[] for i in range(self.n_items)}
                for i in range(num_iter):
                    for j in self.idx_list[0][i]:
                        self.u_dict_map[j].append(i)
                    for l in self.idx_list[1][i]:
                        self.i_dict_map[l].append(i)
                
        else : 
            self.norm_adj_origin = data_config['norm_adj']
            
            self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj_origin).float()
            data_config['norm_adj'] = self.norm_adj
            # self.norm_adj_origin = self.norm_adj
            # print(type(self.norm_adj_origin))
            
            self.norm_adj = self.norm_adj.cuda()
        
            

        self.record_alphas = False

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

         # code for wandb    
        self.wandb = data_config['wandb']
        
        self.wandb_proj_name = data_config['wandb_proj_name']
        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        print('model_type is {}'.format(self.model_type))

        self.weights_save_path = '%sweights/%s/%s/l%s/r%s' % (args.weights_path, args.dataset, self.model_type,
                                                                 str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        """

        print('----self.alg_type is {}----'.format(self.alg_type))
        if self.scc == 2:
            self.model=UCR(self.s_norm_adj_list, self.incd_mat_list , self.idx_list, self.alg_type, self.emb_dim, self.weight_size, self.mess_dropout,data_config, args)
            
            
        else : 
            if self.alg_type in ['ngcf']:

                self.model = NGCF(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout)
            elif self.alg_type in ['mf']:
                self.model = MF(self.n_users, self.n_items, self.emb_dim)
            elif self.alg_type in ['lightgcn']:
                self.model = LightGCN(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout)
            elif self.alg_type in ['dgcf']:
                self.model = DGCF(self.n_users, self.n_items, data_config, args)
            elif self.alg_type in ['ultragcn']:
                self.model = UltraGCN(self.n_users, self.n_items, data_config, args, data_config['incd_mat'], 0,0)
                self.lr = self.model.lr
                self.batch_size = self.model.batch_size
            elif self.alg_type in ['gcmc','GCMC']:
                 self.model = GCMC(self.n_users, self.n_items, self.emb_dim)
            elif self.alg_type in ['scf','SCF']:
                 self.model = SCF(self.n_users, self.n_items, self.emb_dim)
            elif self.alg_type in ['cgmc','CGMC']:
                 self.model = CGMC(self.n_users, self.n_items, self.emb_dim)
            elif self.alg_type in ['sgnn','SGNN']:
                 self.model = SGNN(self.n_users, self.n_items, self.emb_dim)
            else:
                raise Exception('Dont know which model to train')
            
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler()
        

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def save_model(self):
        ensureDir(self.weights_save_path)
        torch.save(self.model.state_dict(), self.weights_save_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.weights_save_path))
    
    def test(self, users_to_test, drop_flag=False, batch_test_flag=False):
        self.model.eval()
        with torch.no_grad():
            if self.alg_type in ['dgcf'] :
                ua_embeddings, ia_embeddings = self.model(self.norm_adj_origin)
#             if self.alg_type in ['ultragcn'] :
#                 if self.scc == 2 :
#                     ua_embeddings, ia_embeddings = self.model(self.s_norm_adj_list)
#                 else : 
#                 ua_embeddings, ia_embeddings = self.model(self.norm_adj)
                
            else : 
                if self.scc == 2:
                    ua_embeddings, ia_embeddings = self.model(self.s_norm_adj_list)

                else : 
                    ua_embeddings, ia_embeddings = self.model(self.norm_adj)

            result = test_torch(ua_embeddings, ia_embeddings, users_to_test)
        
        return result

    def map_idx2clu(self, users, pos_items, neg_items): # idx.type = list
        clu_users = np.asarray([self.u_dict_map[i] for i in users])
        clu_pos_items = np.asarray([self.i_dict_map[i] for i in pos_items])
        clu_neg_items = np.asarray([self.i_dict_map[i] for i in neg_items])
        # cluster numbers of indices
        return clu_users, clu_pos_items, clu_neg_items # np.array

    def train(self):
        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        stopping_step = 0
        should_stop = False
        cur_best_pre_0 = 0.

        n_batch = data_generator.n_train // self.batch_size + 1
        n_users = data_generator.n_users
        print(f"Number of Users : {n_users}")
        print(f"Number of Intereractions : {data_generator.n_train}")
        if self.wandb:
            if args.scc == 2:
                
                args.cl_num=0
                
            self.name = '_'.join([str(args.alg_type), str(args.embed_size), str(self.batch_size), str(args.regs), 'lr'+str(args.lr), 'scc'+str(args.scc), 'k'+str(args.N), 'n'+str(args.cl_num)])
            run=wandb.init(project=self.wandb_proj_name+'_final',entity='ncia-gnn',name=self.name)

            wandb.config.update = {                
                   'embed_size':args.embed_size,
                   'batch_size':self.batch_size,
                   "regs": args.regs,
                   'lr':args.lr,
                   'scc':args.scc,
                   'N':args.N,
                   'cl_num':args.cl_num,
                   'alg_type':args.alg_type,
                
            }


        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            n_batch = data_generator.n_train // self.batch_size + 1
            f_time, b_time, loss_time, opt_time, clip_time, emb_time = 0., 0., 0., 0., 0., 0.
            sample_time = 0.
            cuda_time = 0.
            # for DGCF
            cor_batch_size = int(max(data_generator.n_users/n_batch, data_generator.n_items/n_batch))
            
            for idx in range(n_batch):
                self.model.train()
                self.optimizer.zero_grad()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()
                
                #for DGCF
                cor_users, cor_items = data_generator.sample_cor_samples(data_generator.n_users, data_generator.n_items, cor_batch_size)
                
                sample_time += time() - sample_t1
                
                
                batch_cor_loss = 0.0
                
                if self.alg_type in ['dgcf']:
                    # batch_loss, batch_mf_loss, batch_emb_loss, batch_cor_loss = self.model(users, pos_items, neg_items,cor_users,cor_items)
                    # u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, cor_u_g_embeddings, cor_i_g_embeddings = self.model(users, pos_items, neg_items,cor_users,cor_items)
                    ua_embeddings, ia_embeddings = self.model(self.norm_adj_origin)
                    u_g_embeddings = ua_embeddings[users]
                    pos_i_g_embeddings = ia_embeddings[pos_items]
                    neg_i_g_embeddings = ia_embeddings[neg_items]
                    cor_u_g_embeddings = ua_embeddings[cor_users]
                    cor_i_g_embeddings = ia_embeddings[cor_items]
                    if args.corDecay < 1e-9:
                        batch_cor_loss = torch.zeros(1).cuda()
                    else:
                        batch_cor_loss = args.corDecay *  self.cor_loss(cor_u_g_embeddings, cor_i_g_embeddings)
                    batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                                  neg_i_g_embeddings)

                    batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss + batch_cor_loss
                    
                elif self.alg_type in ['ultragcn']:
                    if self.scc == 2:
                        ua_embeddings, ia_embeddings = self.model(self.s_norm_adj_list)
                    else : 
                        ua_embeddings, ia_embeddings = self.model(self.norm_adj)
                    users, pos_items, neg_items = data_generator.ultra_sampling((users, pos_items), data_generator.n_items, self.model.negative_num, data_generator.train_items, sampling_sift_pos=self.model.sampling_sift_pos)
                    omega_weight = self.get_omegas(torch.tensor(users), torch.tensor(pos_items), torch.tensor(neg_items))
                    u_g_embeddings = ua_embeddings[users]
                    pos_i_g_embeddings = ia_embeddings[pos_items]
                    neg_i_g_embeddings = ia_embeddings[neg_items]
                    batch_loss = self.cal_loss_L(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings,omega_weight)
                    batch_loss += self.model.gamma * self.norm_loss()
                    batch_loss += self.model.lambda_ * self.cal_loss_I(users, pos_items, u_g_embeddings, ia_embeddings)
                    
        
                else : # mf, ngcf, lightgcn
                    if self.scc == 2:
                        ua_embeddings, ia_embeddings = self.model(self.s_norm_adj_list)
                        u_g_embeddings = ua_embeddings[users]
                        pos_i_g_embeddings = ia_embeddings[pos_items]
                        neg_i_g_embeddings = ia_embeddings[neg_items]
#                         batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss_origin(u_g_embeddings, pos_i_g_embeddings,
#                                                                                       neg_i_g_embeddings)
                        #--------------------------#
                        clu_users, clu_pos_items, clu_neg_items = self.map_idx2clu(users=users, pos_items=pos_items, neg_items=neg_items)
                        check_u_pi = torch.tensor((clu_users == clu_pos_items).astype('int')).float().squeeze().cuda() # todo : without gradient
                        check_u_ni = torch.tensor((clu_users == clu_neg_items).astype('int')).float().squeeze().cuda()
                        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                                      neg_i_g_embeddings, check_u_pi=check_u_pi, check_u_ni=check_u_ni)

                    else : 
                        ua_embeddings, ia_embeddings = self.model(self.norm_adj)
                        u_g_embeddings = ua_embeddings[users]
                        pos_i_g_embeddings = ia_embeddings[pos_items]
                        neg_i_g_embeddings = ia_embeddings[neg_items]
                        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss_origin(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
                    

                    batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss + batch_cor_loss
                
                if self.alg_type in ['dgcf']:
                    batch_loss.backward(retain_graph=True)
                else :                     
                    batch_loss.backward()                
                
                self.optimizer.step()
                if self.alg_type in ['ultragcn']:
                    batch_loss = batch_loss / self.batch_size
                    loss += float(batch_loss)    
                else : 
                    loss += float(batch_loss)
                    mf_loss += float(batch_mf_loss)
                    emb_loss += float(batch_emb_loss)

                    try : 
                        batch_reg_loss = batch_cor_loss
                    except : 
                        pass
                    reg_loss += float(batch_reg_loss)
                


            self.lr_scheduler.step()
            # self.optimizer.step()
            try : 
                del ua_embeddings, ia_embeddings, u_g_embeddings, neg_i_g_embeddings, pos_i_g_embeddings
            except :
                pass


            if math.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()

            # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
            if (epoch + 1) % 10 != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                        epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss)
                    training_time_list.append(time() - t1)
                    print(perf_str)
                continue

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            
            ret = self.test(users_to_test, drop_flag=True)
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])
            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0],
                            ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)
                if self.wandb:
                    wandb.log({'train_loss':loss,
                               'mf_loss':mf_loss,
                               'emb_loss':emb_loss,
                               'reg_loss':reg_loss,
                               'recall' : ret['recall'][0], 
                               'precision' : ret['precision'][0], 
                               'hit_ratio' : ret['hit_ratio'][0], 
                               'ndcg' : ret['ndcg'][0], 

                              })
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=20)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop:
                break

            # *********************************************************
            # save the user & item embeddings for pretraining.
            if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
                # save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
                self.save_model()
                if self.record_alphas:
                    self.best_alphas = [i for i in self.model.get_alphas()]
                print('save the weights in path: ', self.weights_save_path)

        if rec_loger != []:
            self.print_final_results(rec_loger, pre_loger, ndcg_loger, hit_loger, training_time_list,n_users)

    def print_final_results(self, rec_loger, pre_loger, ndcg_loger, hit_loger, training_time_list,n_users):
        recs = np.array(rec_loger)
        pres = np.array(pre_loger)
        ndcgs = np.array(ndcg_loger)
        hit = np.array(hit_loger)
        sparsity = (data_generator.n_train + data_generator.n_test)/(data_generator.n_users * data_generator.n_items)
        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)

        final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                     (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                      '\t'.join(['%.5f' % r for r in pres[idx]]),
                      '\t'.join(['%.5f' % r for r in hit[idx]]),
                      '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
        print(final_perf)
        if self.wandb:
            wandb.run.summary["best_recall"] = round(recs[idx][0],5)
            wandb.run.summary["best_ndcg"] = round(ndcgs[idx][0],5)
            wandb.run.summary["best_precision"] = round(pres[idx][0],5)
            wandb.run.summary["best_hit"] = round(hit[idx][0],5)
        # Benchmarking: time consuming
        avg_time = sum(training_time_list) / len(training_time_list)
        time_consume = "Benchmarking time consuming: average {}s per epoch".format(avg_time)
        print(time_consume)

        results_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, self.model_type)
        result_summary_path = '%soutput/summary.result' % (args.proj_path)
        ensureDir(results_path)
        f = open(results_path, 'a')
        
        f.write(
            'datetime: %s\nembed_size=%d, lr=%.5f, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n\t%s\n\n'
            % (datetime.datetime.now(), args.embed_size, args.lr, args.mess_dropout, args.regs,
               args.adj_type, final_perf, time_consume))
              
        f.close()
        ensureDir(results_path) 
        
        f2 = open(result_summary_path, 'a')
        f3 = open(result_summary_path, 'r')
        text = f3.readline()
        a = int(time())
        if len(text)==0:
            f2.write('|{:^12}|{:^13}|{:^10}|{:^14}|{:^10}|{:^10}|{:^12}|{:^14}|{:^10}|{:^10}|{:^10}|{:^10}|{:^12}|{:^12}|{:^14}|{:^10}|\n'.format('Time','Dataset','Model','batch_size','embed','lr','# Cluster','Cluster idx.','# Users','# train', '# test','sparsity','Recall@20','NDCG@20','Precision@20','hit@20'))
        f2.write(f"|{int(time()):^12}|{args.dataset:^13}|{self.alg_type:^10}|{self.batch_size:^14}|{args.embed_size:^10}|{args.lr:^10}|{args.N:^12}|{args.cl_num:^14}|{n_users:^10}|{data_generator.n_train:^10}|{data_generator.n_test:^10}|{sparsity:^10.5f}|{recs[idx][0]:^12.5f}|{ndcgs[idx][0]:^12.5f}|{pres[idx][0]:^14.5f}|{hit[idx][0]:^10.5f}|\n")              
        f2.close()


    def bpr_loss_origin(self, users, pos_items, neg_items): # old version
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss
    def bpr_loss(self, users, pos_items, neg_items, check_u_pi, check_u_ni): # 0429 version, todo : without gradient for some tensors
        '''
        new bpr loss, check whether in-cluster or out-cluster
        '''
        # todo : 'masking zeros' or 'use global embeddings as local embeddings for out-cluster pairs'
        checking_u_pi = torch.cat([check_u_pi.unsqueeze(dim=0).T for _ in range(int(users.shape[1]/2))], dim = 1)
        masking_pi = torch.cat((torch.ones_like(checking_u_pi).cuda(), checking_u_pi), dim = 1)
        checking_u_ni = torch.cat([check_u_ni.unsqueeze(dim=0).T for _ in range(int(users.shape[1]/2))], dim = 1)
        masking_ni = torch.cat((torch.ones_like(checking_u_ni).cuda(), checking_u_ni), dim = 1)
        pos_scores = torch.sum(torch.mul(torch.mul(users, masking_pi), torch.mul(pos_items, masking_pi)), dim=1)
        neg_scores = torch.sum(torch.mul(torch.mul(users, masking_ni), torch.mul(neg_items, masking_ni)), dim=1)
        #--------------------#
        # todo : regularizer for masked embedding? or non-masked(now) embedding?
        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def   sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_sparse_tensor_value(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape

    def cor_loss(self, cor_u_embeddings, cor_i_embeddings):
        cor_loss = torch.zeros(1).cuda()

        if args.cor_flag == 0:
            return cor_loss
        
        ui_embeddings = torch.cat([cor_u_embeddings, cor_i_embeddings],0)
        split_factor = int(args.embed_size/args.n_factors)
        ui_factor_embeddings = torch.split(ui_embeddings, split_factor, 1)
        
        for i in range(0, args.n_factors-1):
            x = ui_factor_embeddings[i].cuda()
            y = ui_factor_embeddings[i+1].cuda()
            cor_loss += self._create_distance_correlation(x, y)

        cor_loss /= ((args.n_factors + 1.0) * args.n_factors/2)

        return cor_loss
    
    def _create_distance_correlation(self, X1, X2):

        def _create_centered_distance(X):
            '''
                Used to calculate the distance matrix of N samples
            '''
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            
            X = X.cuda()
            r = torch.sum(torch.square(X),1,keepdims=True)
            D = torch.sqrt(torch.maximum(r - 2 * torch.matmul(X,X.t()) + r.t(), torch.tensor(0.0)) + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            D = D - torch.mean(D,dim=0,keepdims=True)-torch.mean(D,dim=1,keepdims=True) \
                + torch.mean(D)
            return D
        
        def _create_distance_covariance(D1,D2):
            #calculate distance covariance between D1 and D2
            
            n_samples =int(D1.shape[0])            
            dcov = torch.sqrt(torch.maximum(torch.sum(D1 * D2) / (n_samples * n_samples), torch.tensor(0.0)) + 1e-8)
            
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)
        dcov_12 = _create_distance_covariance(D1,D2)
        dcov_11 = _create_distance_covariance(D1,D1)
        dcov_22 = _create_distance_covariance(D2,D2)

        #calculate the distance correlation
        dcor = dcov_12 / (torch.sqrt(torch.maximum(dcov_11 * dcov_22, torch.tensor(0.0))) + 1e-10)

        return dcor
    
    def get_omegas(self, users, pos_items, neg_items):
        device = self.model.get_device()
        if self.model.w2 > 0:
            pos_weight = torch.mul(self.model.constraint_mat['beta_uD'][users], self.model.constraint_mat['beta_iD'][pos_items]).to(device)
            pow_weight = self.model.w1 + self.model.w2 * pos_weight
        else:
            pos_weight = self.model.w1 * torch.ones(len(pos_items)).to(device)
        
        # users = (users * self.item_num).unsqueeze(0)
        # print(neg_items.shape)
        # print(neg_items.size(1))
        # print(neg_items.flatten())
        if self.model.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.model.constraint_mat['beta_uD'][users], neg_items.size(1)), self.model.constraint_mat['beta_iD'][neg_items.flatten()]).to(device)
            neg_weight = self.model.w3 + self.model.w4 * neg_weight
        else:
            neg_weight = self.model.w3 * self.model.ones(neg_items.size(0) * neg_items.size(1)).to(device)


        weight = torch.cat((pow_weight, neg_weight))
        return weight


    def cal_loss_L(self, user_embeds, pos_embeds, neg_embeds, omega_weight):
        device = self.model.get_device()
        
      
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight = omega_weight[len(pos_scores):].view(neg_scores.size()), reduction='none').mean(dim = -1)
        
        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight = omega_weight[:len(pos_scores)], reduction='none')

        loss = pos_loss + neg_loss * self.model.negative_weight
      
        return loss.sum()



    def cal_loss_I(self, users, pos_items, user_embeds, item_embeds):
        device = self.model.get_device()
        # neighbor_embeds = self.model.item_embeds(self.model.ii_neighbor_mat[pos_items].to(device))    # len(pos_items) * num_neighbors * dim
        neighbor_embeds = item_embeds[self.model.ii_neighbor_mat[pos_items].to(device)]    # len(pos_items) * num_neighbors * dim
        sim_scores = self.model.ii_constraint_mat[pos_items].to(device)     # len(pos_items) * num_neighbors
        user_embeds = user_embeds.unsqueeze(1)
        
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
      
        # loss = loss.sum(-1)
        return loss.sum()

    def norm_loss(self):
        loss = 0.0
        for parameter in self.model.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2
    

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    config = dict()
    

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    
#     plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()  ## original     

    if args.scc == 2:
        norm_adj_list, incd_mat_list, idx_list = data_generator.get_adj_mat(scc=args.scc, N=args.N, coclust = args.coclust)
    elif args.scc == 1 : 
        norm_adj = data_generator.get_adj_mat(scc=args.scc, N=args.N, cl_num = args.cl_num)
    else : 
        plain_adj, norm_adj, mean_adj, incd_mat= data_generator.get_adj_mat(scc=args.scc, N=args.N, cl_num = args.cl_num)  ## clustered sample
    
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['wandb']=args.wandb
    config['wandb_proj_name']=args.dataset
    if args.scc==2 : 
        config['norm_adj_list'] = norm_adj_list
        config['idx_list'] = idx_list
        config['incd_mat_list'] = incd_mat_list
    else :     
        if args.adj_type == 'norm':

            config['norm_adj'] = norm_adj
            config['incd_mat'] = incd_mat
            print('use the normalized adjacency matrix')
        else:
            config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
            config['incd_mat'] = incd_mat
            print('use the mean adjacency matrix')
        # if args.scc == 1 :
        #     config['incd_mat'] = incd_mat
        #     print('use the incidence matrix')
        
    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    Engine = Model_Wrapper(data_config=config, pretrain_data=pretrain_data)
    if args.pretrain:
        print('pretrain path: ', Engine.weights_save_path)
        if os.path.exists(Engine.weights_save_path):
            Engine.load_model()
            users_to_test = list(data_generator.test_set.keys())
            ret = Engine.test(users_to_test, drop_flag=True)
            cur_best_pre_0 = ret['recall'][0]

            pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                           'ndcg=[%.5f, %.5f]' % \
                           (ret['recall'][0], ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1],
                            ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
            print(pretrain_ret)
        else:
            print('Cannot load pretrained model. Start training from stratch')
    else:
        print('without pretraining')
    Engine.train()