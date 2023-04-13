
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import numpy as np
from main import *
    
#Universial Clustering Recommendation
class UCR(nn.Module): 
    
    def __init__(self, s_norm_adj_list, incd_mat_list , idx_list , alg_type, embedding_dim, weight_size, dropout_list,data_config, args):
        super().__init__()
        self.s_norm_adj_list = s_norm_adj_list
        self.incd_mat_list = incd_mat_list
        
        self.idx_list = idx_list
        self.alg_type = alg_type
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_layers = len(self.weight_size)
        self.dropout_list = dropout_list
        self.final_weight_dim = embedding_dim        
        for dim in self.weight_size:
            self.final_weight_dim+=dim
        
            
        self.model_list = nn.ModuleList()
        self.num_model = len(self.s_norm_adj_list) # clustered graph + full graph ex) 3 small cluster + 1 full graph = 4
        self.local_user_embeddings = []
        self.local_item_embeddings = []
        self.attention_1 = nn.ModuleList()        
        self.attention_2 = nn.ModuleList()
        self.attention_1_u = nn.ModuleList()        
        self.attention_2_u = nn.ModuleList()
        self.attention_1_i = nn.ModuleList()        
        self.attention_2_i = nn.ModuleList()
        
        for i in range(self.num_model):
            n_users,n_items = self.incd_mat_list[i].shape[0], self.incd_mat_list[i].shape[1]
            if self.alg_type in ['ngcf' ,'NGCF']:
                self.model_list.append(NGCF(n_users, n_items, self.embedding_dim, self.weight_size, self.dropout_list))
            elif self.alg_type in ['mf' ,'MF']:
                
                self.model_list.append(MF(n_users, n_items, self.embedding_dim))
                self.final_weight_dim = self.embedding_dim
                
            elif self.alg_type in ['lightgcn' ,'LightGCN']:
                self.model_list.append(LightGCN(n_users, n_items, self.embedding_dim, self.weight_size, self.dropout_list))
                self.final_weight_dim = self.embedding_dim
                
            elif self.alg_type in ['dgcf' ,'DGCF']:
                self.model_list.append(DGCF(n_users, n_items, data_config, args))
            elif self.alg_type in ['ultragcn','UltraGCN']:
                self.model_list.append(UltraGCN(n_users, n_items, data_config, args, self.incd_mat_list[i], i,self.num_model)) 
                self.final_weight_dim = self.embedding_dim
                ultra_config = ultra_config_dict(args.dataset)
                self.w1=ultra_config['w1']
                self.w2=ultra_config['w2']
                self.w3=ultra_config['w3']
                self.w4=ultra_config['w4']
                self.negative_weight = ultra_config['negative_weight']
                self.negative_num=ultra_config['negative_num']
                self.gamma =ultra_config['gamma']
                self.lambda_ = ultra_config['lambda_']
                self.initial_weight = ultra_config['initial_weight']
                self.ii_neighbor_num = ultra_config['ii_neighbor_num']
                self.sampling_sift_pos=ultra_config['sampling_sift_pos']
                self.lr=ultra_config['lr']
                self.batch_size=ultra_config['batch_size']
                self.constraint_mat = self.model_list[0].constraint_mat
                self.ii_constraint_mat= self.model_list[0].ii_constraint_mat
                self.ii_neighbor_mat= self.model_list[0].ii_neighbor_mat
                # self.embedding_dim = ultra_config['embedding_dim']
            elif self.alg_type in ['gcmc','GCMC']:
                self.model_list.append(GCMC(n_users, n_items, self.embedding_dim))
                self.final_weight_dim = self.embedding_dim*4
            elif self.alg_type in ['scf','SCF']:
                self.model_list.append(SCF(n_users, n_items, self.embedding_dim))
                self.final_weight_dim = self.embedding_dim*4
            elif self.alg_type in ['cgmc','CGMC']:
                self.model_list.append(CGMC(n_users, n_items, self.embedding_dim))
                self.final_weight_dim = self.embedding_dim*4
            elif self.alg_type in ['sgnn','SGNN']:
                self.model_list.append(SGNN(n_users, n_items, self.embedding_dim))
                self.final_weight_dim = self.embedding_dim*4
                
            if i>=1:             
                with torch.no_grad():
                    self.local_user_embeddings.append(torch.zeros((self.incd_mat_list[0].shape[0], self.final_weight_dim),requires_grad = True,device='cuda').cuda())
                    self.local_item_embeddings.append(torch.zeros((self.incd_mat_list[0].shape[1], self.final_weight_dim),requires_grad = True,device='cuda').cuda())
                local_user = self.incd_mat_list[i].shape[0]
                local_item = self.incd_mat_list[i].shape[1]

                self.attention_1.append(nn.Linear(self.final_weight_dim*2 , 1024))
                self.attention_2.append(nn.Linear(1024 , 1))

    def forward(self, s_norm_adj_list):
        
        user_embed_list = []
        item_embed_list = []
        for i in range(0,self.num_model):
            
            u_g_embeddings, i_g_embeddings = self.model_list[i](s_norm_adj_list[i])
            user_embed_list.append(u_g_embeddings)
            item_embed_list.append(i_g_embeddings)
         # full graph
        user_embd = user_embed_list[0]
        item_embd = item_embed_list[0]


        # Ver 2
        for i in range(1,self.num_model):

            linear_u = self.attention_1[i-1]( torch.cat([user_embed_list[i], user_embd[self.idx_list[0][i-1]]],dim=1))
            alpha_u = self.attention_2[i-1](F.relu(linear_u))
            
            linear_l = self.attention_1[i-1]( torch.cat([item_embed_list[i], item_embd[self.idx_list[1][i-1]]],dim=1))
            alpha_l = self.attention_2[i-1](F.relu(linear_l))

            with torch.no_grad():
                self.local_user_embeddings[i-1][self.idx_list[0][i-1]]=alpha_u * user_embed_list[i]
                self.local_item_embeddings[i-1][self.idx_list[1][i-1]]=alpha_l * item_embed_list[i]
                
        local_user_embd = torch.sum(torch.stack(self.local_user_embeddings, dim=2),dim=2)
        local_item_embd = torch.sum(torch.stack(self.local_item_embeddings, dim=2),dim=2)
        final_u = torch.cat((user_embd, local_user_embd), dim = 1)
        final_i = torch.cat((item_embd, local_item_embd), dim = 1)
        #---concat global and local embedding, but this is not final embedding exactly
        return final_u, final_i 
        
        

class NGCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_layers = len(self.weight_size)
        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()

        self.weight_size = [self.embedding_dim] + self.weight_size
        for i in range(self.n_layers):
            self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.dropout_list.append(nn.Dropout(dropout_list[i]))

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self._init_weight_()
        
    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj):
        
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for i in range(self.n_layers):
            
            side_embeddings = torch.matmul(adj, ego_embeddings)
            # side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            
            sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))

            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = self.dropout_list[i](ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        return u_g_embeddings, i_g_embeddings

class SGNN(nn.Module):
    def __init__(self,  n_users, n_items, emb_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = emb_dim
        self.prop_dim = 128
        self.n_layers = 3
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        self.user_prop = nn.Embedding(n_users, self.prop_dim) ## Assume SGNN_RM model
        self.item_prop = nn.Embedding(n_items, self.prop_dim)
        self.norm = 1/(self.n_users+self.n_items)
        self._init_weight_()
        self.layer_weight = [1 / (l + 1) ** 1 for l in range(self.n_layers + 1)]
        
    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.user_prop.weight)
        nn.init.xavier_uniform_(self.item_prop.weight)

    def forward(self, adj):
        
        embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        propagation_rm = torch.cat((self.user_prop.weight, self.item_prop.weight),dim=0)
        
        all_embeddings = [embeddings]
        
        for i in range(self.n_layers):
            embeddings = torch.matmul(torch.transpose(propagation_rm,1,0), embeddings)
            embeddings = torch.matmul(propagation_rm,embeddings)
            all_embeddings += [self.layer_weight[i + 1] * F.tanh(self.norm * embeddings)]
            
        all_embeddings = torch.cat(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        return u_g_embeddings, i_g_embeddings


        
class GCMC(nn.Module):
    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim        
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        self.filters = nn.ModuleList() 
        self.dropout_list = nn.ModuleList() 
        for i in range(3):
            self.filters.append(nn.Embedding(emb_dim, emb_dim))
            self.dropout_list.append(nn.Dropout(0.3))
        self._init_weight_()
        
    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for i in range(3):
            nn.init.eye_(self.filters[i].weight)
#         self.filter += torch.Tensor(np.random.normal(0, 0.001, (self.emb_dim, self.emb_dim)) + np.diag(np.random.normal(1, 0.001, self.emb_dim))).cuda()

    def forward(self, adj):
        embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_emb = [embeddings]
        for i in range(3):
            embeddings = torch.sparse.mm(adj, embeddings)
            embeddings = F.relu(torch.matmul(embeddings, self.filters[i].weight))
            embeddings = self.dropout_list[i](embeddings)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_emb += [embeddings]
        all_emb = torch.sum(torch.stack(all_emb, dim=2),dim=2)
   
    
        users, items = torch.split(all_emb, [self.n_users, self.n_items])
        
        return users, items       
    
     
class SCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim        
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        self.filter = nn.Embedding(emb_dim, emb_dim)
        # SCF : LAYER 1 LR 0.0001 

        self._init_weight_()
        
    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.eye_(self.filter.weight)

    def forward(self, adj):
        embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_emb = [embeddings]
        
        embeddings = 2 * embeddings -  torch.sparse.mm(adj, embeddings)

        embeddings = F.sigmoid(torch.matmul(embeddings, self.filter.weight))
        all_emb += [embeddings]

        all_emb = torch.cat(all_emb, dim=1)
    
        users, items = torch.split(all_emb, [self.n_users, self.n_items])
        
        return users, items      
    
class CGMC(nn.Module):
    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim        
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        self.filters = nn.ModuleList() 
        for i in range(3):
            self.filters.append(nn.Embedding(emb_dim, emb_dim))
        # SCF : LAYER 1 LR 0.0001 

        self._init_weight_()
        
    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for i in range(3):
            nn.init.eye_(self.filters[i].weight)

    def forward(self, adj):
        embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_emb = [embeddings]
        for i in range(3):
            embeddings = (1-0.2) *  torch.sparse.mm(adj, embeddings) + 0.2*embeddings

            embeddings = F.sigmoid(torch.matmul(embeddings, self.filters[i].weight))
            all_emb += [embeddings]
        
        all_emb = torch.cat(all_emb, dim=1)
    
        users, items = torch.split(all_emb, [self.n_users, self.n_items])
        
        return users, items      
    
    
    
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_layers = len(self.weight_size)

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self._init_weight_()
        
    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj):
        all_emb = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        
        layer_embeddings  = [all_emb]
        
        for i in range(self.n_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            layer_embeddings.append(all_emb)
        layer_embeddings = torch.stack(layer_embeddings, dim=1)
        final_embeddings = layer_embeddings.mean(dim=1)  # output is mean of all layers
        users, items = torch.split(final_embeddings, [self.n_users, self.n_items])
        
        return users, items

class MF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj):
        u_g_embeddings = self.user_embedding.weight
        i_g_embeddings = self.item_embedding.weight
        return u_g_embeddings, i_g_embeddings


    
    
class DGCF(nn.Module):
    def __init__(self, n_users, n_items, data_config, args):
        super().__init__()
    
        #argument settings
        self.pretrain_data = None
        self.n_users = n_users
        self.n_items = n_items

        self.n_fold = 1
        self.norm_adj = data_config['norm_adj']
        self.norm_adj = self.norm_adj.to('cuda')
        # print("check1",type(self.norm_adj))3
        # self.all_h_list, self.all_t_list, self.all_v_list = self.load_adjacency_list_data(self.norm_adj)
        
        # self.A_in_shape = self.norm_adj.tocoo().shape

        # self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.n_factors = args.n_factors
        self.n_iterations = args.n_iterations
        self.n_layers = args.n_layers
        self.pick_level = args.pick_scale
        self.cor_flag = args.cor_flag
        if args.pick == 1:
            self.is_pick = True
        else:
            self.is_pick = False
        self.batch_size = args.batch_size
        #regularization
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        #interval of evaluation
        self.verbose = args.verbose
       
        # initialization of model parameter
        self.init_weights()

        
    def init_weights(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        if self.pretrain_data is None:
            self.weights = nn.ParameterDict({
                'user_embedding': nn.Parameter(initializer(torch.empty(self.n_users, self.emb_dim,requires_grad=True).to('cuda'))),
                'item_embedding': nn.Parameter(initializer(torch.empty(self.n_items, self.emb_dim,requires_grad=True).to('cuda')))
            })
            print('using xavier initialization')
        else:
            #check
            all_weights = nn.ParameterDict({
                'user_embedding': nn.Parameter(self.pretrain_data['user_embed']),
                'item_embedding': nn.Parameter(self.pretrain_data['item_embed'])
            })
            print('using pretrained initialization')
        # create models
        # self.ua_embeddings, self.ia_embeddings, self.f_weight, self.ua_embeddings_t, self.ia_embeddings_t = self._create_star_routing_embed_with_P(pick_=self.is_pick)
        # self.ua_embeddings, self.ia_embeddings = self._create_star_routing_embed_with_P(pick_=self.is_pick)
  

    def _create_star_routing_embed_with_P(self,pick_=False):
        '''
        pick_ : True, the model would narrow the weight of the least important factor down to 1/args.pick_scale.
        pick_ : False, do nothing.
        '''
        t1 = time()
        # p_test=False
        p_train=False

        A_values=torch.ones(self.n_factors,len(self.all_h_list)).to('cuda')
        # get a (n_factors)-length list of [n_users+n_items, n_users+n_items]
        # load the initial all-one adjacency values
        # .... A_values is a all-ones dense tensor with the size of [n_factors, all_h_list].
        

        # get the ID embeddings of users and items
        # .... ego_embeddings is a dense tensor with the size of [n_users+n_items, embed_size];
        # .... all_embeddings stores a (n_layers)-len list of outputs derived from different layers.
        ego_embeddings = torch.cat([self.weights['user_embedding'],self.weights['item_embedding']],0).to('cuda')
        all_embeddings = [ego_embeddings]
        # all_embeddings_t = [ego_embeddings]
        output_factors_distribution = []
       
        factor_num = [self.n_factors, self.n_factors, self.n_factors]
        
        iter_num = [self.n_iterations,self.n_iterations,self.n_iterations]
        for k in range(0,self.n_layers):
            # prepare the output embedding list
            # .... layer_embeddings stores a (n_factors)-len list of outputs derived from the last routing iterations.
            n_factors_l = factor_num[k]
            split_factor = int(self.emb_dim/n_factors_l)
            n_iterations_l = iter_num[k]
            layer_embeddings = []
            # layer_embeddings_t = []

            # split the input embedding table
            # .... ego_layer_embeddings is a (n_factors)-len list of embeddings [n_users+n_items, embed_size/n_factors]
            ego_layer_embeddings = torch.split(ego_embeddings, split_factor, 1)
            # ego_layer_embeddings_t = torch.split(ego_embeddings, split_factor, 1) 
            
            # perform routing mechanism
            for t in range(0, n_iterations_l):
                iter_embeddings = []
                # iter_embeddings_t = []
                A_iter_values = []

                # split the adjacency values & get three lists of [n_users+n_items, n_users+n_items] sparse tensors
                # .... A_factors is a (n_factors)-len list, each of which is an adjacency matrix
                # .... D_col_factors is a (n_factors)-len list, each of which is a degree matrix w.r.t. columns
                # .... D_row_factors is a (n_factors)-len list, each of which is a degree matrix w.r.t. rows
                if t == n_iterations_l - 1:
                    p_test = pick_
                    p_train = False
                A_factors, D_col_factors, D_row_factors = self._convert_A_values_to_A_factors_with_P(n_factors_l, A_values, pick= p_train)
                # A_factors_t, D_col_factors_t, D_row_factors_t = self._convert_A_values_to_A_factors_with_P(n_factors_l, A_values, pick= p_test)
                for i in range(0, n_factors_l):
                   
                    # update the embeddings via simplified graph convolution layer
                    # .... D_col_factors[i] * A_factors[i] * D_col_factors[i] is Laplacian matrix w.r.t. the i-th factor
                    # .... factor_embeddings is a dense tensor with the size of [n_users+n_items, embed_size/n_factors]
                    factor_embeddings = torch.sparse.mm(D_col_factors[i].to('cuda'), ego_layer_embeddings[i].to('cuda'))
                    # factor_embeddings_t = torch.sparse.mm(D_col_factors_t[i].cuda(), ego_layer_embeddings_t[i].cuda())
                    
                    # factor_embeddings_t = torch.sparse.mm(A_factors_t[i].cuda(), factor_embeddings_t.cuda())
                    factor_embeddings = torch.sparse.mm(A_factors[i].to('cuda'), factor_embeddings)

                    factor_embeddings = torch.sparse.mm(D_col_factors[i].to('cuda'), factor_embeddings)
                    # factor_embeddings_t = torch.sparse.mm(D_col_factors_t[i].cuda(), factor_embeddings_t.cuda())
                    
                    iter_embeddings.append(factor_embeddings)
                    # iter_embeddings_t.append(factor_embeddings_t)
                    
                    if t == n_iterations_l - 1:
                        layer_embeddings = iter_embeddings
                        # layer_embeddings_t = iter_embeddings_t 

                    # get the factor-wise embeddings
                    # .... head_factor_embeddings is a dense tensor with the size of [all_h_list, embed_size/n_factors]
                    # .... analogous to tail_factor_embeddings
                    
                    head_factor_embedings = torch.index_select(factor_embeddings, 0 ,torch.tensor(self.all_h_list).to('cuda'))
                    tail_factor_embedings = torch.index_select(ego_layer_embeddings[i].to('cuda'),0,torch.tensor(self.all_t_list).to('cuda'))

                    # .... constrain the vector length
                    # .... make the following attentive weights within the range of (0,1)
                    head_factor_embedings = F.normalize(head_factor_embedings, dim=1)
                    tail_factor_embedings = F.normalize(tail_factor_embedings, dim=1)
    
                    # get the attentive weights
                    # .... A_factor_values is a dense tensor with the size of [all_h_list,1]
                    A_factor_values = torch.sum(torch.mul(head_factor_embedings, F.tanh(tail_factor_embedings)), axis=1)

                    # update the attentive weights
                    A_iter_values.append(A_factor_values)
                    t1 = time()

                # pack (n_factors) adjacency values into one [n_factors, all_h_list] tensor
                A_iter_values = torch.stack(A_iter_values, 0)
                # add all layer-wise attentive weights up.
                A_values += A_iter_values.to('cuda')
                
                # if t == n_iterations_l - 1:
                #     #layer_embeddings = iter_embeddings
                #     output_factors_distribution.append(A_factors)
            t1 = time()
            # sum messages of neighbors, [n_users+n_items, embed_size]
            side_embeddings = torch.cat(layer_embeddings, 1)
            # side_embeddings_t = torch.cat(layer_embeddings_t, 1)
            
            ego_embeddings = side_embeddings
            # ego_embeddings_t = side_embeddings_t
            # concatenate outputs of all layers
            
            # all_embeddings_t += [ego_embeddings_t]
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, 1)
        all_embeddings = torch.mean(all_embeddings, dim=1, keepdims=False)
        # all_embeddings_t = torch.stack(all_embeddings_t, 1)
        # all_embeddings_t = torch.mean(all_embeddings_t, dim=1, keepdims=False)

        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        # u_g_embeddings_t, i_g_embeddings_t = torch.split(all_embeddings_t, [self.n_users, self.n_items], 0)

        # return u_g_embeddings, i_g_embeddings, output_factors_distribution, u_g_embeddings_t, i_g_embeddings_t
        return u_g_embeddings, i_g_embeddings

    
    def _convert_A_values_to_A_factors_with_P(self, f_num, A_factor_values, pick=True):
        A_factors = []
        D_col_factors = []
        D_row_factors = []
        #get the indices of adjacency matrix
        
        A_indices = np.mat([self.all_h_list, self.all_t_list]).transpose()
        D_indices = np.mat([list(range(self.n_users+self.n_items)),list(range(self.n_users+self.n_items))]).transpose()

        #apply factor-aware softmax function over the values of adjacency matrix
        #....A_factor_values is [n_factors, all_h_list]
        t1= time()
        if pick:
            
            A_factor_scores = F.softmax(A_factor_values, 0)
            min_A = torch.min(A_factor_scores, 0)
            index = A_factor_scores > (min_A + 0.0000001)
            index = index.type(torch.float32) * (self.pick_level - 1.0) + 1.0 #adjust the weight of the minimum factor to 1/self.pick_level

            A_factor_scores = A_factor_scores * index
            A_factor_scores = A_factor_scores / torch.sum(A_factor_scores, 0)
        else:
            
            A_factor_scores = F.softmax(A_factor_values, 0)
        # print("A_factor_scores", A_factor_scores.shape)
        # print(f_num)
        
        for i in range(0, f_num):
            # in the i-th factor, couple the adjcency values with the adjacency indices
            # .... A i-tensor is a sparse tensor with size of [n_users+n_items,n_users+n_items]
        
            
            A_i_scores = A_factor_scores[i]
            A_i_tensor = torch.sparse_coo_tensor(torch.tensor(A_indices).T, list(A_i_scores), self.A_in_shape).to('cuda')
        
            
            # get the degree values of A_i_tensor
            # .... D_i_scores_col is [n_users+n_items, 1]
            # .... D_i_scores_row is [1, n_users+n_items]
            D_i_col_scores = 1 / torch.sqrt(torch.sparse.sum(A_i_tensor, dim=1).to_dense())
            D_i_row_scores = 1 / torch.sqrt(torch.sparse.sum(A_i_tensor, dim=0).to_dense())
        
            
            # couple the laplacian values with the adjacency indices
            # .... A_i_tensor is a sparse tensor with size of [n_users+n_items, n_users+n_items]
            D_i_col_tensor = torch.sparse_coo_tensor(D_indices.T, D_i_col_scores, self.A_in_shape)
            D_i_row_tensor = torch.sparse_coo_tensor(D_indices.T, D_i_row_scores, self.A_in_shape)
            
            
            A_factors.append(A_i_tensor)
            D_col_factors.append(D_i_col_tensor)
            D_row_factors.append(D_i_row_tensor)
        
        
        #return a (n_factors)-length list of laplacian matrix
        return A_factors, D_col_factors, D_row_factors
    def load_adjacency_list_data(self, adj_mat):
        tmp = adj_mat.tocoo()
        # tmp = adj_mat
        # print('check',type(tmp))
        all_h_list = list(tmp.row)
        all_t_list = list(tmp.col)
        all_v_list = list(tmp.data)
        return all_h_list, all_t_list, all_v_list

    def create_initial_A_values(n_factors, all_v_list):
        return np.array([all_v_list] * n_factors)
    
    def forward(self,norm_adj):
        
        self.all_h_list, self.all_t_list, self.all_v_list = self.load_adjacency_list_data(norm_adj)
        self.A_in_shape = norm_adj.tocoo().shape
        # create models
        t1 = time()
        u_g_embeddings, i_g_embeddings= self._create_star_routing_embed_with_P(pick_=self.is_pick)
        
        return u_g_embeddings, i_g_embeddings
    
    
class UltraGCN(nn.Module):
    def __init__(self, n_users, n_items, params, args, incd_mat, i, N):
        super(UltraGCN, self).__init__()
        self.user_num = n_users
        self.item_num = n_items
        self.embedding_dim = args.embed_size

        ultra_config = ultra_config_dict(args.dataset)
        self.w1=ultra_config['w1']
        self.w2=ultra_config['w2']
        self.w3=ultra_config['w3']
        self.w4=ultra_config['w4']
        self.negative_weight = ultra_config['negative_weight']
        self.negative_num=ultra_config['negative_num']
        self.gamma =ultra_config['gamma']
        self.lambda_ = ultra_config['lambda_']
        self.initial_weight = ultra_config['initial_weight']
        self.ii_neighbor_num = ultra_config['ii_neighbor_num']
        self.sampling_sift_pos=ultra_config['sampling_sift_pos']
        self.lr=ultra_config['lr']
        self.batch_size=ultra_config['batch_size']
            

        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)


        train_mat = incd_mat.todok()
        items_D = np.sum(train_mat, axis = 0).reshape(-1)
        users_D = np.sum(train_mat, axis = 1).reshape(-1)

        beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

        constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                          "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}
         # Compute \Omega to extend UltraGCN to the item-item occurrence graph
        ii_cons_mat_path = '../Data/' + args.dataset + f'/_ii_constraint_mat_{i}_of_{N}'
        ii_neigh_mat_path = '../Data/' + args.dataset + f'/_ii_neighbor_mat_{i}_of_{N}'

        if os.path.exists(ii_cons_mat_path):
            ii_constraint_mat = pload(ii_cons_mat_path)
            ii_neighbor_mat = pload(ii_neigh_mat_path)
        else:
            ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, self.ii_neighbor_num)
            pstore(ii_neighbor_mat, ii_neigh_mat_path)
            pstore(ii_constraint_mat, ii_cons_mat_path)
            
        self.constraint_mat = constraint_mat
        self.ii_constraint_mat = ii_constraint_mat
        self.ii_neighbor_mat = ii_neighbor_mat

        self.initial_weights()
    
    def initial_weights(self):
        nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
        nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)


    
    def get_device(self):
        return self.user_embeds.weight.device

    def forward(self, adj):
        u_g_embeddings = self.user_embeds.weight
        i_g_embeddings = self.item_embeds.weight
        return u_g_embeddings, i_g_embeddings
    
    
'''
Useful functions
'''

def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def pstore(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f)
	print('store object in path = {} ok'.format(path))

def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero = False):
    
    print('Computing \\Omega for the item-item graph... ')
    A = train_mat.T.dot(train_mat)	# I * I
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    if ii_diagonal_zero:
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis = 0).reshape(-1)
    users_D = np.sum(A, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('i-i constraint matrix {} ok'.format(i))

    print('Computation \\Omega OK!')
    return res_mat.long(), res_sim_mat.float()

def ultra_config_dict(dataset):
    data_config = dict()
    if dataset in ['electronics']:

        data_config['w1']=1e-7
        data_config['w2']=1
        data_config['w3']=1e-7
        data_config['w4']=1
        data_config['negative_weight'] = 75
        data_config['negative_num']=75
        data_config['gamma'] =5e-5
        data_config['lambda_'] = 1e-7
        data_config['initial_weight'] = 1e-3
        data_config['ii_neighbor_num'] = 10
        data_config['sampling_sift_pos']=True
        data_config['lr']=1e-3
        data_config['batch_size']=2048


    elif dataset in ['yelp2018']:
        data_config['w1']=1e-8
        data_config['w2']=1
        data_config['w3']=1e-8
        data_config['w4']=1
        data_config['negative_weight'] = 800
        data_config['negative_num']=300
        data_config['gamma'] =1e-4
        data_config['lambda_'] =5e-4
        data_config['initial_weight'] = 1e-4
        data_config['ii_neighbor_num'] = 10
        data_config['sampling_sift_pos']=False
        data_config['lr']=1e-3
        data_config['batch_size']=1024
        

    elif dataset in ['gowalla']:
        data_config['w1']=1e-6
        data_config['w2']=1
        data_config['w3']=1e-6
        data_config['w4']=1
        data_config['negative_weight'] = 1500
        data_config['negative_num']=300
        data_config['gamma'] =1e-4
        data_config['lambda_'] =5e-4
        data_config['initial_weight'] = 1e-4
        data_config['ii_neighbor_num'] = 10
        data_config['sampling_sift_pos']=False
        data_config['lr']=1e-4
        data_config['batch_size']=512
    
        
    elif dataset in ['amazon-book']:
        data_config['w1']=1e-8
        data_config['w2']=1
        data_config['w3']=1e-8
        data_config['w4']=1
        data_config['negative_weight'] = 800
        data_config['negative_num']=300
        data_config['gamma'] =1e-4
        data_config['lambda_'] =5e-4
        data_config['initial_weight'] = 1e-4
        data_config['ii_neighbor_num'] = 10
        data_config['sampling_sift_pos']=False
        data_config['lr']=1e-3
        data_config['batch_size']=1024
        
    elif dataset in ['ml-1m']:
        data_config['w1']=1e-7
        data_config['w2']=1
        data_config['w3']=1e-87
        data_config['w4']=1
        data_config['negative_weight'] = 200
        data_config['negative_num']=200
        data_config['gamma'] =1e-4
        data_config['lambda_'] =1e-3
        data_config['initial_weight'] = 1e-3
        data_config['ii_neighbor_num'] = 10
        data_config['sampling_sift_pos']=False
        data_config['lr']=1e-3
        data_config['batch_size']=1024
        # data_config['embedding_dim']=6
        
        
        
    return data_config
     