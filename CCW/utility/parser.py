import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run SNGCF.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')

    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='gowalla',
                        help='Choose a dataset from {gowalla, yelp2018, amazon-book, movielens-20m}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.') 
    parser.add_argument('--layer_size', nargs='?', default='[64]',
                        help='Output sizes of every layer')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='sngcf',
                        help='Specify the name of model (sngcf).')
    parser.add_argument('--adj_type', nargs='?', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='rw',
                        help='Specify the type of the graph convolutional layer from {rw, rw_single, rw_fixed, rw_single_svd, rw_svd, rw_final, rw_linear}.')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    parser.add_argument('--scc', type=int, default=0,
                        help='Spectral CC Sampling. 0 : scc , 1 : original')
    parser.add_argument('--N', type=int, default=5,
                        help='the number of cluster')
    parser.add_argument('--cl_num', type=int, default=0,
                        help='select cluster')
    
    
    
    
    parser.add_argument('--cor_flag', type=int, default=1,
                        help='Correlation matrix flag')
    parser.add_argument('--corDecay', type=float, default=0.0,
                        help='Distance Correlation Weight')
    
        
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Layer numbers.')
    parser.add_argument('--n_factors', type=int, default=4,
                        help='Number of factors to disentangle the original embed-size representation.')
    parser.add_argument('--n_iterations', type=int, default=2,
                        help='Number of iterations to perform the routing mechanism.')
    
    parser.add_argument('--pick', type=int, default=0,
                        help='O for no pick, 1 for pick')
    parser.add_argument('--pick_scale', type=float, default=1e10,
                        help='Scale')
    parser.add_argument('--wandb', type=bool, default=False,
                        help='wandb record')
    parser.add_argument('--coclust', type=str, default='scc',
                        help='coclustering methods')
    return parser.parse_args()
