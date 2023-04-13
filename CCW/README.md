# Separated-Collaborative-Filtering

## Pytorch-NGCF
Original Pytorch  Implementation can be found [here](https://github.com/liu-jc/PyTorch_NGCF)

## Run the Code

### Options
- ```--scc``` : Whether use or not Spectral Co-Clustering. 0 : unable / 1 : Cluster Only / 2 : Full(global) + cluster(local)
- ```--N``` : The number of cluster.
- ```--cl_num``` 몇 번째 cluster로 실험할건지(scc옵션 1)

```
python main.py --dataset gowalla --alg_type ngcf --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 4096 --epoch 5000 --verbose 1 --mess_dropout [0.1,0.1,0.1] --scc 2 --N 3

```
