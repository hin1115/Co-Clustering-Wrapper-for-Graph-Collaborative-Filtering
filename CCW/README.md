# Separated-Collaborative-Filtering

## Pytorch-NGCF
아래 링크 참고하여 수정함    
Original Pytorch  Implementation can be found [here](https://github.com/liu-jc/PyTorch_NGCF)

## 진행상황
- NGCF 원본 동작 확인 ```--scc 0``` 옵션
- 개별 cluster에 대해 학습 확인 ```--scc 1 --cl_num ?``` 옵션
- incd matrix(Full, clustered)는 최초 실행시 파일로 저장
### 주요 코드 부분
- ```utility/load_data.py``` : ```def get_adj_mat```, ```def co_clustering```
- ```Models.py``` : ```class UCR``` 새로 만들었음 (clustering + full 섞은 버전)

## Run the Code

**원본 대비 옵션 4개 추가** 
- ```--scc``` : Spectral Co-Clustering 적용 여부. 0 : 미적용(full버전) / 1 : cluster 1개만 실행 / 2 : Full(global) + cluster(local)
- ```--N``` : 몇 개의 cluster로 나눌 건지
- ```--cl_num``` 몇 번째 cluster로 실험할건지(scc옵션 1)

```
python main.py --dataset gowalla --alg_type ngcf --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 4096 --epoch 5000 --verbose 1 --mess_dropout [0.1,0.1,0.1] --scc 2 --N 3

```
