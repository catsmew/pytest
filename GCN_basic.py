### GCN Cora dataset

## 데이터 분포 확인
## PyG에서 제공하는 Planetoid 데이터셋
from torch_geometric.datasets import Planetoid
## 그래프의 각 노드에 연결된 엣지의 개수(Degree)를 계산하는 함수
from torch_geometric.utils import degree 
## 아이템의 개수를 셀 수 있는 파이썬 기본 라이브러리
from collections import Counter
## 데이터 시각화 라이브러리
import matplotlib.pyplot as plt

## Cora 데이터셋을 PyG의 Planetoid 클래스를 사용해 불러옴
dataset = Planetoid(root="/Users/chanwoo/Cora", name = 'Cora')
## Cora 데이터셋의 첫 번째 그래프를 data 객체로 가져옴(Cora는 1개의 그래프만 있음)
data = dataset[0]

## 각 노드의 차수를 계산(edge_index[0]은 엣지의 출발 노드를 의미)
degrees = degree(data.edge_index[0]).numpy()
## 각 차수에 해당하는 노드의 개수를 계산
numbers = Counter(degrees)

### Cora dataset은 2,708개의 Node, 각 노드의 Degree 분포는 Long-tail 분포
fig, ax = plt.subplots()
ax.set_xlabel('Node degree')
ax.set_ylabel('Numbers of nodes')
plt.bar(numbers.keys(), numbers.values())
# plt.show()

### PyG의 GCN Layer를 활용해 Cora 데이터셋에 GNN 적용
## PyTorch의 기본 패키지 설치
import torch
## PyTorch의 다양한 함수(활성화 함수, 손실 함수 등)가 포함된 모듈
import torch.nn.functional as F
## PyG에서 제공하는 Graph Convolution Layer
from torch_geometric.nn import GCNConv

dataset = Planetoid(root="/Users/chanwoo/py_code/data", name = 'Cora')
data = dataset[0]

# 정확도 함수
def accuracy(y_pred, y_true):
    """Calculate accuracy"""
    return torch.sum(y_pred == y_true) /len(y_true)

## GCN Class 정의(PyTorch의 nn.Module을 상속받아 GCN 모델 정의)
class GCN(torch.nn.Module):
    """Graph Convolution Network"""
    ## __init()__: GCN 클래스의 초기화 메서드
    ## 입력 데이터의 특성 개수, 히든 레이어의 차원, 출력 결과 값의 차원을 입력 받음. 이후 2개의 GCN 레이어를 쌓음
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        ## 첫 번째 GCN 레이어 정의
        self.gcn1 = GCNConv(dim_in, dim_h)
        ## 두 번째 GCN 레이어 정의
        self.gcn2 = GCNConv(dim_h, dim_out)
        
    ## forward(): 입력 데이터 x와 엣지 인덱스 edge_indeex를 입력으로 받아서, GCN 통과시킨 후 예측 값 반환
    ## 데이터에서 노드를 나타내는 x와 엣지를 나타내는 edge_index를 입력으로 받음. log_softmax를 통해 classfication 수행
    def forward(self, x, edge_index):
        ## 첫 번째 GCN 레이어를 거쳐 중간 표현을 계산
        h = self.gcn1(x, edge_index)
        ## ReLU 활성화 함수를 적용
        h = torch.relu(h)
        ## 두 번째 GCN 레이어 적용
        h = self.gcn2(h, edge_index)
        ## 클래스 확률을 log_softmax로 변환해 반환
        return F.log_softmax(h, dim = 1)
    
    ## fit(): 손실 함수, 옵티마이저를 불러오고 훈련을 진행
    def fit(self, data, epochs):
        ## 교차 엔트로피 손실 함수 정의
        criterion = torch.nn.CrossEntropyLoss()
        ## Adam 옵티마이저 설정
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr = 0.01, 
                                     weight_decay = 5e-3)
        ## 모델을 훈련 모드로 전환
        self.train()
        ## 주어진 에포크 수만큼 반복하며 모델을 학습
        for epoch in range(epochs + 1):
            ## 옵티마이저 기울기 초기화
            optimizer.zero_grad()
            ## 모델에 데이터 입력해 예측 결과 Get
            out = self(data.x, data.edge_index)
            ## 훈련 데이터에 대한 손실 값을 계산
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            ## 훈련 데이터에 대한 정확도 계산
            acc = accuracy(out[data.train_mask].argmax(dim = 1),
                           data.y[data.train_mask])
            ## 역전파를 통해 기울기 계산
            loss.backward()
            ## 옵티마이저를 통해 모델 파라미터 업데이트
            optimizer.step()
            
            ## 매 20 에포크마다 검증 데이터에 대한 손실과 정확도를 출력
            if(epoch % 20 == 0):
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy(out[data.val_mask].argmax(dim = 1),
                                   data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss: .3f} | Train Acc:'
                      f' {acc*100:>5.2f}% | Val Loss: {val_loss: .2f} |'
                      f'Val Acc: {val_acc*100:.2f}%')
    ## 평가 중에는 기울기 계산을 하지 않도록 설정            
    @torch.no_grad()
    def test(self, data):
        ## 모델을 평가 모드로 전환
        self.eval()
        ## 테스트 데이터에 대해 예측 수행
        out = self(data.x, data.edge_index)
        acc = accuracy(out.argmax(dim = 1)[data.test_mask], data.y[data.test_mask])
        return acc
    
# 인슽턴스 저장 및 100 에포크 훈련, 테스트셋에 적용
# Hidden Layer의 피쳐 수는 16
## Cora 데이터셋의 입력 피처 수, 히든 레이어 차원(16), 출력 클래스 수를 지정하여 GCN 모델을 초기화
gcn = GCN(dataset.num_features, 16, dataset.num_classes)
## 모델 구조를 출력
print(gcn) # 아웃 클래스 7개

# Train
gcn.fit(data, epochs = 500)

# Test
acc = gcn.test(data)
print(f'\nGCN test accuracy: {acc*100:.2f}%\n')
print(data)

