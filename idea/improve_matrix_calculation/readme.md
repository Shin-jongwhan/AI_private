### 251007
# 행렬 연산 개선
#### 행렬 곱연산은 단순한 계산으로 하면 O(m * n * k)만큼 걸린다.
- m : 첫 번째 행렬의 행 수
- n : 첫 번째 행렬의 열 수
- k : 두 번째 행렬의 열 수
#### <br/>

#### 그리고 머신 러닝에서는 은닉층 개수만큼 행렬의 곱연산이 이루어져야 한다. 이를 i라고 하자.
#### 그러면 O(m * n * k * i)만큼 연산이 이루어져야 한다.
#### <br/><br/>

### 그러면 이 코드를 한 번 봐보자.
#### https://github.com/Shin-jongwhan/AI_private/blob/main/script/deep_learning_from_scratch/ch4.%EC%8B%A0%EA%B2%BD%EB%A7%9D%20%ED%95%99%EC%8A%B5/two_layer_net.py
### 나는 여기서 연속적인 연산에 대해 집중해본다.
#### 이건 순전파 쪽인데, 최종적으로 필요한 값은 y이고, 중간 값들을 같이 계산해야 한다.
#### 중간 계산을 모두 생략해버릴 수 있다면 계산을 아주 획기적으로 줄일 수 있을 것 이다.
#### 그런데 문제는 역전파 때 각 중간값들을 꼭 캐시로 남겨두어야 이용할 수 있다. 
```
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
```
### <br/>
