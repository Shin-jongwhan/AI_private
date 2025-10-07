### 251007
# 행렬 연산 개선
#### 행렬 곱연산은 단순한 계산으로 하면 O(m * n * k)만큼 걸린다.
- m : 첫 번째 행렬의 행 수
- n : 첫 번째 행렬의 열 수
- k : 두 번째 행렬의 열 수
#### <br/>

#### 그리고 머신 러닝에서는 은닉층 개수만큼 행렬의 곱연산이 이루어져야 한다. 이를 i라고 하자.
#### 그러면 O(m * n * k * i)만큼 연산이 이루어져야 한다.
### <br/><br/>

## 중첩 루프의 연산 횟수를 수학적으로 해결하기
### 다음과 같이 중첩루프의 연산을 줄일 수 있다.
### 이러한 방법을 이용하면 머신 러닝에서도 연산을 줄일 수 있다.
### 왜냐면 머신 러닝은 중첩 루프의 계산이 매우 많기 때문이다.
#### https://github.com/Shin-jongwhan/AI_private/blob/main/script/nesting_loop_calulator.py

### <br/><br/>

## 머신 러닝 연산
#### https://github.com/Shin-jongwhan/AI_private/blob/main/script/deep_learning_from_scratch/ch4.%EC%8B%A0%EA%B2%BD%EB%A7%9D%20%ED%95%99%EC%8A%B5/two_layer_net.py
### 나는 여기서 연속적인 연산에 대해 집중해본다.
### 학습 때는 중간 계산값을 저장해야 한다.
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

### 하지만 추론 때는 다르다.
#### 추론 때는 학습 때 순전파의 계산 과정과 거의 비슷하지만 추론에서만 쓰이는 아주 약간의 수식만 다를 뿐, 같은 매트릭스를 사용하며, 중간 결과물은 필요하지 않고 최종 값만 있으면 된다.
#### -> 그러면 중간 계산 과정을 생략하고 최종 결과물 계산만 하면 된다. 
### <br/>

### 합성곱 (CNN) 에서도 이러한 기법이 적용될 수 있다.
#### CNN은 한번 합성곱 연산을 수행할 때 3번을 연산한 후 각 3번의 값을 더해서 최종 매트릭스를 산출한다.
#### 이것도 한 번의 연산으로 단축시킬 수 있다. 
#### <img width="531" height="943" alt="image" src="https://github.com/user-attachments/assets/5c3baeb4-1512-4724-9b72-d789542f981a" />
