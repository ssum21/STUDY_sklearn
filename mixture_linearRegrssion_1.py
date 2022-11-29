import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# boston 데이터를 위한 모듈을 불러옵니다. 
from sklearn.datasets import load_boston

"""
1. 사이킷런에 존재하는 데이터를 불러오고, 
   불러온 데이터를 학습용 데이터와 테스트용 데이터로
   분리하여 반환하는 함수를 구현합니다.
   
   Step01. 사이킷런에 존재하는 boston 데이터를 
           (X, y)의 형태로 불러옵니다. 
   
   Step02. 불러온 데이터를 
           학습용 데이터와 테스트용 데이터로 분리합니다.
           
           학습용 데이터로 전체 데이터의 80%를 사용하고, 
           테스트용 데이터로 나머지 20%를 사용합니다.
           
           동일한 결과 확인을 위하여 random_state를
           100으로 설정합니다.
"""
def load_data():
    
    X, y  = load_boston(return_X_y = True)
     
    print("데이터의 입력값(X)의 개수 :", X.shape[1])
    #shape는 (row,cloumns)를 반환해주는 함수임.
    
    train_X, test_X, train_y, test_y = train_test_split(X,y, test_size=0.2, random_state=100)
    
    return train_X, test_X, train_y, test_y
    
"""
2. 다중 선형회귀 모델을 불러오고, 
   불러온 모델을 학습용 데이터에 맞추어 학습시킨 후
   해당 모델을 반환하는 함수를 구현합니다.

   Step01. 사이킷런에 구현되어 있는 
           다중 선형회귀 모델을 불러옵니다.

   Step02. 불러온 모델을 학습용 데이터에 맞춰
           학습시킵니다.
"""
def Multi_Regression(train_X,train_y):
    
    multilinear = LinearRegression()
    
    multilinear.fit(train_X,train_y)
    #fit(X,y) : X,y 데이터셋에 대해서 모델을 학습시킵니다.
    
    return multilinear
    
"""
3. 모델 학습 및 예측 결과 확인을 위한 main 함수를 완성합니다.
   
   Step01. 학습이 완료된 모델을 활용하여 
           테스트 데이터에 대한 예측을 수행합니다.
        
   Step02. 사이킷런 회귀 모델 내에 구현되어 있는 
           score 함수를 사용하여 
           모델 학습 평가 점수를 model_score 변수에
           저장합니다. 
   
   Step03. 학습된 모델의 beta_0와 beta_i들을 
           각각 변수 beta_0와 beta_i_list에 저장합니다.
"""
def main():
    
    train_X, test_X, train_y, test_y = load_data()
    
    multilinear = Multi_Regression(train_X,train_y)
    
    predicted = multilinear.predict(test_X)
    #test_y를 적지 않는 이유는 X의 예측값을 알고 싶은 것이기 때문임.
    
    model_score = multilinear.score(test_X, test_y)
    
    print("\n> 모델 평가 점수 :", model_score)
     
    #Y=beta_0+beta_1*X_1+beta_2*X_2...N
    beta_0 = multilinear.intercept_
    beta_i_list = multilinear.coef_
    
    print("\n> beta_0 : ",beta_0)
    print("> beta_i_list : ",beta_i_list)
    
    return predicted, beta_0, beta_i_list, model_score
    
if __name__ == "__main__":
    main()

"""
다중 선형 회귀 모델 구현하기
다중 선형 회귀는 입력값이 1개일 경우 적용하는 단순 선형 회귀 알고리즘과 달리 입력값 X가 여러 개일 때 사용할 수 있는 회귀 알고리즘입니다.

다중 선형 회귀 또한 사이킷런에 구현되어 있는 라이브러리를 활용하여 간단하게 모델을 구현해보겠습니다.

단순 선형 회귀 실습을 통해 LinearRegression 클래스를 정의하고 사용한 것을 기억하시나요?

사실 위 클래스는 다중 선형 회귀에서도 사용이 가능합니다. 사이킷런에서는 선형 회귀 라는 이름으로 단순/다중 선형 회귀의 구분 없이 동일한 모듈을 활용할 수 있습니다.

사이킷런에 저장된 데이터를 불러오고, 불러온 데이터를 다중 선형 회귀 모델을 사용해 예측을 진행해보도록 하겠습니다.

우리가 사용할 데이터는 1978년에 발표된 ‘보스턴 주택 가격 데이터’로, 미국 보스턴 지역의 주택 가격에 영향을 미치는 요소들(X_i) 및 주택 가격(y)으로 구성되어 있습니다.
"""