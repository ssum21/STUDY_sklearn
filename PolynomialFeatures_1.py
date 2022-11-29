"""
다항 회귀 모델 구현하기
다항 회귀는 Y를 X에 대한 임의의 다항 함수로 모델링하는 선형 회귀를 의미합니다.

다항 회귀는 먼저 입력 데이터 X에 대한 전처리를 진행해준 후 다중 선형 회귀를 적용함으로써 구현됩니다.

사이킷런을 이용하면 입력 데이터에 대한 변환(전처리) 을 편리하게 진행할 수 있습니다.

따라서 이번 시간에는 사이킷런을 활용하여 다항 회귀를 구현해보겠습니다.

다항 회귀를 위한 사이킷런 함수/라이브러리

PolynomialFeatures(degree, include_bias): Polynomial 객체를 생성합니다.
degree: 만들어줄 다항식의 차수를 의미합니다.
include_bias : 편향 변수의 추가 여부를 설정합니다.(True/False) True로 설정하게 되면, 해당 다항식의 모든 거듭제곱이 0일 경우 편향 변수를 추가합니다. 이는 회귀식에서 β_0  와 같은 역할을 합니다.
[PolynomialFeatures].fit_transform(X): 데이터 X와 X의 degree제곱을 추가한 데이터를 반환합니다.
fit(X)와 transform(X) 을 각각 분리해서 진행하는 것도 가능합니다.

실습
PolynomialFeature 객체를 활용하여 각 변수 값을 제곱하고, 데이터에 추가하는 함수 Polynomial_transform()를 구현합니다.
사이킷런에 구현되어 있는 다중 선형회귀 모델을 불러와 전체 데이터에 대한 학습을 진행하고, 해당 모델을 반환하는 함수 Multi_Regression() 을 구현합니다.
모델 학습 및 예측 결과 확인을 위한 main() 함수를 완성합니다.
실행 버튼을 눌러 변환된 데이터와 실제 데이터, 예측값의 시각화 그래프를 확인합니다.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 다항 회귀의 입력값을 변환하기 위한 모듈을 불러옵니다.
from sklearn.preprocessing import PolynomialFeatures

def load_data():
    
    np.random.seed(0)
    
    X = 3*np.random.rand(50, 1) + 1
    y = X**2 + X + 2 +5*np.random.rand(50,1)
    
    return X, y
    
"""
1. PolynomialFeature 객체를 활용하여 
   각 변수 값을 제곱하고, 
   데이터에 추가하는 함수를 구현합니다.
   
   Step01. Polynomial 객체를 
           생성합니다.
           
           degree를 2로 설정하고, 
           include_bias 파라미터를 
           True로 설정합니다.
   
   Step02. 변수 값을 제곱하고 
           이를 X에 추가시킨 후 
           poly_X에 저장합니다.
"""
def Polynomial_transform(X):
    
    poly_feat = PolynomialFeatures(degree=2,include_bias=True)
    
    poly_X = poly_feat.fit_transform(X)
    
    print("변환 이후 X 데이터\n",poly_X[:3])
    
    return poly_X
    
"""
2. 다중 선형회귀 모델을 불러오고, 
   불러온 모델을 학습용 데이터에 맞추어 
   학습시킨 후 해당 모델을 반환하는 
   함수를 구현합니다.

   Step01. 사이킷런에 구현되어 있는 
           다중 선형회귀 모델을 불러옵니다.

   Step02. 불러온 모델을 제곱값이 추가된 
           데이터에 맞춰 학습시킵니다.
"""
def Multi_Regression(poly_x, y):
    
    multilinear = LinearRegression()
    
    multilinear.fit(poly_x, y)
    
    return multilinear
    
    
# 그래프를 시각화하는 함수입니다.
def plotting_graph(x,y,predicted):
    fig = plt.figure()
    plt.scatter(x, y)
    
    plt.scatter(x, predicted,c='r')
    plt.savefig("testPolynomialFeatures.png")
    
    
"""
3. 모델 학습 및 예측 결과 확인을 위한 
   main 함수를 완성합니다.
   
   학습이 완료된 모델을 활용하여 
   테스트 데이터에 대한 예측을 수행합니다.
"""
def main():
    
    X,y = load_data()
    
    poly_x = Polynomial_transform(X)
    
    linear_model = Multi_Regression(poly_x,y)
    
    predicted = linear_model.predict(poly_x)
    
    plotting_graph(X,y,predicted)
    
    return predicted
    
if __name__=="__main__":
    main()