# AlexNet  

 **AlexNet** 은 2012년 ILSVRC(ImageNet Large Scale Visual Recognition Challenge)에서 가장 높은 점수를 얻은 CNN(Convolutional Neural Network)이다.  
논문의 1저자인 Alex Krizhevsky의 이름을 따서 AlexNet이라는 이름을 갖게 되었다.  

## 1. Introduction  

 논문의 저자들은 성능을 향상시키기 위해 대용량의 데이터 셋을 수집하고 일반적인 CNN 모델보다 더 강력한 모델을 구축하였으며, **과적합(overfitting)** 을 피하기 위해 기존 모델들에서 사용되지 않았던 기술(dropout)을 사용하였다. 사용된 데이터 셋은 **ImageNet** 으로 120만 개의 고화질 이미지(high-resolution image)와 1000개의 각기 다른 클래스들로 이루어져 있다.  

 AlexNet의 성능 비교 대상은 비슷한 크기를 가지는 기존의 신경망(Neural Network)이다. CNN은 신경망과 비교해 더 적은 연결(**sparse connection**)과 파라미터를 가져 훈련(train) 하기 쉽다는 이점이 있지만 대용량의 고화질 이미지를 데이터 셋으로 사용할 때 시간이 오래 걸린다는 단점이 존재했다. 이를 해결하기 위해 AlexNet은 **두 개** 의 **GTX 530 3GB GPU** 를 **병렬연결** 하여 훈련했으며 5~6일이라는 시간이 소모되었다.  

## 2. The Dataset  

본래 ImageNet은 약 22000개의 범주(category)를 가지는 1500만 개 이상의 고화질 이미지 데이터 셋이다. ILSVRC에서는 이 ImageNet에서 1000개의 범주를 가지는 대략 1000개의 이미지를 사용하였다. AlexNet이 이 ImageNet을 사용한 이유는 ILSVRC-2010 버전의 ImageNet이 유일하게 테스트 셋에 레이블이 있기 때문이다.  

## 3. The Architecture  

### 3.1 ReLU Nonlinearity  

기존 신경망 모델의 출력함수 f는 입력값이 x일 때, **f(x) = tanh(x)** 혹은 **f(x) = (1+e^(-x))^-1** (하이퍼탄젠트와 시그모이드)를 사용했다. 이 함수들은 **saturating nonlinearity** 로 훈련 시간이 **non-saturating nonlinearity** 인 **ReLU(f(x) = max(0,x))** 보다 더 느리다. AlexNet에서는 마지막 layer를 제외하고 모두 ReLU를 사용하였으며, CIFAR-10에서 training error가 25%에 도달하기까지 tanh과 ReLU를 비교한 결과 속도에서 **6배** 의 차이가 발생했다.  

| 이름 | 설명  
|:----:|:----  
|Saturating nonlinearity | 어떤 입력값 x가 무한대로 가면 함수의 값도 무한대로 간다.  
|Non-saturating nonlinearity | 어떤 입력값 x가 무한대로 가면 함수의 값이 어떤 범위 내에서만 움직인다.  
> Table 1. Saturating nonlinearity & Non-saturating nonlinearity  


<img src="ReLU.png" alt="https://user-images.githubusercontent.com/43739827/91699092-badb6a00-ebae-11ea-9785-a242dae97375.png"></img>  
