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

기존 신경망 모델의 출력 함수 f는 입력값이 x일 때, **f(x) = tanh(x)** 혹은 **f(x) = (1+e^(-x))^-1** (하이퍼 탄젠트와 시그모이드)를 사용했다. 이 함수들은 **saturating nonlinearity** 로 훈련 시간이 **non-saturating nonlinearity** 인 **ReLU(f(x) = max(0,x))** 보다 더 느리다. AlexNet에서는 마지막 layer를 제외하고 모두 ReLU를 사용하였으며, CIFAR-10에서 training error가 25%에 도달하기까지 tanh과 ReLU를 비교한 결과 속도에서 **6배** 의 차이가 발생했다.  

| 이름 | 설명  
|:-----|:----  
|Saturating nonlinearity | 어떤 입력값 x가 무한대로 가면 함수의 값도 무한대로 간다.  
|Non-saturating nonlinearity | 어떤 입력값 x가 무한대로 가면 함수의 값이 어떤 범위 내에서만 움직인다.  
> Table 1. Saturating nonlinearity & Non-saturating nonlinearity  


<img alt="ReLU.png" src="https://user-images.githubusercontent.com/43739827/91699092-badb6a00-ebae-11ea-9785-a242dae97375.png"></img>  
> Fig 1. ReLU function  

### 3.2 Training on Multiple GPUs  

논문 작성 당시 120만 개의 훈련 데이터를 이용해 학습하기에는 메모리의 한계가 존재했기 때문에, 두 개의 GTX 580 GPU를 병렬연결하여 사용하였다.  

### 3.3 Local Response Normalization(LRN)  

ReLU의 장점 중 하나는 시그모이드나 하이퍼 탄젠트와 달리 데이터에 정규화를 요구하지 않는다는 것이다.  
> 하이퍼 탄젠트와 시그모이드가 가질 수 있는 양의 결괏값은 최대 1로 한정되어 있으나, ReLU는 양수의 입력값을 그대로 반환하기 때문에 제한을 두지 않는다.  

하지만 ReLU는 양의 방향으로 무한히 커질 수 있으며 이런 경우에는 인접값들이 무시될 수도 있기 때문에 정규화를 수행하는것이 일반적으로 좋다.  
해당 논문에서는 AlexNet의 정규화 과정으로 **LRN(Local Response Normalization)** 을 사용했다. LRN의 값 (b(x,y))^i는 아래와 같다.  

<img alt="LRN.png" src="https://user-images.githubusercontent.com/43739827/91700356-bd3ec380-ebb0-11ea-87c7-8037fe4ed18c.PNG"></img>  

(a(x,y))^i는 좌표 (x,y)에서 커널 i를 적용하고 ReLU 함수를 적용한 뉴런이다. N은 layer의 전체 커널 수를 나타내며 나머지 상수 k, n, α, β는 **validation set** 을 구축하기 위한 **하이퍼 파라미터** 이다. AlexNet에서는 해당 파라미터들에 대하 **k = 2, n = 5, α = 10^-4, β = 0.75** 를 사용하였다.  

### 3.4 Overlapping Pooling  

기존의 CNN들은 Pooling을 할 때 stride와 kernel size를 **같게** 하여 이미지의 크기를 절반으로 줄였지만, AlexNet에서는 **stride=2, kernel size=3** 으로 설정하여 Pooling이 겹치도록 하였다. 논문의 저자의 따르면 이러한 overlapping pooling 방식이 기존의 방식보다 과적합을 줄이는데 더 효율적이었다고 말하고있다.  
> stride를 s, kernel size를 z라 했을 때, s=z라면 기존의 pooling 방식이며 s < z면 overlapping pooling이다.  

### 3.5 Overall Architecture  

AlexNet은 각기 가중치를 가지는 8개의 layer로 구성되어있다. 초기 5개의 layer들은 Convolutional layer이며 나머지 3개의 layer들은 **전결합층(fully-connected layer)** 이다. 출력층인 마지막 전결합층은 1000개의 뉴런에 대해 **softmax** 를 사용하였다.  

<img alt="Architecture.png" src="https://user-images.githubusercontent.com/43739827/91701194-e744b580-ebb1-11ea-9a07-0c40e3a73ed4.png"></img>  
> Fig 2. AlexNet Architecture  

첫 번째 convolutional layer는 224x224x3의 이미지를 입력받아 11x11x3 크기의 커널을 96개 사용하며 이 때 stride는 4이다. 첫 번째 convolutional layer에는 정규화가 사용된다.  
두 번째 convolutional layer는 첫 번째 convolutional layer의 출력값을 입력받으며 5x5x48 크기의 커널을 256개 사용한다. 두 번째 convolutional layer에도 정규화가 사용되며, 나머지 layer들에는 정규화가 더 이상 사용되지 않는다.  
> layer에서 사용되는 정규화 방식들은 모두 LRN이다.  

세 번째 convolutional layer는 두 번째 convolutional layer의 출력값을 입력받으며 3x3x256 크기의 커널을 384개 사용한다.  
네 번째 convolutional layer는 세 번째 convolutional layer의 출력값을 입력받으며 3x3x192 크기의 커널을 384개 사용한다.  
다섯 번째 convolutional layer는 네 번재 convolutional layer의 출력값을 입력받으며 3x3x192 크기의 커널을 256개 사용한다.  
나머지 전결합층들은 각각 4096개의 커널을 가진다.  

## 4. Reducing Overfitting  

AlexNet에서는 6천만 개의 파라미터를 사용한다. 하지만 ILSVRC 버전의 ImageNet을 학습하게 되면 너무 많은 파라미터로 인해 과적합이 발생할 수 있으므로 AlexNet은 이 과적합을 예방하기 위해 두 가지 기술을 사용하였다.  

### 4.1 Data Augmentation  

이미디 데이터를 처리하는 데 있어 과적합을 줄이는 가장 쉬운 방법은 데이터 셋의 크기를 키우는 것이다. **데이터 증가(Data Augmentation)** 는 원본 이미지들을 조금씩 변경하여 새로운 이미지로 취급하는 것이다. 이렇게 변형된 이미지들은 디스크에 저장되지 않는다.  

첫 번째 데이터 증가 방법은 **이미지 변환 및 수평 대칭(image translation and horizontal reflections)** 이다. 256x256 크기의 이미지로부터 수평 대칭한 224x224 크기의 패치를 무작위로 추출하여 훈련시킨다. 이렇게 하여 훈련 셋의 크기를 2048로 키울 수 있었으며, 데이터 증가를 하지 않고 훈련 시 과적합이 발생하였음을 확인하였다.  
두 번째 데이터 증가 방법은 훈련 이미지의 RGB 채널 값을 키우는 것이다.  

### 4.2 Dropout

각기 다른 모델의 예측값들을 합치는 것은 test error를 줄이는 데 상당히 효율적이지만, 큰 신경망에서는 이미 훈련에 많은 시간을 투자했음에도 오랜 시간이 추가적으로 소모되기 때문에 상당히 비효율적이다. 그렇기 때문에 AlexNet은 이러한 고전적인 방법을 사용하지 않고 **dropout** 이라는 기술을 채택하였다. dropout은 버릴 뉴런의 비율을 선택하는 것으로 버려지는 뉴런은 정보(data)를 다음 layer에 넘겨주지 않으며 역전파에서도 해당되지 않는다.  

이런 dropout을 사용하는 이유는 layer의 뉴런이 다른 뉴런들의 존재에 의존하지 않게 되어 layer의 복잡도가 줄어들기 때문이다. AlexNet에서는 최초 두 개의 전결합층에서 dropout을 사용했으며, dropout을 하지 않고 기존의 신경망처럼 훈련했을 경우 과적합이 발생하는것을 확인하였다.
