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

이런 dropout을 사용하는 이유는 layer의 뉴런이 다른 뉴런들의 존재에 의존하지 않게 되어 layer의 복잡도가 줄어들기 때문이다. AlexNet에서는 최초 두 개의 전결합층에서 dropout을 사용했으며, dropout을 하지 않고 기존의 신경망처럼 훈련했을 경우 과적합이 발생하는 것을 확인하였다.

## 5. Details of learning  

AlexNet을 훈련하는 데에는 **batch_size = 128, momentum = 0.9, weight decay = 0.0005인 stochastic gradient descent** 를 사용하였다.  
> Gradient Descent를 계산함에 있어 데이터의 부분집합(Mini Batch)을 사용하는 방법이다.  

모든 layer에는 동일한 **학습률(learning rate)** 을 사용했으며 validation error가 현재의 학습률에 대해서 더 이상 개선되지 않을 때에는 10으로 나누었다. 초기 학습률은 0.01로 시작하였으나 모델이 완성되기까지 총 세 번 학습률이 줄어들었다.  
AlexNet은 대략 90번 훈련하였으며 훈련 셋은 120만 개의 이미지를 사용하였다. NVIDIA GTX 580 3GB GPU 두 개를 사용했을 때 훈련은 5~6일 소모되었다.

## 6. Result  

ILSVRC-2010에서의 ImageNet을 테스트 셋으로 사용했을 때 AlexNet은 top-1과 top-5 error rate가 37.5%, 17.0% 였다.  
|Model|Top-1|Top-5  
|:----|:----|:----  
|Sparse Coding | 47.1% | 28.2%  
|SIFT + FVs | 45.7% | 25.7%  
|CNN | 37.5% | 17,0%  
> Table 2. ILSVRC-2010 ImageNet(Test set)  

ILSVRC-2012에서의 ImageNet을 validation과 테스트 셋으로 사용했을 때 AlexNet은 top-1과 top-5 error rate가 36.7%와 15.3% 였다.  
|Model|Top-1(Val)|Top-5(Val)|Top-5(Test)  
|:----|:---------|:---------|:----------  
|SIFT + FVs | - | - | 26.2%  
|1 CNN | 40.7% | 18.2% | -  
|5 CNNs | 38.1% | 16.4% | 16.4%  
|1 CNN* | 39.0% | 16.6% | -  
|7 CNNs* | 36.7% | 15.4% | 15.3%  
> Table 3. ILSVRC-2012 ImageNet(Validation & Test set)  
>> asterisk(*)은 ImageNet 2011로 pre-train 되어있음을 의미한다.

## 7. Implementation  

### 1. Tools / Libraries  

AlexNet을 구현함에 있어 준비해야하는 라이브러리는 다음과 같다.  

* Tensorflow  
> 머신러닝 모델을 훈련이나 구현등에 사용하는 오픈소스 플랫폼이다.  

* Keras  
> 신경망 모델을 구현하고 CPU와 GPU를 이용하여 실행하기위해 사용하는 오픈소스 라이브러리다.  

* Matplotlib  
> 시각화를 위한 파이썬 툴이다.  

```Python  
>>> import tensorflow as tf  
>>> from keras.datasets import cifar10  
>>> from keras.models import Sequential  
>>> from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization  
>>> import numpy as np  
```  

### 2. Dataset  

CIFAR-10 데이터셋은 10개의 클래스를 가지는 **32x32 픽셀** 로 구성된 60000개의 컬러 이미지이며, 하나의 클래스마다 6000개의 이미지가 포함된다.  
이 중 50000개가 훈련 이미지로 사용되며 나머지 10000개가 시험 이미지로 사용된다.  

<img alt="cifar10.png" src="https://user-images.githubusercontent.com/43739827/91971486-3a566e00-ed54-11ea-88b1-95d1f122f341.PNG"></img>  
> Fig 3. CIFAR-10 Data Set  

CIFAR-10에 대한 더 구체적인 정보는 AlexNet의 1저자인 [Akex Jrizhevsky의 홈 페이지](https://www.cs.toronto.edu/~kriz/cifar.html)에서 확인할 수 있다.  

먼저 **keras.datasets** 로 불러온 cifar10을 train과 test로 나눈다.  

```Python  
>>> (x_train, y_train), (x_test, y_test) = cifar10.load_data()
```  

훈련 데이터 셋에서 후에 있을 모델의 파라미터를 검증하기위해 일부를 validation set으로 나눈다.  

```Python  
>>> x_val, y_val = x_train[:5000], y_train[:5000]  
>>> x_train, y_train = x_train[5000:], y_train[5000:]
```  

데이터 셋을 텐서플로우 함수나 메소드에서 사용하기 위해서는 **tf.data.Dataset** API를 사용해 변환할 필요가 있다.  
여기서는 **tf.data.Dataset.from_tensor_slices** 메소드를 사용하여 train, test, validation 데이터 셋으로 나누어 반환하였다.  

```Python  
>>> train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))  
>>> test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))  
>>> val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
```

먼저 훈련 데이터 셋과 레이블이 제대로 분리되었는지 시각화하여 확인하였다.  

```Python  
>>> Class_label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
              'frog', 'horse', 'ship', 'truck']  
>>> %matplotlib inline  
>>> from matplotlib import pyplot as plt  
>>> plt.figure(figsize=(20, 20))  
## take() 메서드는 해당 데이터 내에서 지정한 갯수만큼을 반환한다.  
>>> for i, (image, label) in enumerate(train_dataset.take(5)):  
>>>    ax = plt.subplot(5,5,i+1)  
>>>    plt.imshow(image)  
## label.numpy()[0]을 하는 이유는 단순히게 넘파이 변환을 하면 넘파이 배열이되어 Class_label이라는 리스트 내에서 요소를 불러올 수가 없다.   
>>>    plt.title(Class_label[label.numpy()[0]])  
>>>    plt.axis('off')  
>>>    print(label.numpy().astype)  
```  

<img alt="Visualize.png" src="https://user-images.githubusercontent.com/43739827/92082086-0dab6080-edff-11ea-8458-f4c74c56bbfd.png"></img>  
> Fig 4. CIFAR-10 Training data set visualization  


이제 AlexNet에 이미지를 입력하기 위해 기존 CIFAR-10의 크기인 32x32를 227x227로 변환한다. 논문에서는 입력 이미지의 크기가 224x224이라고 언급되어있으나  
이는 오타인것으로 보인다. 이것에 대한 설명으로는 [Learn OpenCV-Understanding AlexNet](https://www.learnopencv.com/understanding-alexnet/)의 Input 챕터에서 확인할 수 있다.  

이미지를 전처리하기위해 **process_image**  라는 함수를 생성하였다.  

```Python  
## tf.image는 이미지의 전처리나 부호화-암호화 연산등을 수행하는 함수다.  
## 각 이미지를 정규화하여(per_image_standardization) 학습을 빨리하고 local optimum의 위험을 줄인다.  
## image의 resize 함수는 입력 이미지의 크기를 변형하고 텐서로 변환한다. 이 때의 텐서 type은 float32 이다.
>>> def process_image(image, label):  
>>>    image = tf.image.per_image_standardization(image)
>>>    image = tf.image.resize(image, (227, 227))
>>>    return image, label
```  

### 3. Data/Input Pipeline  

먼저 데이터 셋을 **shuffle** 하기위해서는 데이터 셋들의 크기를 알 필요가 있다. **tf.data.experimental.cardinality** 는 데이터 셋의 크기를 반환하는 기능을 한다.  

```Python  
>>> tf.data.experimental.cardinality(train_dataset)
```  

```  
<tf.Tensor: shape=(), dtype=int64, numpy=45000>
```  

```Python  
>>> train_ds_size = tf.data.experimental.cardinality(train_dataset).numpy()  
>>> test_ds_size = tf.data.experimental.cardinality(test_dataset).numpy()  
>>> val_ds_size = tf.data.experimental.cardinality(val_dataset).numpy()  
>>> print("Training data size:", train_ds_size)  
>>> print("Test data size:", test_ds_size)  
>>> print("Validation data size:", val_ds_size)
```  

```  
Training data size: 45000  
Test data size: 10000  
Validation data size: 5000
```  

입력 데이터를 정제하기위해 세 가지의 과정을 거친다.  
1. 데이터 셋 내부 데이터 전처리  
2. 데이터 셋 shuffle  
3. 데이터 셋 내부 데이터 batch  

```Python  
>>> train_dataset = (train_dataset  
                    .map(process_image)  
                    .shuffle(buffer_size=train_ds_size)  
                    .batch(batch_size=32, drop_remainder=True))  
>>> test_dataset = (test_dataset  
                    .map(process_image)  
                    .shuffle(buffer_size=train_ds_size)  
                    .batch(batch_size=32, drop_remainder=True))  
>>> val_dataset = (val_dataset  
                    .map(process_image)  
                    .shuffle(buffer_size=train_ds_size)  
                    .batch(batch_size=32, drop_remainder=True))  
>>> print(train_dataset)  
>>> print(test_dataset)  
>>> print(val_dataset)  
```  

```
<BatchDataset shapes: ((32, 227, 227, 3), (32, 1)), types: (tf.float32, tf.uint8)>  
<BatchDataset shapes: ((32, 227, 227, 3), (32, 1)), types: (tf.float32, tf.uint8)>  
<BatchDataset shapes: ((32, 227, 227, 3), (32, 1)), types: (tf.float32, tf.uint8)>
```  

#### Shuffle  

훈련하기 전에 데이터를 shuffle 하는것은 머신 러닝에서 전통적인 과정이다. 데이터 병합을 수행할 때 일반적으로 차례대로 축적된 이미지 혹은 데이터는 같은 클래스를 가진다.  
즉, 데이터가 클래스별로 정렬되어 있는 것이다. 이러한 데이터를 아무런 처리없이 훈련하게 되면 신경망은 가장 먼저 나타나는 클래스의 패턴에 대해 집중적으로 학습하게 된다.  
이러한 이유로 모델의 분산이 커지게 되기 때문에 shuffle을 하여 각 클래스를 대표하는 데이터가 에포크당 한 번씩은 나타나도록 한다.  

* Shuffle을 하게 되면 얻게되는 두 가지 장점  
1. 데이터 셋 내에는 훈련 데이터의 각 데이터가 망에 독립적으로 영향을 미칠 수 있도록 충분한 분산이 존재한다. 따라서 데이터 셋의 부분집합이 아닌 전체 데이터에 더 일반화 할 수 있는  
망을 구축할 수 있게된다.  

2. validation set은 훈련 셋에서 추출하는 것이므로 shuffle을 하지 않고 추출하게 되면 훈련 셋의 전체 클래스를 추출하는 것이 아닌 특정한 클래스만을 가지게 될 수 있다.  

#### Batch  

망을 훈련하는 방법에는 두 가지가 존재한다.  

1. 모든 훈련 데이터를 망에 훈련시킨다.  
2. 훈련 데이터를 8, 16, 32, 64처럼 일부분으로 작게 나누고(**Batch**), 각 반복마다 하나의 배치가 망에 사용된다.  

첫 번째 방법은 작은 데이터 셋을 활용할 경우 잘 작동하지만 데이터 셋의 크기가 크다면 많은 메모리 자원을 소비하게 되며 이는 메모리 부족(Out of memory) 현상을 초래한다.  
두 번째 방법은 큰 데이터 셋을 이용해 학습할 때  메모리를 효율적으로 관리하는 전통적인 학습 방법이다. 배치 사이즈를 지정하면 정해진 크기만큼의 메모리를 사용해 반복한다.  

### 4. Model Implementation  

AlexNet [논문에 나와있는 구조](https://github.com/galaxy1014/DL_paper_summary/tree/master/Neural%20Network/AlexNet#35-overall-architecture)를 확인하고 이와 동일한 형태로 구현하였다.  

```Python  
>>> model = Sequential()  

# first layer  
>>> model.add(Conv2D(96, (11, 11), strides=4, activation='relu',
                 input_shape=(227,227,3)))  
>>> model.add(BatchNormalization())  
>>> model.add(MaxPooling2D(pool_size=(3, 3), strides=2))  

# second layer  
>>> model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))  
>>> model.add(BatchNormalization())  
>>> model.add(MaxPooling2D(pool_size=(3, 3), strides=2))  

# third layer  
>>> model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))  

# forth layer  
>>> model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))  

# fifth layer  
>>> model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))  
>>> model.add(MaxPooling2D(pool_size=(3, 3), strides=2))  

>>> model.add(Flatten())  
>>> model.add(Dense(4096, activation='relu'))  
>>> model.add(Dropout(0.5))  
>>> model.add(Dense(4096, activation='relu'))  
>>> model.add(Dropout(0.5))  
>>> model.add(Dense(1000, activation='softmax'))  

>>> opt = tf.keras.optimizers.SGD(lr=0.001, decay=5e-5, momentum=0.9)  
>>> model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])  
>>> model.summary()
```
> 논문에서는 정규화 방법으로 LRN을 사용했으나 요즘의 딥러닝 알고리즘에선 LRN을 사용하는 대신 BatchNormalization을 사용하고 있다.  

```  
Model: "sequential"  
_________________________________________________________________  
Layer (type)                 Output Shape              Param #     
=================================================================  
conv2d (Conv2D)              (None, 55, 55, 96)        34944       
_________________________________________________________________  
batch_normalization (BatchNo (None, 55, 55, 96)        384         
_________________________________________________________________  
max_pooling2d (MaxPooling2D) (None, 27, 27, 96)        0           
_________________________________________________________________  
conv2d_1 (Conv2D)            (None, 27, 27, 256)       614656      
_________________________________________________________________  
batch_normalization_1 (Batch (None, 27, 27, 256)       1024        
_________________________________________________________________  
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 256)       0           
_________________________________________________________________  
conv2d_2 (Conv2D)            (None, 13, 13, 384)       885120      
_________________________________________________________________  
conv2d_3 (Conv2D)            (None, 13, 13, 384)       1327488     
_________________________________________________________________  
conv2d_4 (Conv2D)            (None, 13, 13, 256)       884992      
_________________________________________________________________  
max_pooling2d_2 (MaxPooling2 (None, 6, 6, 256)         0           
_________________________________________________________________  
flatten (Flatten)            (None, 9216)              0           
_________________________________________________________________  
dense (Dense)                (None, 4096)              37752832    
_________________________________________________________________  
dropout (Dropout)            (None, 4096)              0           
_________________________________________________________________  
dense_1 (Dense)              (None, 4096)              16781312    
_________________________________________________________________  
dropout_1 (Dropout)          (None, 4096)              0           
_________________________________________________________________  
dense_2 (Dense)              (None, 1000)              4097000     
=================================================================  
Total params: 62,379,752  
Trainable params: 62,379,048  
Non-trainable params: 704  
_________________________________________________________________
```  

### 5. Training and Results  

모델을 학습하는 과정에서 callback 함수로 **EarlyStopping** 을 사용하여 더 이상 validation loss의 개선이 확인되지 않는다면 학습을 끝내도록 하였다.  

```Python  
>>> from keras.callbacks import EarlyStopping  
# val_loss를 대상으로 관측하며 최소값으로부터 값이 커진다면 반복을 취소하며 최소 50번의 반복을 지켜본다.
>>> early_stopping = EarlyStopping(monitor='val_loss', verbose = 1, mode='min', patience=50)
>>> hist = model.fit(train_dataset, validation_data=val_dataset, validation_freq=1,
                 epochs=100, batch_size=132, callbacks=[early_stopping])
```  

학습된 모델에 대해서 train_loss, val_loss, train_acc, val_acc의 변화를 그래프로 확인한다.  

```Python  

```
