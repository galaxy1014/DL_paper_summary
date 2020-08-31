# AlexNet  

 **AlexNet** 은 2012년 ILSVRC(ImageNet Large Scale Visual Recognition Challenge)에서 가장 높은 점수를 얻은 CNN(Convolutional Neural Network)이다.  
논문의 1저자인 Alex Krizhevsky의 이름을 따서 AlexNet이라는 이름을 갖게 되었다.  

## 1. Introduction  

 논문의 저자들은 성능을 향상시키기위해 대용량의 데이터셋을 수집하고 일반적인 CNN 모델보다 더 강력한 모델을 구축하였으며, 과적합을 피하기위해 기존 모델들에서  
사용되지 않던 기술(dropout)을 사용하였다. 사용된 데이터셋은 **ImageNet** 으로 120만개의 고화질 이미지(high-resolution image)와 1000개의 각기 다른 클래스들로  
이루어져 있다.  

 AlexNet의 성능 비교대상은 비슷한 크기를 가지는 기존의 신경망(Neural Network)이다. CNN은 신경망과 비교해 더 적은 연결(**sparse connection**)과  
파라마티러를 가져 훈련(train)하기 쉽다는 이점이 있지만 대용량의 고화질 이미지를 데이터셋으로 사용할 때 시간이 오래 걸린다는 단점이 존재했다. 이를 해결하기위해  
AlexNet은 **두 개** 의 **GTX 530 3GB GPU** 를 **병렬 연결** 하여 훈련했으며 5~6일이라는 시간이 소모되었다.  
