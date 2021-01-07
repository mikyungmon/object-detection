# CNN

- CNN이란? 

Convolutional Neural Network의 약자로 합성곱 신경망이라 부르며 딥러닝에서 주로 영상 데이터를 처리할 때 쓰인다.

- CNN이 유용한 이유? 

 이에 대한 답은 일반 DNN(Deep Neural Network)의 문제점에서부터 출발한다. 일반 DNN은 기본적으로 1차원 형태의 데이터를 사용하기 때문에 (예를들면 1028x1028같은 2차원 형태의)이미지가 입력값이 되는 경우, 이것을 flatten시켜서 한줄 데이터로 만들어야 하는데 
 
 이 과정에서 이미지의 공간적/지역적 정보(spatial/topological information)가 손실되게 된다. 또한 추상화과정 없이 바로 연산과정으로 넘어가 버리기 때문에 학습시간과 능률의 효율성이 저하된다.

 이러한 문제점에서부터 고안한 해결책이 CNN이다.
 
 # 1-1 Convolution
 
 Convolution의 사전적 정의는 합성곱이다. 사실 Convolution은 처음 등장한 개념이 아니라 CNN이 등장하기 한참 전부터 이미지 처리에서 사용되었던 개념이다.
 
 ![Convolution](/images/15.PNG)

빨간 박스는 필터가 적용될 영역이고 stride(빨간색 박스인 filter를 몇칸씩 이동할 것인지)에 따라 이미지의 feature map가 만들어진다. 

filter(kernel)의 구성에 따라 이미지의 특징을 뽑을 수 있다.

# 1-2 Filter(Kernel)

 ![Kernel](/images/2.PNG) 
 
 Filter라는 것을 통해서 어떻게 이미지의 feature를 뽑을 수 있는지 예시를 통해 봐보자
 
 이미지 처리에 자주 등장하는 sobel filter는 이미지의 가로,세로 feature를 뽑아낼 수 있는 필터이다.
 
  ![original](/images/3.PNG)
 
  ![result](/images/4..PNG)
  
  원본 이미지에 두 개의 필터를 차례대로 적용시키면 위와 같은 결과를 확인할 수 있다. 왼쪽은 sobel -x를 적용, 오른쪽은 sobel -y를 적용한 것이다.
  
  각 이미지의 특징을 보니 왼쪽은 세로선이 detect되고 오른쪽은 가로선이 detect된 것을 확인할 수 있다.
  
  위의 두 개의 결과를 합치면 아래와 같이 원본 사진의 feature를 확인할 수 있다.
  
  ![plus](/images/5.PNG)
  
  이와 같이 CNN에서도 여러 개의 filter를 이용해 이미지의 세부 특징을 추출해서 학습할 수 있다.
  
  위와 같은 이미지 처리에서는 sobel필터와 같이 유명한 filter들을 사용자가 직접 찾아서 사용해야 했지만, CNN은 신경망에서 학습을 통해 자동으로 적합한 필터를 생성해 준다는 것이 특이사항이다.
  
  # 1-3 Channel
  
  ![channel](/images/6.PNG)
  
  우리가 보통 생각하는 color이미지는 red , green, blue 채널이 합쳐진 이미지이다.
  
  즉, 하나의 color이미지는 3개의 Channel로 구성되어 있다.
  
  보통은 연산량을 줄이기 위해(오차를 줄이기 위해) 전처리에서 이미지를 흑백(channel이 1)으로 만들어 처리한다.
  
  흑백의 경우 다음 그림과 같이 처리된다.
  
   ![black](/images/7.PNG)
   
   color이미지의 경우에는 다음 그림과 같이 처리된다.
   
   ![color](/images/8.PNG)
   
   ![color](/images/14.PNG)
   
   Multi Channel CNN에서 주의해서 보아야 할 점은 
   
   1. input 데이터의 channel수와 filter의 channel수는 같아야 한다. 예를 들어 위의 그림과 같이 input이 channel 3이라면 filter도 channel이 3이어야 한다.)
   
   2. input 데이터의 channel수와 관계 없이 filter의 개수만큼 output 데이터가 나온다.
   
   # 1-4 Padding
   
   ![convolution](/images/9.PNG)
   
   1-1에서 본 convolution을 떠올려보면 convolution 레이어에서는 filter와 stride의 작용으로 feature map의 크기는 input 데이터보다 작다.
   
   이렇게 input 데이터보다 output 데이터가 작아지는 것을 방지하는 방법이 padding이다.
   
   ![convolution](/images/10.PNG)
   
   위와 같이 0으로 둘러싸서 input 데이터보다 output 데이터가 작아지는 것을 방지하는 padding을 zero padding이라고 한다. 단순히 0을 덧붙였으므로 특징에는 영향을 미치지 않는다.
   
   이렇게 padding을 하게 되면 convolution을 하더라도 크기가 작아지지 않는다.
   
   padding에는 두 가지 옵션이 있다.
   
   1. valid - padding 0을 의미. 즉, input보다 output의 크기가 작아진다.
   
   2. same - padding이 존재하여 input과 output의 크기가 같아진다.
   
   # 1-5 Pooling
   
   이미지의 크기를 계속 유지한 채 Fully Conneted Layer로 가게 된다면 연산량이 기하급수적으로 늘어날 것이다.
   
   적당히 크기도 줄이고, 특정 feature을 강조할 수 있어야 하는데 그 역할을 pooling 레이어에서 하게 된다.
   
   방법은 세 가지가 있다.
   
   ![pooling](/images/11.PNG)
   
   1. max pooling
   
   2. average pooling
   
   3. min pooling
   
   일반적으로 pooling의 크기와 stride크기를 같게 설정하여 모든 원소가 한번씩은 처리가 되도록 설정한다.
   
   CNN에서는 주로 max pooling을 사용한다. 이는 뉴런이 가장 큰 신호에 반응하는 것과 유사하다고 한다.
   
   이렇게 하면 노이즈가 감소하고 속도가 빨라지며 영상의 분별력이 좋아진다.
   
   특징은 다음과 같다.
   
   1. 학습 대상 파라미터가 없다.
   
   2. pooling 레이어를 통과하면 행렬의 크기가 감소한다.
   
   3. pooling 레리어를 통과해도 채널의 수는 변경 없다.
   
   # 전체 구조
   
   ![structure](/images/12.PNG)
   
   ![structure](/images/16.PNG)
   
   지금까지 CNN의 구성 요소들을 보았고 이는 CNN의 전체적인 구조 그림이다.
   
   CNN은 위 이미지와 같이 이미지의 특징을 추출하는 부분과 클래스를 분류하는 부분으로 나눌 수 있다. 특징 추출 영역은 Convolution Layer와 Pooling Layer를 여러 겹 쌓는 형태로 구성된다.
   
   
   Convolution Layer는 input 데이터에 필터를 적용 후 활성화 함수를 반영하는 필수 요소이다. Convolution Layer 다음에 위치하는 Pooling Layer는 선택적인 레이어이다. 
   
   CNN 마지막 부분에는 이미지 분류를 위한 Fully Connected 레이어가 추가된다. 이미지의 특징을 추출하는 부분과 이미지를 분류하는 부분 사이에 이미지 형태의 데이터를 배열 형태로 만드는 Flatten 레이어가 위치하는 것이다.
   
   ![structure](/images/17.PNG)
   
   위 CNN 그래프에서 특이한 점중 하나는 Fully connected  Softmax 함수 중간에 Dropout layer (드롭아웃) 라는 계층이 있는 것을 볼 수 있는데 이는 overfitting을 막기위한 방법으로 뉴럴 네트워크가 학습중일 때 랜덤하게 뉴런을 없애서 학습을 방해함으로써 학습이 학습용 데이터에 치우지는 현상을 막는 역할을 한다.
   
   **특징 추출 단계(feature extraction)**
   
   - convolution layer - 필터를 통해 이미지의 특징을 추출
   - pooling layer - 특징을 강화시키고 이미지의 크기를 줄임
   
   convolution과 pooling을 반복하면서 이미지의 특징을 추출한다.
   
   **이미지 분류 단계(classification)**
   
   - flatten layer - 데이터 타입을 fully connected 형태로 변경. input 데이터의 shape 변경만 수행
   - softmax layer - classification 수행
   
   # 파라미터
   
   - convolution filter의 갯수
   
   - filter의 사이즈
   
   - padding 여부
   
   - stride
   
  
   # 요약
   
   - CNN(Convolutional Neural Network)은 이미지의 공간 정보를 유지하면서 인접 이미지와의 특징을 효과적으로 인식하고 강조하는 방식으로 이미지의 특징을 추출하는 부분과 이미지를 분류하는  부분으로 구성된다. 
   
   - 특징 추출 영역은 Filter를 사용하여 공유 파라미터 수를 최소화하면서 이미지의 특징을 찾는 Convolution 레이어와 특징을 강화하고 모으는 Pooling 레이어로 구성된다.
   
   - 이미지의 특징을 추출하는 부분과 이미지를 분류하는 부분 사이에 이미지 형태의 데이터를 배열 형태로 만드는 Flatten 레이어인 Fully Connected 레이어가 존재한다.
   
   -  Fully connected 레이어와 Softmax 함수 중간에 Dropout layer (드롭아웃) 라는 계층이 있는 것을 볼 수 있는데 이는 overfitting(오버피팅)을 막기위한 방법이다.
   
   - CNN은 Filter의 크기, Stride, Padding과 Pooling 크기로 출력 데이터 크기를 조절하고, 필터의 개수로 출력 데이터의 채널을 결정한다.
   
   
 
   
   
  
  
