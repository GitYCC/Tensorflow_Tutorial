# 實作Tensorflow系列教程

## 一步一腳印的學Tensorflow

我想完成一套Tensorflow教程，將Deep Learning一些重要的概念一一的點出來，並且使用Tensorflow來實現或驗證這些概念。本教程有三個面向我希望做到的，我希望觀念講解時可以深入淺出，我希望呈現程式碼時可以結構嚴謹，我希望可以完整呈現Tensorflow的實用面。

**本教程「網頁版」請至我的個人網站查看：[http://www.ycc.idv.tw/tag__實作Tensorflow/](http://www.ycc.idv.tw/tag__實作Tensorflow/)**

## Ch01 Simple Logistic Classification on MNIST

建立一個簡單的單層Neurel Network。

![Simple Neurel Network](https://raw.githubusercontent.com/GitYCC/Tensorflow_Tutorial/master/img/TensorflowTutorial.002.jpeg)

## Ch02 Build First Deep Neurel Network (DNN)

開始建立第一個Deep Learning，並仔細介紹Deep Learning的重要組成，包括：Hidden Layer、Activation Function、Mini-Batch Gradient Descent、Weight Regularization、Dropout和Optimizer。

![DNN](https://raw.githubusercontent.com/GitYCC/Tensorflow_Tutorial/master/img/TensorflowTutorial.003.jpeg)

## Ch03 Build First Convolutional Neurel Network (CNN)

介紹影像處理上最廣為人使用的Convolutional Neurel Network，引入Convolution Layer和Pooling Layer的概念，並在最後完成最簡單的CNN架構：LeNet5。

![CNN](https://raw.githubusercontent.com/GitYCC/Tensorflow_Tutorial/master/img/TensorflowTutorial.006.jpeg)

## Ch04 Autoencoder

建立一個DNN的Autoencoder，揭露Embedding Code的神奇效果，藉由壓縮與還原找出一個精簡描述一群數據的Embedding空間，在這空間上數據不需要人為給予Labels，機器會自行分類成為一個個合理的群體，所以Autoencoder可以用於Unsupervised Learning上。

![Autoencoder](https://github.com/GitYCC/Tensorflow_Tutorial/blob/master/img/TensorflowTutorial.007.jpeg?raw=true)

![Embedding Code](https://raw.githubusercontent.com/GitYCC/Tensorflow_Tutorial/master/img/04_output_9_0.png)

## Ch05 Word2Vec

介紹兩種Word2Vec模型：Skip-gram和CBOW，揭露Embedding Vector的神奇效果，利用壓縮上下文的關係，我們可以建立一個Embedding的空間，在這個空間語意相近的兩個字，它們的Embedding Vector也會彼此相似。

![word2vec](https://raw.githubusercontent.com/GitYCC/Tensorflow_Tutorial/master/img/TensorflowTutorial.008.jpeg)

![Embedding Vector](https://raw.githubusercontent.com/GitYCC/Tensorflow_Tutorial/master/img/05_output_13_0.png)

## CH06 Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM)

介紹具有時序性的Neurel Network—RNN，並點出一般簡易型的RNN因為共用權重以及等效於非常深的網路，會遇到的梯度爆炸與梯度消失問題。LSTM是另外一種型態的RNN，利用建立「長期記憶」來避免梯度消失問題，至於梯度爆炸問題則可以使用Gradient Clipping的手法解決。

![LSTM](https://github.com/GitYCC/Tensorflow_Tutorial/raw/master/img/TensorflowTutorial.012.jpeg)