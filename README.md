# <div align=center> Performance Comparison of CNN for COVID-19 Abnormalities Identification using Chest Radiographs </div>

<div align=right> <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/HJK02130/Performance-Comparison-of-CNN-for-COVID-19-Abnormalities-Identification-using-Chest-Radiographs?style=flat-square"> <img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/HJK02130/Performance-Comparison-of-CNN-for-COVID-19-Abnormalities-Identification-using-Chest-Radiographs?style=flat-square"> <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/HJK02130/Performance-Comparison-of-CNN-for-COVID-19-Abnormalities-Identification-using-Chest-Radiographs?style=flat-square"> </div>

### Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Usage](#usage)
4. [Architecture](#architecture)
5. [Result](#result)
6. [Conclusion](#conclusion)
7. [Reference](#reference)
8. [Developer](#developer)

### Overview
[[Kaggle : SIIM-FISABIO-RSNA COVID-19 Detection]](https://www.kaggle.com/competitions/siim-covid19-detection/)<br/>

The most commonly used main diagnostic method for COVID-19 is Polymerase Chain Reaction (PCR). PCR is a low-sensitivity diagnostic method, and the time required to diagnose is long and the price of diagnostic is burdensome. And there is a safety problem that medical personnel may become infected. In this situation, SIIM partnered with RSNA to hold a machine learning COVID-19 challenge to diagnose COVID-19 using chest radiographs. Like other pneumonias, COVID-19 infection causes inflammation and fluid in the lungs, so chest radiographs can confirm lung abnormalities, and diagnostic using chest radiographs can make up for the weak point of disadvantages of PCR. In this project, we implemented Convolutional Neural Network(CNN) based classification using RSNA data provided by Kaggle and compared the performance of models that using different data and model. We reduced the 4 image labels of RSNA data that including chest Radiographs into 2(positive/negative) to implement binary classification algorithm. To evaluate the performance of the COVID-19 classification model, we used accuracy, precision, recall, and F1 score as evaluation metrics. As a result of performance evaluation, The pre-trained VGGNet19 model without augmentation showed the highest performance with Accuracy 0.81, Precision 0.54, Recall 0.77, and F1-Score 0.63 among the CNN architecture based deep learning models. It is expected that performance evaluation of CNN based COVID-19 classification model using chest radiographs will be helpful in diagnosis and treatment by supplementing the shortcomings of PCR through improving the accuracy of predicting COVID-19 inflection.
<br/><br/>
현재 COVID-19의 주진단방법은 Polymerase Chain Reaction(PCR)이다. PCR은 민감도가 낮은 진단 검사 방법이며 감염여부 진단을 위해 필요한 시간이 길고, 부담스러운 진단 비용이 발생한다. 또한, 의료인이 감염될 수 있는 안전성 문제가 있다. 이러한 상황에서 SIIM은 RSNA와 파트너쉽을 맺고 Kaggle에서 머신러닝 COVID-19 challenge를 개최하였고, COVID-19를 영상학적으로 진단하고자 하였다. 다른 폐렴과 마찬가지로 COVID-19 감염 또한 폐에 염증과 체액을 유발하기 때문에 Chest X-ray 사진으로 폐의 이상을 확인할 수 있으며, 영상학적인 진단을 통해 앞서 말한 PCR의 단점을 보완할 수 있다. 본 프로젝트에서는 Kaggle에서 제공하는 RSNA 데이터를 사용하여 Convolutional Neural Network(CNN) 기반의 분류 알고리즘을 구현하고 서로 다른 학습 모델과 서로 다른 데이터를 사용한 알고리즘 간의 성능을 비교하였다. 흉부 X-ray 영상을 포함한 RSNA 데이터의 4가지 label을 양성 및 음성의 2가지로 축소하여 이진 분류하는 알고리즘을 구현하였으며, Augmentation 여부 및 딥러닝 모델 종류에 따른 결과를 비교·분석하였다. COVID-19 분류 모델의 성능 평가를 위해 성능 평가 지표로 정확도, 정밀도, 재현율, F1 스코어를 사용하였다. 성능 평가 결과 테스트한 CNN 아키텍처(architecture) 기반 딥러닝 모델중 데이터에 Augmentation을 적용하지 않은 Pre-trained VGGNet19 모델이 Accuracy 0.81, Precision 0.54, Recall 0.77, F1-Score 0.63으로 가장 높은 성능을 보였다. 흉부 X-ray 영상을 이용항 딥러닝 기반 COVID-19 분류 모델의 성능 평가를 통해 뇌연령 예측 모델의 정확도를 개선하고 한계점을 해결한다면 PCR의 단점을 보완하고 피진단자의 COVID-19 예측 정확도를 향상시켜 진단과 치료에 도움이 될 수 있을 것으로 기대한다.


### Requirements
+ Python 3.8
+ Pytorch 1.3.1
+ GPU

### Usage
src/<br/>
├── aug_image.ipynb<br/>
    ├── eda.ipynb<br/>
    ├── main_Aug.ipynb<br/>
    ├── main_NoAug.ipynb<br/>
    ├── preprocessing.ipynb<br/>
    └── split.ipynb<br/>

### Architecture
<div align=center>  <img src="./img/architecture.png"> </div>

<br/>

+ Data
	+ [RSNA Data](https://www.kaggle.com/competitions/siim-covid19-detection/data)
		<br/>

+ Preprocessing
	+ Resize Chest Radiographs to 512 x 512
	+ Split Dataset
		+ Train : Validation : Test = 4 : 1 : 1<br/>
	+ labeling
		+ Positive label : Typical Appearance, Indeterminate Appearance labeled data
		+ Negative label : Negative pneumonia labeled data
	+ Elminate Atypical Appearance labeled data
	+ Eliminate inconsistency data

<br/>

+ Model

||Base Model|Model I|Model II|
|:---:|:---|:---|:---|
|Image Size|512x512|512x512|512x512|
|Classification|Binary|Binary|Binary|
|Augmentation|-|crop, blur, jitter|-|
|Model|VGGNet|VGGNet|EfficientNet|

<br/>

+ Modeling setting
	+ Epoch : 400
	+ Batch size : 32
	+ Learning rate : 0.001
	+ Model optimizing based on loss of validation set
	+ Early Stopping : Patient 

### Result
+ Performance of each model in Test Set

||Accuracy|Precision|Recall|F1-score|
|:---:|:---|:---|:---|:---|
|Base Model (VGGNet19)|0.81|0.54|0.77|0.63|
|Model I (VGGNet19, Augmentation)|0.79|0.53|0.74|0.62|
|Model II (EfficientNet-B4)|0.76|0.41|0.68|0.51|
<br/>
The pre-trained VGGNet19 model without augmentation showed the highest performance with Accuracy 0.81, Precision 0.54, Recall 0.77, and F1-Score 0.63 among the CNN architecture based deep learning models.

### Conclusion
Since the process of identifying a better algorithm through performance comparison is a process of obtaining better results through more various attempts, so this is meaningful for the next step. Comparing the performance of more various deep learning models and improving the limitations is expected to allow us to build a COVID-19 classification deep learning model using chest radiographs data with higher accuracy than our result. improved algorithms based on this study could be used not only for COVID-19 but also for other lung diseases.

### Reference
[1] Kaggle. (2022). SIIM-FISABIO-RSNA COVID-19 Detection. https://www.kaggle.com/competitions/siim-covid19-detection<br/>
[2] Time Traveler. (2021). 1. 왜 COVID-19을 영상학적으로 판단하나요?. https://89douner.tistory.com/251<br/>
[3] Mohamed Elgendi, Muhammad Umer Nasir, Qunfeng Tang, David Smith, John-Paul Grenier, Catherine Batte, Bradley Spieler, William Donald Leslie, Carlo Menon, Richard Ribbon Fletcher, Newton Howard, Rabab Ward, William Parker, Savvas Nicolaou. (2021). The Effectiveness of Image Augmentation in Deep Learning Networks for Detecting COVID-19: A Geometric Transformation Perspective, 8, https://doi.org/10.3389/fmed.2021.629134<br/>
[4] Incredible.AI. (2020). Pytorch Image Augmentation. http://incredible.ai/pytorch/2020/04/25/Pytorch-Image-Augmentation/<br/>
[5] Pytorch. (n.d.). Multi-GPU Example. https://tutorials.pytorch.kr/beginner/former_torchies/parallelism_tutorial.html

### Developer
Hyunji Kim, Yoonjeong Kang, Minjeong Seo, Byeongjoon Yoo, Miran Jo <br/>
Hyunji Kim <a href="mailto:hjk02130@gmail.com"> <img src ="https://img.shields.io/badge/Gmail-EA4335.svg?&style=flat-squar&logo=Gmail&logoColor=white"/> </a> 
[<img src="https://img.shields.io/badge/Notion-000000?style=flat-square&logo=Notion&logoColor=white"/>](https://read-me.notion.site/Hyunji-Kim-9dbdb62cc84347feb85b3c58225bb63b)
	<a href = "https://github.com/HJK02130"> <img src ="https://img.shields.io/badge/Github-181717.svg?&style=flat-squar&logo=Github&logoColor=white"/> </a>
