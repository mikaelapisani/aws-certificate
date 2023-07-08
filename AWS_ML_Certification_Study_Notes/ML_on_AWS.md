# ML on AWS

## **Amazon Forecast**

Combines time series with additional variables 

AutoML choose best model for your time series data 

works with any time series 

Concepts: datasets groups, predictors, and forecasts 

### **ARIMA**

**stationary** time series

ARIMA acts like a filter to separate the signal from the noise, and then extrapolates the signal in the future to make predictions.

**simple** datasets

### **Amazon Forecast CNN-QR**

(**CNNs**) - Quantile Regression 

best for **large** datasets 

accepts related historical time series data 

### **ETS**

- The ETS algorithm is especially useful for datasets with **seasonality** and other prior assumptions about the data
- exponential smoothing
- **simple** datasets <100 time series

### **Prophet**

- Contain an **extended time period** (months or years) of detailed historical observations (hourly, daily, or weekly)
- Have multiple **strong seasonalities**
- Includ**e previously known** important, but irregular, events
- Have **missing** data points or large outliers
- Have **non-linear** growth trends that are approaching a limit

### **NPTS**

- **non-parametric** time series
- good for **sparse** data or containing many 0s
- Has variants for **seasonal** / climatological forecast

### **deepAR**

Amazon Forecast DeepAR+ is a supervised learning algorithm for forecasting scalar **(one-dimensional)** time series using recurrent neural networks (**RNNs**).

## **Amazon Fraud Detector**

Build, deploy, and manage fraud detection models

upload your historical data 

build custom model from a template 

exposes API 

applications: try before you buy , new accounts, online payments 

## Amazon CodeGuru

automated code reviews , finds things that you can improve , resource leaks, supports Java and Python 

## **Amazon Comprehend**

Understand valuable insights from **text** within documents. 

For example create a classifier through CreateDocumentClassifier 

**NLP**

**extract** **key phrases** (relevant phrases in a sentences), **entities**, language detection, **sentiment** analysis, syntax  

## **Amazon Lex**

Build **chatbots** with conversational AI

utterances - invoke to fulfill the intent (I want to order a pizza)

lambda functions are invoked to fulfill the intent 

**slots** specify extra information needed by the intent (toppings, pizza size) 

slot value (Pepperoni) 

Supports voice and text input 

## **Amazon Transcribe**

convert **speech to text**

supports streaming audio 

speaker identification 

channel identification 

custom vocabulary 

automatic language identification

## **Amazon Translate**

**machine translation service** 

csv or TMX format 

appropriate for proper names, brand names, etc 

## **Amazon Polly**

convert **text to audio**

speech marks 

SSML alternative to plain text - emphasize , pronunciation 

many voices and languages 

Lexicons allow you to map specific words and phrases to a specific pronunciation.

## **Amazon Textract**

extract printed text, handwriting, and data from any document

**OCR**

## **Amazon Rekognition**

offers pre-trained and customizable **computer vision** (CV) models

facial analysis (is a male of female, seems to be smiling, age range, wearing glasses), **celebrity** recognition, face comparison, text in image, video analysis 

images from s3 

video must come from Kinesis Video Stream 

## Amazon Personalize

**recommendation** engine 

API - feed data, explicit Avro schema, GetRecommendations , GetPersonalizedRanking 

datasets: users, items, interactions

recipes: what kind of model - USER_PERSONALIZARTION, PERSONALIZED_RANKINGS, RELATED_ITEMS 

Solutions: train model, optimize for relevance, Hyperparameter HPO 

Campaigns: deploys solution version , deploys capacity 

hidden_dimension

bptt - backpropagation 

recency_mask  weights recent events 

min/max_user_history_length_percentile (when data is not clean filter out robots, outliers)  

exploration_weight 0-1 controls relevance 

exploration_item_age_cut_off - how far back in time you go 

## **Amazon SageMaker**

 framework agnostic for develop and deploy ML models (see [SageMaker](SageMaker.md) section)

## DeepRacer

Reinforcement learning 

## Amazon Lookout

- **equipment** metrics vision
- detect abnormalities from sensors
- monitor **metrics**

## Amazon monitor

**end-to-end** system for monitoring industrial **equipment** and predictive maintenance 

## **Amazon Kendra**

**search** engine for internal support 

## **Amazon Augmented AI (AI2I)**

when using Amazon Rekognition or Textract, you can use Amazon Augmented AI to **get human review** of low-confidence predictions or random prediction samples.

## **AWS DeepLens**

**video camera** with pre-trained models 

## Contact Lens for Amazon Connect

for customer support call centers , sentiment analysis , categorize calls automatically

## TorchServe

model serving framework for pytorch 

## AWS Neuron

inference chips designed for ML inference 

## AWS Panorama

Computer vision at the edge 

brings computer vision to your IP cameras 

## AWS DeepComposer

AI powered keyboard 

for educational proposes 

[Continue to Modeling section](Modeling.md)