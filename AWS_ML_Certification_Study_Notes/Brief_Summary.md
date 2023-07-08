# Brief Summary

## Storage

**EBS →** Volume for EC2 

**FSx →** File System 

**Redshift →** database, save large amount of data ,OLAP 

**S3 →** save files

different classes: 

- standard general propose (common: high latency and availability)
- IA (not access frequently, backup)
- one zone infrequent access (secondary backup, when access data is removed)
- glacier ( instant retrieval - by time from 5 min to 90 days, flexible retrieval: intervals until 90 days, deep archive - long time 180 days,  intelligent tier - change among classes as needed)

partition by folders 

encryption types: SS-E (handle by AWS), KMS (user knows key, more secure), SSE-C (user manage encryption) Client Side Encryption (outside AWS)  

**Avro →** streaming,  serialization  ****

**DynamoDB →** NOSql 

**RDS/Aurora →** relational database , OLTP 

**ElasticSearch →** index documents

**Elastic Cache →** cache 

## Data ingestion

**Glue →** ETL, convert csv to parquet , code python or scala 

**Kinesis →** 

Data Streaming → real time, streaming incoming data , not huge amount of data. data divided into shards  

1000/s or 1Mb 

not save data in S3

Analytics →  queries and ETLs, stream ML models 

Firehose → near real time, generally connects to Data Stream - saves data into database , transform json to parquet 

Video streaming → video 

**Data Pipeline →** migration inside AWS

**Step Functions →** visualization of workflows 

**DMS  →** migration data from outside of AWS (databases)

**DataSync →** migration data from outside of AWS (any object)

**AWS Batch →** run jobs in docker images 

**Data imputation** → KNN for numerical data , deep learning for categorical data 

## Data Analytics

**Athena →** sql queries 

**QuicksSight →** visualization , pre-built algorithms (Autonarrative times series, Cut) 

## AWS services

**Amazon Forescast →** time series 
Types: ARIMA (stationary, simple) , Auto() ,ETC-(simple), Prophet - (seasonality, long periods) 

**Fraud Detection →** detect froud 

**CodeGuru →** code review

**Comprehend  →** NLP , sentiment analysis

**lex →** chatbots, supports voice also

**Transcribe →** transform audio to text

**Polly →** transform text to audio

**Translate →** machine translation 

**Texextract →** OCR

**Rekognition →** computer vision

**Personalize  →** recommender system

**DeepRacer →** reinforcement learning 

**Lookout  →** IoT monitor devices 

**Monitor →** monitor end-to-end devices 

**Kendra →** search

**AI2I →** augmented human AI for Rekognition and Texextract 

**DeepLens →** camera ****

**ContactLens →** call center 

**TorchServer →** torch framework 

**AWS Neuron →** inference ****chip 

**Panorama →** computer vision at the edge 

**AWS Componser  →** music ****

## SageMaker

**Feature Store →** file system 

**Input →** csv generally , expects first column to be the target variable, and do not contain header. Other inputs: text, recordio protobuf 

**data Wrangler →** like glue but for SageMaker Studio 

**Bias →** Clarify ****

**JumpStart →** model zoo 

**Canvas →** no-code 

**SageMaker Pipelines →** DAGs , experiments 

 edge → when low latency - integration with **Neo** 

### **Algorithms:**

**DeepAR  →** time series , 1 dimension  

**XGboost →** boosting ****tree **,** num_trees, num_class. Decision tree, random forest, xgboost are not sensitive to scale , cpu 1.0 gpu 1.2 

**LightGBM →** gradient boost , csv

**CatBoost →** gradient boost  categorical features , csv, CPU only 

**AutoGluon-Tabular →** auto ml , csv 

**Kmeans →** clustering ****

**Factorization machines →** for classification and regression , high dimensional sparse datasets, recommender systems , recordio protobuf 

**Linear Learner →** L1, L2, float32 csv , normalized and shuffle training data 

**TabTransformer  →** tabular transformer , csv or text 

**KNN →** classifier ****

**Seq2Seq →** machine translation, recordio protobuf

**Object2Vec →** embedings, Word2Vec , documents 

**BlazingText →** Word2Vec, words  

**LDA →** topic modeling, CPU, recordio protobuf or csv 

**NTM →**  topic modeling, GPU 

**TextClassification →** csv, tensorflow, with small data 

**CutForest →** anomaly detection, CPU 

**IPInsights →** detect suspicious attacks

**Computer Vision →** object detection, image segmentation 

**RLEstimator →** reinforcement ****

**Folders  →** code, input/data, output, model, checkpoints 

**Hyperparameter tuning →** tune few parameters, short ranges, logarithmic, one training at a time   

**GridSearch →** search parameters , categorical categories , not need MaxTrainingJobs

**RandomSearch  →** search randomly 

**Bayesian Optimization  →** like regression problem 

**Hyperband  →** dynamically reallocate resources ****

**Docker →** set SAGEMAKER_PROGRAM env variable 

## MLOps

**CloudWatch  →** MONITOR, LOGS, METRICS

**CloudTrail →** AUDIT , API 

**GuardDuty →** monitor aws accounts 

**AWSBatch →** schedule batch computing

**EMR →** hadoop cluster , EMRBF to access S3 

**Instance type  →** p2, p3 - deep learning , M5 for CPU, spot instances save 90%, inference c2,c5, elastic inference (gpu), serverless inference