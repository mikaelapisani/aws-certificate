# Udemy questions   
   
- simplest way to automating deletion of old data in S3 - S3 Lifecycle   
- shards Data Streams - max throughput - 1 MB /s or 1000 messages / s   
- video data in real time - Kinesis Video Streams   
- Glue ETL - serverless Apache Spark platform   
- OLAP - Redshift   
- Which kind of graph is best suited for visualizing outliers in your training data?   
• Pair Plot   
**• Box and Whisker plot**   
• Tree map   
- What sort of data distribution would be relevant to flipping heads or tails on a coin?   
**• Binomial distribution**   
• Poisson distribution   
• Normal distribution   
- What is a serverless, fully-managed solution for querying unstructured data in S3?   
• Amazon Redshift   
• Amazon RDS   
**• Amazon Athena**   
- Which of the following is NOT a feature of Amazon Quicksight's Machine Learning Insights?   
**• Visualization of precision and recall**   
• Anomaly detection   
• Forecasting   
• Auto-narratives   
- Which imputation technique for missing data would produce the best results?   
• Mean or median replacement   
• Dropping   
• **MICE**   
- Your deep neural network seems to converge on different solutions with different accuracy each time you train it. What's a likely explanation?   
    - learning rate is too small   
    - the batch size is too small   
    - **the batch size is too large**   
- Your neural network's accuracy on its training data is increasing beyond the accuracy on test or validation data. What might be a valid thing to try to prevent this overfitting?   
• **Use dropout**   
• Add more layers to the model   
• Implement gradient checking   
- You're implementing a machine learning model for **fraud detection**, where most of your training data does not indicate fraud. The cost of a incorrectly identifying an actual fraudulent transaction is much higher than the cost of incorrectly identifying a non-fraudulent transaction. Which metric should you focus on for your model?   
• Precision   
• **Recall**   
• RMSE   
- You are building a deep learning model that learns to **classify** pictures of plants into their species. What would be an appropriate activation function at the output layer?   
• Leaky ReLU   
• **Softmax** converts outputs from your neural network to probabilities of a given classification.   
• Sigmoid (logistic) converts outputs to a probability between 0 and 1. While it may be useful for binary classification problems, it's not appropriate for multiple classifications.   
- You are developing a deep learning model to complete the words in an unfinished sentence. What might be an appropriate model type to start with?   
• CNN   
• MLP   
• **LSTM**   
   
Where does the training code used by SageMaker come from?   
• Jupyter notebooks   
**• A Docker image registered with ECR**   
• A GitHub repository   
   
Which SageMaker algorithm would be best suited for identifying topics in text documents in an unsupervised setting?   
• BlazingText can predict labels for sentences, but only if you've trained it in a supervised setting. It's not appropriate for working with entire documents.   
• Object2Vec   
• **LDA**   
   
Which SageMaker algorithm would be best suited for assigning pixels in an image to specific object classifications?   
**• Semantic Segmentation** Semantic Segmentation gives you a map of pixels to objects, and not just a list of objects in the image.   
****• Image Classification   
• Object Detection   
   
A Hyperparameter tuning job in SageMaker is using more time and resources than you are willing to spend. What might be one way to make it more efficient?   
• Try to optimize more hyperparameters concurrently.   
• Use linear scales for the hyperparameters   
**• Limit your hyperparameter ranges as much as possible**   
   
Which SageMaker feature automates algorithm selection, preprocessing, and model tuning?   
**• SageMaker Autopilot**   
• Automatic Model Tuning   
• SageMaker Model Monitor   
   
You have a dump of social media posts related to your company, and which to classify them based on sentiment. Which service could perform this task?   
• Amazon Augmented AI   
• Amazon TexTract   
**• Amazon Comprhend**   
   
Even though you are constantly feeding it new data, you're finding that your recommendations from Amazon Personalize are becoming less relevant over time. How might you address the issue?   
• Enable incremental training and solution versions   
**• Manually do a full retrain at least weekly**   
• Adjust the exploration_weight hyperparameter (although this parameter controls relevance won’t change over time)   
   
Your chatbot developed with Amazon Lex allows users to order a pizza. In the terminology of Amazon Lex, what would "Pepperoni" (a topping choice) be called?   
• Utterance   
• Slot   
**• Slot Value**   
   
How may you customize the pronunciation of specific acronyms in Amazon Polly on new text it hasn't seen before?   
• Using Speech Marks   
• Using SSML   
**• Using Lexicons**   
   
You wish to predict inventory demand over time using Amazon Forecast. Which model would you select for this application?   
• ARIMA (it might be automatically be selected)   
• DeepAR   
• **AutoML**   
   
Where does SageMaker's automatic scaling get the data it needs to determine how many endpoints you need?   
• **CloudWatch**   
• CloudTrail   
• EC2 The performance of the EC2 nodes that make up your endpoints are not read directly by SageMaker, but via CloudWatch.   
   
Your SageMaker inference is based on a Tensorflow or MXNet network. You want it to be fast, but don't want to pay for P2 or P3 inference nodes. What's a good solution?   
• Use inference pipelines   
**• Use Elastic Inference**   
• Use an M4 inference node   
   
You are deploying SageMaker inside a VPC, but the Internet access from SageMaker notebooks is considered a security hole. How might you address this?   
**• Disable internet access when creating the notebook, and set up a NAT Gateway to allow the outbound connections needed for training and hosting.**   
• Enable network isolation   
• Enable SSL / TLS in SageMaker   
   
You want to deploy your trained semantic segmentation model from SageMaker to an embedded ARM device in a car. Which services might you use to accomplish this?   
• Lambda and Lex   
**• SageMaker Neo and IoT GreenGrass**   
• AWS DeepRacer and DeepLens   
   
When constructing your own training container for SageMaker, where should the actual training script files be stored?   
**• /opt/ml/code/** This is where your code should go. The SAGEMAKER_PROGRAM environment variable will indicate the specific script that should be run.   
****• /opt/ml/model/   
• /opt/ml/train/   