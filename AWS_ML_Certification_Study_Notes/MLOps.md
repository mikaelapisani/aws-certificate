# MLOps

**Build machine learning solutions for performance, availability, scalability, resiliency, and fault tolerance.**

## Amazon CloudWatch

monitors your AWS resources and the applications that you run on AWS in real time. Collect and track metrics, create customized dashboards, and set alarms that notify you or take actions when a specified metric reaches a threshold that you specify. 

enables you to **monitor**, store, and access your **log files** from EC2 instances, AWS CloudTrail, and other sources. Notify you when certain thresholds are met. 

CloudWatch events - delivers a near real-time stream of system events that describe changes in AWS resources. 

## AWS CloudTrail

**captures API** calls and related events made by or on behalf of your AWS account and delivers the log files to an Amazon S3 bucket that you specify.

- 90 days - This is the default trail. Information in this trail is kept for the last 90 days in a
rolling fashion
- Amazon SNS: You can be notified when CloudTrail publishes new log files to your Amazon S3 bucket.

## AWS GuardDuty

 threat detection service that continuously monitors your AWS accounts and workloads for attacks.

## Trusted Advisor

 ****make recommendations to save money

## VPC flow logs

if enabled capture information about IP traffic  

### AWS Management

console for audit 

- Multiple regions, Multiple AZs
- Auto Scaling groups

## Containers

- Dockerfile
- Docker Image
- Registry

### Amazon Elastic Container Registry (Amazon ECR)

Registry for docker images in AWS 

### Amazon Elastic Container Service (Amazon ECS)

### Amazon Elastic Kubernetes Service (Amazon EKS)

## AWS Batch

AWS Batch plans, schedules, and executes your batch computing workloads across the full range of AWS compute services and features, such as Amazon EC2 and Spot Instances

## AWS Lambda

is a compute service that lets you run code without provisioning or managing servers.

## AWS Fargate

AWS Fargate is a serverless, pay-as-you-go compute engine that lets you focus on building applications without managing servers. AWS Fargate is compatible with both Amazon Elastic Container Service (ECS) and Amazon Elastic Kubernetes Service (EKS)

## **Amazon EC2**

 Amazon Elastic Compute Cloud (Amazon EC2) is a web service that provides secure, resizable compute capacity in the cloud. 

## **AWS Deep Learning AMIs (DLAMI)**

 s your one-stop shop for deep learning in the cloud 

## Golden image

  ****A golden AMI is an AMI that you standardize through configuration, consistent security patching, and hardening

## Amazon EMR

**Amazon EMR** is a managed cluster platform that simplifies running big data frameworks, such as Apache Hadoop and Apache Spark , on AWS to process and analyze vast amounts of data.

Managed **Hadoop** service:  MapReduce → Yarn → HDFS

Map - transform data , reduce - aggregate transformed data  

Spark, Flink , Hive, EMR Notebook

process data in parallel across the cluster 

EC2 instances → node

master node → in charge of distribute data across nodes 

core nodes → store data in HDFs and run tasks  

tasks nodes → process 

transient vs long run - transient automatically terminate when running process finishes  (step mode vs cluster)

EBS for HDFS 

**EMRFS** - access s3 file as it were in HDFS 

**Security** - IAM policies and roles, Kebreros, SSH 

## Spark

SparkContext → Cluster manager → Executors → Tasks  

Spark SQL 

Spark Streaming → real time streaming - can connect to Kinesis 

MMLib → distributed and scalable → logistic regression, decision trees, clustering, kmeans , topic modeling LDA , pipelines , recommendation engine 

Graphix 

Zeppelin notebooks 

## Apply basic AWS security practices to machine learning solutions

- IAM: centralized mechanism for creating a managing user permissions. Can create groups of users with the same permissions.
- multifactor authentication MFA
- SSL/TLS to connect to servers
- AWS Config - keep track of resources
- use CloudTrail API and user activity - auditing
- CloudWatch - monitoring
- use encryption
- zone, geographic location
- Layers: perimeter, environment, data
- Type of credentials: user/pass, multi-factor, user access key, EC2 key pair
- Secret manager: centrally manages secrets
- AWS single sing on (SSO): sing in to the user portal
- AWS Security token service (STS): temporal limited privilegies for IAM credentials
- S3 bucket policies
- SecurityHub: to centralize all security issues

**Regions**

IAM is global resource

ask for choose a region

- S3
- SageMaker

VPC - isolation 

Security Group - acts as a firewall for the instance 

Network ACLs - optional layer for VPC

Subnet routing - group instances 

### Data Protection

- Encryption:
    - client-side (before)
    - server-side (after)
- Protecting in transit
    - HTTPS
    - TLS encrytption web based workflows
    - Glacier : secure and durable service for low-cost data archiving and long-term backup. Use Vault Lock for enforcing policies.
- certificate manager - ACM managing SSL/TLS
- Macicie- is a data security service that uses machine learning (ML) and pattern matching to discover and help protect your sensitive data.
- KMS - Key management and cryptography in your app
- Incident Response
    - APIs for automate runtime tasks
    - EBS snapshots
    - CloudFormation  provisioning and configuring  resources for you
- inter-node and inter-container encryption

## Choosing instance types

- **deep learning training**
    - M5 for CPU
    - Inference (less demanding) - C4 and C5 or check [Inference section at Amazon SageMaker](SageMaker.md)
    - GPU instances - P2 and P3 for training
    - **Spot instances** for training save up to 90% - .Using spot instances to train deep learning models : uses spare EC2 capacity that is available for less than the On-Demand price. 
    Spot can be interrupted at any time - make sure to use checkpoints and can increase waiting time

## Deploy and operationalize machine learning solutions

- Exposing endpoints and interacting with them
- ML model versioning
- A/B testing
- Retrain pipelines
- ML debugging/troubleshooting
    - Detect and mitigate drop in performance
    - Monitor performance of the model