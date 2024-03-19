"""In case huggingface is down, use this script.

It will perform the following steps:
- download the thenlper/gte-large zip file from AWS S3.
- unzip the directory
- load the model as sentence-transformers SentenceTransformer
"""
import subprocess
from sentence_transformers import SentenceTransformer

subprocess.run("aws s3 cp s3://anyscale-materials/rag-bootcamp-mar-2024/thenlper_gte-large.zip thenlper_gte-large-aws.zip", shell=True, check=True)
subprocess.run("unzip thenlper_gte-large-aws.zip", shell=True, check=True)
model = SentenceTransformer(model_name_or_path='./gte-large/thenlper_gte-large')