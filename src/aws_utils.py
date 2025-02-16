import boto3
from dotenv import load_dotenv
import os

def connet_S3():
    load_dotenv()
    ACCESS_KEY = os.getenv('ACCESS_KEY')
    SECRET_KEY = os.getenv('SECRET_KEY')
    s3 = boto3.client(service_name='s3',
                  region_name='ap-northeast-2',
                  aws_access_key_id=ACCESS_KEY,
                  aws_secret_access_key=SECRET_KEY)
    
    return s3

def download_model(s3, bucket='optiquantbucket', prefix='main/'):
    obj_list = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    models_list = obj_list['Contents']
    key = models_list[1]['Key']
    file_name = key.split('/')[-1]
    PATH = f'/workspace/src/{file_name}'

    s3.download_file(bucket, key, PATH)
    model = file_name.split('-')[0]

    return model