import boto3
import logging
from botocore.exceptions import ClientError
import os
from botocore.config import Config
from botocore.utils import fix_s3_host
from boto3.session import Session

def s3_upload(key: str, secret_key: str, endpoint_url: str, filePath: str, bucket: str):
    my_config = Config(
        region_name = '',

        s3 = {
            'addressing_style': 'path'
        },
        retries = {
            'max_attempts': 2,
            'mode': 'standard'
        }
    )

    session = Session(
        aws_access_key_id=key,
        aws_secret_access_key=secret_key
        )
    s3 = session.resource(
        service_name="s3",
        config=my_config,
        endpoint_url='https://s3.cluster.predictive-quality.io'
        )
    s3.meta.client.meta.events.unregister('before-sign.s3', fix_s3_host)

    bucket = s3.Bucket("ggr-bucket-cbf77f1e-eea2-4b4a-88b2-ae787daf3f42")

    client = s3.meta.client

    if os.path.isdir(filePath):
        for root,dirs,files in os.walk(filePath):
            for file in files:
                client.upload_file(os.path.join(root,file), bucket, file)

    elif os.path.isfile(filePath):
        client.upload_file(filePath, bucket, filePath.split('/')[-1]) # This is only for linux path format
                                                                      # TODO: add Windows adaptation
    else:
        print('Invalid Path!')

