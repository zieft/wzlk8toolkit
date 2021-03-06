import os

from boto3.session import Session
from botocore.config import Config
from botocore.utils import fix_s3_host

# key1 = sys.argv[0]
# key2 = sys.argv[1]
key1 = '4G8F4PBHBLNX7ZOW8N5P'


def s3Download(key1: str, key2: str, folder):
    my_config = Config(
        region_name='',

        s3={
            'addressing_style': 'path'
        },
        retries={
            'max_attempts': 2,
            'mode': 'standard'
        }
    )

    session = Session(
        aws_access_key_id=key1,
        aws_secret_access_key=key2
    )
    s3 = session.resource(
        service_name="s3",
        config=my_config,
        endpoint_url='https://s3.cluster.predictive-quality.io'
    )

    s3.meta.client.meta.events.unregister('before-sign.s3', fix_s3_host)

    bucket = s3.Bucket("ggr-bucket-cbf77f1e-eea2-4b4a-88b2-ae787daf3f42")

    keys = []
    pairs = {}
    os.mkdir('./dataCache')
    os.chdir('./dataCache')

    for files in bucket.objects.filter(Prefix='mini3'):
        print(files.key)
        keys.append(str(files.key))
    keys.pop(0)
    tempName = 0

    for key in keys:
        pairs[str(tempName)] = key
        bucket.download_file(key, str(tempName))
        tempName += 1

    os.mkdir('../mini3')
    tempName = 0

    for file in os.listdir(os.getcwd()):
        os.rename(file, '../' + pairs[str(tempName)])
        tempName += 1

        # try:
        #     print('Copying ', files.key)
        #     bucket.download_file(files.key, files.key)
        # except botocore.exceptions.ClientError as e:
        #     if e.response['Error'['Code']] == '404':
        #         print('The object "', files.key, '" is a directory, or does not exist.')
        #     else:
        #         raise
