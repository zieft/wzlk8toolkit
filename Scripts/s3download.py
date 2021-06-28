"""
This script is supposed to run in k8-3 under directory ~/wzlk8toolkitCache/
with python version >= 3.6
Package of boto3 is prerequesited
"""

import os
import sys

os.system('pip3 install boto3')

from boto3.session import Session
from botocore.config import Config
from botocore.utils import fix_s3_host

# download mc
if os.system('ls ../mc') != 0:
    os.system('wget https://dl.min.io/client/mc/release/linux-amd64/mc; chmod +x ../mc')



key1 = sys.argv[1]
key2 = sys.argv[2]
folder = sys.argv[3]

endpoint_url = 'https://s3.cluster.predictive-quality.io'
bucket_ggr = "ggr-bucket-cbf77f1e-eea2-4b4a-88b2-ae787daf3f42"
initdir = os.getcwd()

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
    endpoint_url=endpoint_url
)

s3.meta.client.meta.events.unregister('before-sign.s3', fix_s3_host)

bucket = s3.Bucket(bucket_ggr)

# download dataset
def s3tok8_3(folderToDownload, newFolderName):
    keys = []
    pairs = {}
    os.mkdir('./{}'.format(newFolderName))
    os.chdir('./{}'.format(newFolderName))

    for files in bucket.objects.filter(Prefix=folderToDownload):
        print(files.key)
        keys.append(str(files.key))
    keys.pop(0)
    tempName = 0

    for key in keys:
        pairs[str(tempName)] = key
        bucket.download_file(key, str(tempName))
        tempName += 1

    os.mkdir('../{}'.format(folderToDownload))
    tempName = 0

    for file in os.listdir(os.getcwd()):
        os.rename(file, '../' + pairs[str(tempName)])
        tempName += 1

    os.chdir(initdir)

s3tok8_3(folder, 'dataCache')
s3tok8_3('k8_3configuration', 'configCache')
s3tok8_3('mgFileTemplates', 'mgFilesCache')
# downloade configurations
