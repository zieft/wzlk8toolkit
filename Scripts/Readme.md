# photogrammetry.py

This script provide a way to automatic compute a meshroom pipeline.

Input: raw images, [meshroom project file]
Output: 

#### system requirement for Windows 10:

python ver. 3.6 or higher

inside RWTH Network or through VPN

installed python packages: paramiko, boto3

#### run the script by using command below:

python3 path/to/wzlk8toolkit/Scripts/photogrammetry.py [password to SSH private Key] [s3 key] [s3 secret key] [s3 bucket] [path to S3 secret key ] [folder name of the dataset] [name of the project file which contain computation pipeline]

#### Note: all [Options] must be provided
#### if you want to run a default computation pipeline, set [name of the project file which contain computation pipeline] = ''.
#### please note that large dataset (number of images > 25) may cause memory exhaust when a project file is not provided.
