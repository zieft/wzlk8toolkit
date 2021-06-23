import os
import sys
sys.path.append(os.path.curdir)
from time import sleep
import core.ssh


Pkeypassword = sys.argv[1]
print(Pkeypassword)
key1 = '4G8F4PBHBLNX7ZOW8N5P'
key2 = sys.argv[2]
print(key2)
bucket = 'ggr-bucket-cbf77f1e-eea2-4b4a-88b2-ae787daf3f42'

ssh, workdir = core.ssh.PKeyLogin_Trans('/home/yulin/Desktop/id_rsa', '{}'.format(Pkeypassword), '137.226.78.226', 22, 'ggr_yz')
# TODO: hard coded

# create a pvc and bound a pod for meshroom.
core.ssh.kubectlApply(ssh, '/home/ggr_yz/yaml_test/pvcJobMeshroom.yaml')
# TODO: hard coded, try to use yaml templates to generate new file to local path.

# wait till pods are running TODO: better idea?
sleep(30)
# download dataset to master node (k8-3)
core.ssh.k8_3s3download(ssh, key1, key2, 'mini3')
# copy file into persistent volume through pod-for-meshroom





# get host machine IP:port of minio-service
# svcIP = core.ssh.getSvcIp(ssh)
# create an alias in minio client on master node.
# core.ssh.mcAliasSet(ssh, workdir, 'myminio', svcIP, key1, key2, 's3v4')
