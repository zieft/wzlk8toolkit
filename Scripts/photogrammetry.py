import os
import sys
from time import sleep

import core.ssh

sys.path.append(os.path.curdir)

Pkeypassword = sys.argv[0]
key1 = '4G8F4PBHBLNX7ZOW8N5P'
key2 = sys.argv[1]
bucket = 'ggr-bucket-cbf77f1e-eea2-4b4a-88b2-ae787daf3f42'

ssh, workdir = core.ssh.PKeyLogin_Trans('/home/yulin/Desktop/id_rsa', '{}'.format(Pkeypassword), '137.226.78.226', 22,
                                        'ggr_yz')  # TODO: hard coded

# create a pvc, a minio server, and expose the port.
core.ssh.kubectlApply(ssh,
                      '/home/ggr_yz/yaml_test/svcForDownLoad.yaml')  # TODO: hard coded, try to use yaml templates to generate new file to local path.
# wait till pods are running TODO: better idea?
sleep(20)
# get host machine IP:port of minio-service
svcIP = core.ssh.getSvcIp(ssh)
# create an alias in minio client on master node.
core.ssh.mcAliasSet(ssh, workdir, 'myminio', svcIP, key1, key2, 's3v4')
# download dataset to the persistent Volume
# core.ssh.mcDownload(ssh, key1, key2, bucket, 'mini3', 'mini3')
