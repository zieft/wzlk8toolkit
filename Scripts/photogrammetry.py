import os
import sys
sys.path.append(os.path.curdir)
from time import sleep
import core.ssh


Pkeypassword = sys.argv[1]
key1 = sys.argv[2]
key2 = sys.argv[3]
bucket = sys.argv[4]
folder = sys.argv[5]

print('SSH connection established!')
ssh, workdir = core.ssh.PKeyLogin_Trans('/home/yulin/Desktop/id_rsa', '{}'.format(Pkeypassword), '137.226.78.226', 22, 'ggr_yz')
# TODO: hard coded

# create a pvc and bound a pod for meshroom.
print('Creating Persistent Volume and meshroom container...')
core.ssh.kubectlApply(ssh, '/home/ggr_yz/yaml_test/pvcJobMeshroom.yaml')
# TODO: hard coded, try to use yaml templates to generate new file to local path.

# wait till pods are running TODO: better idea?
print('Waiting for container to run...')
sleep(30)

# download dataset to master node (k8-3)
print('Downloading dataset to the master node...')
core.ssh.k8_3s3download(ssh, key1, key2, 'mini3') #TODO: 'mini3' hard coded
sleep(30)

# get full pod name
print('Getting pod name...')
fullPodName = core.ssh.kubectlGetFullPodName(ssh, 'yz-meshroom')

# copy file into persistent volume through pod-for-meshroom
print('Moving dataset into computation units...')
core.ssh.kubectlCp(ssh, '/home/ggr_yz/mini3', fullPodName, 'mini3')

# remove cache folders/scripts in k8-3
print('Clearing cache...')
ssh.exec_command('rm -rf dataCache/ mini3/ s3download.py')
core.ssh.getstatus(ssh)

# Change AliceVision_install to AliceVision_bundle in the container.
print('Initiating computing environment...')
_,stdout,_ = ssh.exec_command('kubectl exec -n ggr {} -- mv /opt/AliceVision_install /opt/AliceVision_bundle'.format(fullPodName))
print(stdout.read().decode())
core.ssh.getstatus(ssh)

# insert project.mg file
    # TODO: think about how

# run a standard pipeline
print('Computing...Please wait...')
_,stdout,_ = ssh.exec_command('kubectl exec -n ggr {} -- meshroom_batch -i /tmp/mini3 -o /tmp/MeshroomCache/mini3_out'.format(fullPodName))
print(stdout.read().decode())
core.ssh.getstatus(ssh)

# stop Job to free pvc for minio to bound.
print('Dismounting finished Job...Please wait...')
core.ssh.kubectlDelete(ssh, 'job', 'yz-meshroom')
sleep(40)

# create a minio server in the cluster, expose port 9000 as service
print('Creating minIO Service... Waiting...')
core.ssh.kubectlApply(ssh, '/home/ggr_yz/yaml_test/svcForUpload.yaml')
sleep(30)
# get host machine IP:port of minio-service
svcIP = core.ssh.getSvcIp(ssh)

# create an alias in minio client on master node.
print('Creating myminio in Minio Client...')
core.ssh.mcAliasSet(ssh, workdir, 'myminio', svcIP, key1, key2, 's3v4')

# upload output data to s3 storage
print('Uploading output data to S3 Storage...')
# core.ssh.mcUpload(ssh, key1, key2,  'MeshroomCache', bucket, folder+'_out')
_, stdout, _ =ssh.exec_command('./mc cp --attr Cache-Control=max-age=90000,min-fresh=9000\;key1={}\;key2={} --recursive myminio/MeshroomCache/ s3/ggr-bucket-cbf77f1e-eea2-4b4a-88b2-ae787daf3f42/mini3_out'.format(key1, key2))
print(stdout.read().decode())
sleep(60)

# delete job, svc and pvc
print('Job finished, clearing cache...')
core.ssh.kubectlDelete(ssh, 'job', 'minio-server')
core.ssh.kubectlDelete(ssh, 'service', 'minio-service')
core.ssh.kubectlDelete(ssh, 'persistentvolumeclaim', 'yz-pvc')

# close session
ssh.close()
print('Done!')

