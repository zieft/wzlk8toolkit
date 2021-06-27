import os
import sys
sys.path.append(os.path.curdir)
from time import sleep
import core.ssh


Pkeypassword = sys.argv[1]
userName = sys.argv[2]
key1 = sys.argv[3]
key2 = sys.argv[4]
bucket = sys.argv[5]
rsa_id_path = sys.argv[6]
folder = sys.argv[7]
mgFileName = sys.argv[8]

print(core.ssh.printTime(), 'SSH connection established!')
ssh, workdir = core.ssh.PKeyLogin_Trans(rsa_id_path, '{}'.format(Pkeypassword), '137.226.78.226', 22, userName)

# download dataset to master node (k8-3)
print(core.ssh.printTime(), 'Downloading dataset to the master node...')
core.ssh.k8_3s3download(ssh, key1, key2, folder) #TODO: test change
sleep(5)

# create a pvc and bound a pod for meshroom.
print(core.ssh.printTime(), 'Creating Persistent Volume and meshroom container...')
core.ssh.kubectlApply(ssh, '{}/wzlk8toolkitCache/k8_3configuration/pvcJobMeshroom.yaml'.format(workdir))
# TODO: hard coded, try to use yaml templates to generate new file to local path.

# wait till pods are running TODO: better idea?
print(core.ssh.printTime(), 'Waiting for container ready to run...')
sleep(30)

# get full pod name
print(core.ssh.printTime(), 'Getting pod name...')
fullPodName = core.ssh.kubectlGetFullPodName(ssh, 'yz-meshroom')
print(core.ssh.printTime(), 'Pod name is: ', fullPodName)

# copy file into persistent volume through pod-for-meshroom
print(core.ssh.printTime(), 'Moving dataset into computation units...')
core.ssh.kubectlCp(ssh, '{}/wzlk8toolkitCache/mini3'.format(workdir), fullPodName, 'mini3')

# Change AliceVision_install to AliceVision_bundle in the container.
print(core.ssh.printTime(), 'Initiating computing environment...')
_,stdout,_ = ssh.exec_command('kubectl exec -n ggr {} -- mv /opt/AliceVision_install /opt/AliceVision_bundle'.format(fullPodName))
print(stdout.read().decode())
core.ssh.getstatus(ssh)

# insert project.mg file
    # TODO: think about how

if mgFileName != '':
    """
    given .mg file should be put in the raw dataset folder and must be generated or
    converted from Meshroom version 2021.1!
    """
    # generate a project .mg file from dataset
    print(core.ssh.printTime(), 'Generating a standard .mg file.')
    _, stdout, _ = ssh.exec_command('kubectl exec -n ggr {} -- meshroom_batch -i /tmp/{} --compute no --save /tmp/{}/generatedMgTemplate.mg'.format(fullPodName, folder, folder))
    print(core.ssh.printTime(), stdout.read().decode())
    _, stdout, _ = ssh.exec_command('kubectl exec -n ggr {} -- rm -rf /tmp/{}/MeshroomCache'.format(fullPodName, folder))
    # download mgFileEditer.py from github.com/zieft/wzlk8toolkit and run with python2.7
    print(core.ssh.printTime(), 'Inserting customized computing graph')
    urlToMgFileEditer = 'https://raw.githubusercontent.com/zieft/wzlk8toolkit/master/Scripts/mgFileEditor.py'
    ssh.exec_command('kubectl exec -n ggr {} -- cd /tmp/{} && wget {} && python mgFileEditor.py {}'.format(fullPodName, folder, urlToMgFileEditer, mgFileName))
    core.ssh.getstatus(ssh)
    _, stdout, _ = ssh.exec_command('kubectl exec -n ggr {} -- meshroom_compute --cache /tmp/MeshroomCache {}'.format(fullPodName, mgFileName))
    print(core.ssh.printTime(), stdout.read().decode())


else:
    # run a standard pipeline
    print(core.ssh.printTime(), 'Computing...Please wait...')
    _,stdout,_ = ssh.exec_command('kubectl exec -n ggr {} -- meshroom_batch -i /tmp/mini3 -o /tmp/MeshroomCache/mini3_out'.format(fullPodName))
    print(stdout.read().decode())
    core.ssh.getstatus(ssh)

# stop Job to free pvc for minio to bound.
print(core.ssh.printTime(), 'Dismounting finished Job...Please wait...')
core.ssh.kubectlDelete(ssh, 'job', 'yz-meshroom')
sleep(40)

# create a minio server in the cluster, expose port 9000 as service
print(core.ssh.printTime(), 'Creating minIO Service... Waiting...')
core.ssh.kubectlApply(ssh, '{}/wzlk8toolkitCache/k8_3configuration/svcForUpload.yaml'.format(workdir))
sleep(30)
# get host machine IP:port of minio-service
svcIP = core.ssh.getSvcIp(ssh)

# create an alias in minio client on master node.
print(core.ssh.printTime(), 'Creating myminio in Minio Client...')
core.ssh.mcAliasSet(ssh, workdir, 'myminio', svcIP, key1, key2, 's3v4')

# upload output data to s3 storage
print(core.ssh.printTime(), 'Uploading output data to S3 Storage...')
# core.ssh.mcUpload(ssh, key1, key2,  'MeshroomCache', bucket, folder+'_out')
# _, stdout, _ = ssh.exec_command('./mc cp --attr Cache-Control=max-age=90000,min-fresh=9000\;key1={}\;key2={} --recursive myminio/MeshroomCache/ s3/ggr-bucket-cbf77f1e-eea2-4b4a-88b2-ae787daf3f42/mini3_out'.format(key1, key2))
pathJob = 'MeshroomCache'
core.ssh.mcUpload(ssh, key1, key2, pathJob, bucket, folder)
# print(stdout.read().decode())
# sleep(60)

# delete job, svc and pvc
print(core.ssh.printTime(), 'Job finished, clearing cache...')
core.ssh.kubectlDelete(ssh, 'job', 'minio-server')
core.ssh.kubectlDelete(ssh, 'service', 'minio-service')
core.ssh.kubectlDelete(ssh, 'persistentvolumeclaim', 'yz-pvc')
# remove cache folders/scripts in k8-3
ssh.exec_command('rm -rf wzlk8toolkitCache')
core.ssh.getstatus(ssh)

# close session
ssh.close()
print(core.ssh.printTime(), 'All done!')

