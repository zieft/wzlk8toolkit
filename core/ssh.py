import re
import paramiko
import time


def login(username: str, password: str, port: int, hostIP: str):
    ssh = paramiko.SSHClient()
    # Allow connection to the host which not in the allowed-list
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # set connection
    ssh.connect(hostIP, username=username, port=port, password=password)

    # execute command
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("ls -l")

    # ssh_stdout is the output from the executed command above
    print(ssh_stdout.read())

    # close session
    ssh.close()

def PKeyLogin(privateKeyPath: str, keyPassword: str, hostIP: str, port: int, username: str):
    # Using a local generated secret_key file
    # if the password was not set when creating key pears, then leave keyPassword = ''
    pkey = paramiko.RSAKey.from_private_key_file(privateKeyPath, password=keyPassword)

    ssh = paramiko.SSHClient()
    ssh.connect(hostname=hostIP,
                port=port,
                username=username,
                pkey=pkey)

    stdin, stdout, stderr = ssh.exec_command('ls -l')
    print(stdout.read())

    ssh.close()

def login_Trans(hostIP, port, username: str, password: str):
    trans = paramiko.Transport((hostIP, port))
    trans.connect(username=username, password=password)

    ssh = paramiko.SSHClient()
    ssh._transport = trans

    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("ls -l")
    print(ssh_stdout.read())

    trans.close()

def PKeyLogin_Trans(keyFilePath: str, keyPassword: str, hostIP: str, port: int, username: str):
    """

    :param keyFilePath:
    :param keyPassword:
    :param hostIP:
    :param port:
    :param username:
    :return:
    """
    pkey = paramiko.RSAKey.from_private_key_file(keyFilePath, password=keyPassword)

    trans = paramiko.Transport((hostIP, port))
    trans.connect(username=username, pkey=pkey)

    ssh = paramiko.SSHClient()
    ssh._transport = trans

    stdin, stdout, stderr = ssh.exec_command('pwd')
    workdir = stdout.read().decode().strip()

    # ssh.close()
    # to close session, use ssh.close()

    return ssh, workdir


def printTime():
    now = int(round(time.time()*1000))
    now2 = '[{}]'.format(time.strftime('%d-%m-%Y %H:%M:%S',time.localtime(now/1000)))

    return now2

def getstatus(session):
    stdin, stdout, stderr = session.exec_command('echo $?')
    if stdout.read().decode() == '0\n':
        print(printTime(), 'Success!')
    else:
        print(printTime(), 'failed!')

def pwd(session):
    stdin, stdout, stderr = session.exec_command('pwd')
    print(stdout.read().decode())
    getstatus(session)

def ps_ef(session):
    stdin, stdout, stderr = session.exec_command('ps -ef')
    print(stdout.read().decode())



def mcAliasSet(session, workdir, aliasName, url, key, secretKey, api):
    stdin, stdout, stderr = session.exec_command(
        workdir + '/mc alias set {} {} {} {} --api {}'.format(aliasName, url, key, secretKey, api))
    print(stdout.read().decode())

    return aliasName

def k8_3s3download(session, key1, key2, folder):
    _, stdout,_ = session.exec_command('mkdir ./wzlk8toolkitCache; cd wzlk8toolkitCache/; wget https://raw.githubusercontent.com/zieft/wzlk8toolkit/master/Scripts/s3download.py ; python3 s3download.py {} {} {}'.format(key1, key2, folder))
    # stdin, stdout, stderr = session.exec_command('wget https://raw.githubusercontent.com/zieft/wzlk8toolkit/master/Scripts/s3download.py')
    print(stdout.read().decode())
    # stdin, stdout, stderr = session.exec_command('python3 s3download.py {} {} {}'.format(key1, key2, folder))
    # print(stdout.read().decode())
    getstatus(session)

def mcUpload(session, key1, key2, pathJob, bucket, pathS3):
    # cmd = './mc cp --attr Cache-Control=max-age=90000,min-fresh=9000;key1={};key2={} --recursive myminio/{} s3/{}/fromCluster/{}'.format(key, secretKey, pathJob, bucket, pathS3)
    cmd = './mc cp --attr Cache-Control=max-age=90000,min-fresh=9000\;key1={}\;key2={} --recursive myminio/{} s3/{}/outputsFromCluster/{}_out'.format(key1, key2, pathJob, bucket, pathS3)
    # print(cmd)
    stdin, stdout, stderr = session.exec_command(cmd)
    print(stdout.read().decode())



def kubectlApply(session, yamlPath):
    stdin, stdout, stderr = session.exec_command('kubectl apply -f {}'.format(yamlPath))
    output = stdout.read().decode()
    print(output)
    getstatus(session)

def kubectlGetFullPodName(session, podName: str):
    stdin, stdout, stderr = session.exec_command('kubectl get pods -n ggr')
    output = stdout.read().decode()
    fullName = re.findall(r'{}-.....'.format(podName), output)

    return fullName[0]
    # Another method:
    # _, stdout, _ = session.exec_command("kubectl get pods -n ggr | grep {} | awk '{{print $1}}'".format(podName))
    # fullName = stdout.read().decode()
    #
    # return fullName.split()

def getSvcIp(session):
    stdin, stdout, stderr = session.exec_command('kubectl describe svc -n ggr minio-service')  # TODO: hard coded
    output = stdout.read().decode()
    svcIP = re.findall(r'10.*?9000', output)
    fullIP = 'http://' + svcIP[0]
    print('hostIP is: ', fullIP)

    return fullIP

def kubectlCp(session, absfromPath, fullPodName, relToPath):
    stdin, stdout, stderr = session.exec_command('kubectl cp {} ggr/{}:/tmp/{}'.format(absfromPath, fullPodName, relToPath))
    output = stdout.read().decode()
    getstatus(session)

def kubectlDelete(session, type, name):
    """

    :param session:
    :param type: job, service, persistentvolumeclaim
    :param name:
    :return:
    """
    stdin, stdout, stderr = session.exec_command('kubectl delete -n ggr {} {}'.format(type, name))
    getstatus(session)

