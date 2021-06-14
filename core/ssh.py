import paramiko


def login(username: str, password: str, port: int, hostIP: str):
    ssh = paramiko.SSHClient()
    # Allow connection to the host which not in the allowed-list
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # set connection
    ssh.connect(hostIP, username=username, port=port, password=password)

    # execute command
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("ls -l")

    # ssh_stdout is the output from the excuted command above
    print(ssh_stdout.read())

    # close session
    ssh.close()


def PKeyLogin(privateKeyPath: str, keyPassword: str, hostIP: str, port: int, username: str):
    # 指定本地的RSA私钥文件
    # 如果建立密钥对时设置的有密码，password为设定的密码，如无不用指定password参数
    pkey = paramiko.RSAKey.from_private_key_file(privateKeyPath, password=keyPassword)

    # 建立连接
    ssh = paramiko.SSHClient()
    ssh.connect(hostname=hostIP,
                port=port,
                username=username,
                pkey=pkey)

    # 执行命令
    stdin, stdout, stderr = ssh.exec_command('ls -l')

    # 结果放到stdout中，如果有错误将放到stderr中
    print(stdout.read())

    # 关闭连接
    ssh.close()


# def Trans(hostIP: str, port: int):
#     trans = paramiko.Transport((hostIP, port))
#     return transObj


def login_Trans(hostIP, port, username: str, password: str):
    # 建立连接
    trans = paramiko.Transport((hostIP, port))
    trans.connect(username=username, password=password)

    # 将sshclient的对象的transport指定为以上的trans
    ssh = paramiko.SSHClient()
    ssh._transport = trans

    # 剩下的就和上面一样了
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("ls -l")
    print(ssh_stdout.read())

    # 关闭连接
    trans.close()


def PKeyLogin_Trans(keyFilePath: str, keyPassword: str, hostIP: str, port: int, username):
    pkey = paramiko.RSAKey.from_private_key_file(keyFilePath, password=keyPassword)

    # 建立连接
    trans = paramiko.Transport((hostIP, port))
    trans.connect(username=username, pkey=pkey)

    # 将sshclient的对象的transport指定为以上的trans
    ssh = paramiko.SSHClient()
    ssh._transport = trans

    # 执行命令，和传统方法一样
    stdin, stdout, stderr = ssh.exec_command('df -hl')
    print(stdout.read().decode())

    # 关闭连接
    trans.close()

# def sftp_send(transObj, ):
#
# def sftp_download(transObj):
