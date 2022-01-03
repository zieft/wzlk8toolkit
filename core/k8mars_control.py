import subprocess
import time
from ruamel import yaml
import sys
import os
import re
sys.path.append('..')
from wzlk8toolkit import PathInfo
from yaml_templates import yaml_templates
import S3config
import time
from minio import Minio

class K8marsJobCreator:
    """
    Create yaml file.
    """
    def __init__(self):
        self.__unique_id = self.__add_unique_id()
        self.__job_name = self.__get_job_name()
        self.__name = self.__job_name + '-' + self.unique_id
        self.__pvc_name = 'ggr-pvc-for-' + self.name
        self.__notation = self.__get_notation()
        self.__job_dir = self.__create_job_dir()
        self.__yaml_dir = self.__create_yaml_dir()
        self.__data_dir = self.__create_data_dir()
        self.__yaml_file_generated = self.__create_new_yaml_file()
        self.__yaml_pvc = self.__generate_yaml_pvc()
        self.__yaml_minio_client = self.__set_yaml_minio_client()
        self.__yaml_wzlk8toolkit = self.__set_yaml_wzlk8toolkit()
        self.__yaml_minio_server = "minio-service-" + self.name
        # self.__yaml_minio_server = YamlFileModelMinioService() # TODO: unique minio service port
        # self.__cmd_photogrammetry = CMDPhotogrammetry(self.datasett_name, self.mg_file_name).commands

    def __str__(self):
        return self.__name

    @property
    def notation(self):
        return self.__notation

    @notation.setter
    def notation(self, param):
        self.__notation = param

    @property
    def yaml_minio_server(self):
        return self.__yaml_minio_server

    @yaml_minio_server.setter
    def yaml_minio_server(self, value):
        self.__yaml_minio_server = value

    @property
    def yaml_wzlk8toolkit(self):
        return self.__yaml_wzlk8toolkit

    @yaml_wzlk8toolkit.setter
    def yaml_wzlk8toolkit(self, value):
        self.__yaml_wzlk8toolkit = value

    @property
    def yaml_minio_client(self):
        return self.__yaml_minio_client

    @yaml_minio_client.setter
    def yaml_minio_client(self, value):
        self.__yaml_minio_client = value

    @property
    def yaml_pvc(self):
        return self.__yaml_pvc

    @yaml_pvc.setter
    def yaml_pvc(self, value):
        self.__yaml_pvc = value

    @property
    def yaml_file_name(self):
        return self.__yaml_file_generated

    @property
    def mg_file_name(self):
        return self.__mg_file_name

    @property
    def yaml_dir(self):
        return self.__yaml_dir

    @property
    def data_dir(self):
        return self.__data_dir

    @property
    def name(self):
        return self.__name

    @property
    def job_dir(self):
        return self.__job_dir

    @property
    def unique_id(self):
        return self.__unique_id

    @property
    def dataset_name(self):
        return self.__dataset_name

    @property
    def pvc_name(self):
        return self.__pvc_name

    def __get_job_name(self):
        job_name = input('Please enter [job] name: ')
        return job_name

    def __add_unique_id(self):
        time_stamp = time.time()
        time_local = time.localtime(time_stamp)
        time_id = time.strftime("%m%d%H%M%S",time_local)
        return time_id

    def get_dataset_name(self):
        dataset_name = input('Please enter [dataset] name (the name of the dataset folder in S3): ')
        return dataset_name

    def get_mg_file_name(self):
        mg_file_name = input('Please enter [.mg file] name (without .mg): ')
        return mg_file_name

    def __get_notation(self):
        notation = input('Please enter notation (optional): ')
        return notation

    def __create_job_dir(self):
        job_dir = PathInfo.cache_folder / '{}/'.format(self.name)
        try:
            os.mkdir(job_dir)
            print('Job directory [{}] created!'.format(job_dir))
        except FileExistsError:
            print('Folder exists.')

        return job_dir

    def __create_yaml_dir(self):
        yaml_dir = self.job_dir / "yaml_files/"
        try:
            os.mkdir(yaml_dir)
        except FileExistsError:
            print('Folder exists.')

        return yaml_dir

    def __create_data_dir(self):
        data_dir = self.job_dir / "data_files/"
        try:
            os.mkdir(data_dir)
        except FileExistsError:
            print('Folder exist.')

        return data_dir

    def __create_new_yaml_file(self):
        yaml_file_name = self.__yaml_dir / (self.name + '.yaml')
        with open(yaml_file_name, mode='a') as f:
            pass

        return yaml_file_name

    def __set_pvc_name(self):
        self.__yaml_pvc.set_pvc_name(self.name)

    def __generate_yaml_pvc(self):
        yaml = YamlFileModelPVC()
        yaml.__set_pvc_name = self.pvc_name

        return yaml

    def __set_yaml_minio_client(self):
        raw_text = yaml_templates.yaml_minio_client

        return raw_text

    def __set_yaml_wzlk8toolkit(self):
        raw_text = yaml_templates.yaml_Jobwzlk8toolkit
        cmd = """['sh', '-c', 'echo The app is running! && MINIO_ROOT_USER=admin MINIO_ROOT_PASSWORD=password minio server /storage --console-address ":9001" && mkdir /storage/recieve && sleep 36000']"""
        pvc_name = self.pvc_name
        pod_name = self.name
        image_name = "zieft/wzlk8toolkit:v0.3"
        minio_SVC_name = self.name
        yaml = raw_text.format(pvc_name, pod_name, pvc_name, image_name, cmd, minio_SVC_name)

        return yaml

    def __generate_yaml_wzlk8toolkit(self):
        yaml = YamlFileModelWzlk8toolkit()
        cmd_photogrammetry = CMDPhotogrammetry(self.dataset_name, self.mg_file_name).commands
        yaml.set_pod_name(self.name)
        yaml.set_pvc_name(self.__pvc_name)
        yaml.set_commands(cmd_photogrammetry)

        return yaml

    def write_yaml_file(self, what_to_write='all'):
        if what_to_write == 'all':
            with open(self.__yaml_file_generated, 'w') as f:
                # yaml_pvc_string = yaml.dump(self.__yaml_pvc.text)
                # yaml_job_string = yaml.dump(self.__yaml_wzlk8toolkit.text)
                yaml_job_wzlk8toolkit = self.yaml_wzlk8toolkit
                # f.writelines(yaml_pvc_string)
                # f.write('---\n')
                # f.writelines(yaml_job_string)
                f.write(yaml_job_wzlk8toolkit)

class CMDPhotogrammetry:
    """
    Only called by YamlFileModelWzlk8toolkit class.
    """
    def __init__(self, dataset_name, mg_file_name):
        self.__dataset_name = dataset_name # without .mg
        self.__mg_file_name = mg_file_name
        self.__commands = self.__generate_commands()

    @property
    def commands(self):
        return self.__commands

    @property
    def dataset_name(self):
        return self.__dataset_name

    @property
    def mg_file_name(self):
        return self.__mg_file_name

    def __generate_commands(self):
        cmds = ['sh', '-c', '--', self.__cmd_create_mg_template(), self.__cmd_copy_mg_to_dataset(),
                self.__cmd_run_mgFileEditor(), self.__cmd_run_photogrammetry()]
        return cmds

    def __cmd_create_mg_template(self):
        # make sure there is no other .mg file exists in the dataset folder.
        cmd = "meshroom_batch -i " \
              "/storage/fromS3/{} " \
              "--compute no " \
              "--save /storage/fromS3/{}/generatedMgTemplate.mg " \
              "-o /storage/CacheFiles/MeshroomCache".format(self.dataset_name, self.dataset_name)

        return cmd

    def __cmd_copy_mg_to_dataset(self):
        cmd = "cp /storage/fromS3/mgTemplates/{}.mg /storage/fromS3/{}".format(self.mg_file_name, self.dataset_name)

        return cmd

    def __cmd_run_mgFileEditor(self):
        cmd = "python /opt/scripts/wzlk8toolkit/Scripts/mgFileEditor.py /storage/fromS3/{}/{}.mg".format(self.dataset_name, self.mg_file_name)

        return cmd

    def __cmd_run_photogrammetry(self):
        # meshroom_compute --cache CacheFolder .mgFile
        cmd = 'meshroom_compute --cache /storage/CacheFiles/MeshroomCache /storage/dataset/{}.mg'.format(self.mg_file_name)

        return cmd

class K8marsJobManager:
    """
    Communicate with the cluster.
    """
    def __init__(self):
        self.__job_list = []

    @property
    def job_list(self):
        return self.__job_list

    def add_new_job(self, K8marsJobCreator_obj):
        self.__job_list.append(K8marsJobCreator_obj)

    def show_jobs_by_index(self):
        a = 1
        print('-----------------start of jobs list----------------------')
        for job in self.__job_list:
            print('Job ' + str(a) + ') ' + job.__str__())
            a += 1
        print('------------------end of jobs list-----------------------')

    def select_job(self):
        job_index = int(input('please type job index to select a job: ')) - 1
        selected_job = self.__job_list[job_index]


        return selected_job

class ClusterPod:
    def __init__(self, name='', ):
        self.name = name

class MinIOClientLocal:
    @staticmethod
    def set_host(s3_host=S3config.s3_host, key1=S3config.key1, key2=S3config.key2):
        cur_dir = os.getcwd()
        os.chdir(PathInfo.k8mars_dir)

        cmd = 'mc config host add s3 {} {} {} --api s3v4'.format(s3_host, key1, key2)
        output = os.popen(cmd).readlines()

        os.chdir(cur_dir)
        return output[0].strip()

    @staticmethod
    def cp_from_s3_to_local(data_to_be_copied, location, bucket=S3config.bucket):
        cur_dir = os.getcwd()
        os.chdir(PathInfo.k8mars_dir)

        cmd = "mc cp --recursive s3/{}/{} {}".format(bucket, data_to_be_copied, location)
        output = os.popen(cmd).readlines()

        os.chdir(cur_dir)
        return output[0].strip()

    @staticmethod
    def cp_from_local_to_s3(data_to_be_copied, location_in_s3, bucket=S3config.bucket):
        cur_dir = os.getcwd()
        os.chdir(PathInfo.k8mars_dir)

        cmd = "mc cp --recursive {} s3/{}/{} ".format(data_to_be_copied, bucket, location_in_s3)
        output = os.popen(cmd).readlines()

        os.chdir(cur_dir)
        return output[0].strip()

class KubectlController:
    def __init__(self, k8marsjobcreator_obj):
        self.job = k8marsjobcreator_obj

    def show_options_for_job(self):
        print('----------------------------------------------------------')
        print('1) Run Photogrammetry')
        print('2) Run Mesh-postprocessing')
        print('3) Get status')
        print('4) Delete Job')

    def select_option_for_job(self):
        selected = input('Please select: ')
        if selected == '1':
            # Get names for dataset and provided .mg file
            self.job.__dataset_name = self.job.get_dataset_name()
            self.job.__mg_file_name = self.job.get_mg_file_name()
            # Copy provided .mg file from S3 -> local cache folder through Local Minio Client
            provided_mg_file_location_in_s3 = 'mgFileTemplates/' + self.job.__mg_file_name
            MinIOClientLocal.cp_from_s3_to_local(provided_mg_file_location_in_s3, self.job.__data_dir)
            # Create Pods for Photogrammetry
            cmd_apply_job = " -n {} apply " \
                            "-f {}".format(S3config.name_space, self.job.__yaml_file_generated)
            self.__kubectl_apply(cmd_apply_job)
            # Get full pod name
            fullPodName = self.__get_pod_name()
            # Wait for pod ready
            pod_status = self.__get_pod_status(fullPodName)
            while pod_status is not True:
                time.sleep(10)
                print("Pod is not ready, retry in 10 sec.")
                pod_status = self.__get_pod_status(fullPodName)
            # Generate a .mg file according to the dataset
            cmd_generate_mg = "exec -n ggr {} -- meshroom_batch " \
                               "-i /storage/fromS3/{} " \
                               "--compute no " \
                               "--save /storage/toS3/CacheBlender/{}.mg " \
                               "-o /storage/toS3/".format(fullPodName, self.job.__dataset_name, self.job.name)
            self.__kubectl_run(cmd_generate_mg)
            # Copy generated .mg file from Persistent Volume to S3 Storage through Minio Client Container
            ## Get MinioClient pod name
            MC_pod_name = self.__get_pod_name(type="yz-minio-client")
            ## Set S3 host
            cmd_set_S3_host = "exec -n ggr {} -- " \
                              "mc config host add s3 " \
                              "{} {} {}" \
                              "-api s3v4".format(MC_pod_name, S3config.s3_host, S3config.key1, S3config.key2)
            self.__kubectl_run(cmd_set_S3_host)
            ## get MinIO Service IP
            svc_IP = self.__get_svc_ip()
            ## Set alias
            cmd_set_alias = "exec -n ggr {} -- " \
                            "mc alias set " \
                            "{} {} {}".format(MC_pod_name, self.job.name, svc_IP, S3config.key1, S3config.key2)
            self.__kubectl_run(cmd_set_alias)
            ## PV -> S3
            cmd_pv_2_s3 = "exec -n ggr {} -- " \
                          "mc cp " \
                          "--attr Cache-Control=max-age=90000,min-fresh=9000\;" \
                          "key1={}\;" \
                          "key2={}" \
                          " --recursive " \
                          "{}/toS3/CacheBlender/{}.mg " \
                          "s3/{}/temp/cache/{}.mg".format(MC_pod_name, S3config.key1, S3config.key2,
                                                          self.job.name, self.job.name, S3config.bucket,
                                                          self.job.name)
            self.__kubectl_run(cmd_pv_2_s3)
            ## S3 -> Local
            generated_mg_file_location_in_s3 = "s3/{}/temp/cache/{}.mg".format(S3config.bucket, self.job.name)
            MinIOClientLocal.cp_from_s3_to_local(generated_mg_file_location_in_s3, self.job.__data_dir)
            # Merge
            provided_mg_file_location_in_Local = self.job.__data_dir / "{}.mg".format(self.job.__mg_file_name)
            generated_mg_file_location_in_Local = self.job.__data_dir / "{}.mg".format(self.job.name)
            self.__merge_mg_files(provided_mg_file_location_in_Local, generated_mg_file_location_in_Local)
            # Copy converted(provided) .mg file from Local -> S3
            destiny_location_in_s3_and_pv = "datasets/{}/{}.mg".format(self.job.__dataset_name,
                                                                       self.job.__mg_file_name)
            MinIOClientLocal.cp_from_local_to_s3(provided_mg_file_location_in_Local,
                                                 destiny_location_in_s3_and_pv)
            # Copy converted(provided) .mg file from S3 -> PV
            cmd_s3_2_pv = "exec -n ggr {} -- " \
                          "mc cp --recursive " \
                          "s3/{}/{}.mg " \
                          "{}/fromS3/{}.mg ".format(MC_pod_name, S3config.bucket,
                                                    destiny_location_in_s3_and_pv,
                                                    self.job.name, destiny_location_in_s3_and_pv)
            self.__kubectl_run(cmd_s3_2_pv)
            # Run Photogrammetry
            cmd_photogrammetry = "exec -n ggr {} -- " \
                                 "meshroom_compute " \
                                 "--cache /storage/CacheFiles/CacheMeshroom " \
                                 "{}/fromS3/{}.mg".format(fullPodName, self.job.name,
                                                          destiny_location_in_s3_and_pv)
            self.__kubectl_run(cmd_photogrammetry)

        if selected == '2':
            pass # TODO
        if selected == '3':
            pass # TODO
        if selected == '4':
            pass # TODO

    def __kubectl_apply(self, cmd: str):
        cur_dir = os.getcwd()
        os.chdir(PathInfo.k8mars_dir)
        cmd = "kubectl " + cmd
        output = os.popen(cmd).readlines()

        os.chdir(cur_dir)
        return output[0].strip()

    def __kubectl_run(self, cmd: str):
        """
        Run kubectl command through git bash using kubectl.exe in k8mars folder.
        :param cmd: str, kubectl command without kubectl
        :return:
        """
        cmd = 'kubectl ' + cmd
        current_dir = os.getcwd()
        work_dir = PathInfo.k8mars_dir
        if current_dir != work_dir:
            os.chdir(work_dir)
        # splited_cmd = cmd.split()
        # process = subprocess.Popen(splited_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # stdout, stderr = process.communicate()
        stdout = os.popen(cmd).readlines()
        os.chdir(current_dir)

        return stdout[0].strip()

    def __get_pod_name(self, type="job"):
        output = self.__kubectl_run('get pods -n {}'.format(S3config.name_space))
        if type== "job":
            fullName = re.findall(r'{}-.....'.format(self.job.name), output)
        else:
            fullName = re.findall(r'{}-.....'.format(type), output)

        return fullName[0]

    def __get_pod_status(self, full_pod_name):
        # 获取容器状态，ready or not
        description = str(self.__kubectl_run('-n ggr describe pod {}'.format(full_pod_name)))
        container_status_list = re.findall(r'ContainersReady(.*?)\\n', description)
        container_status = container_status_list[0].strip()
        print('ContainerReady:', container_status)
        if container_status == "Ready":
            return True
        else:
            return False

    def __get_svc_ip(self):
        output = self.__kubectl_run('describe svc -n ggr {}'.format(self.job.yaml_minio_server))
        svcIP = re.findall(r'10.*?9000', output)

        return svcIP

    def __merge_mg_files(self, provided_file, generated_file):
        """
        The provided .mg file will be rewrite.
        :param provided_file: str., abs. path to the provided .mg file
        :param generated_file: str., abs. path to the generated .mg file
        :return: str., stdout
        """
        script_path = PathInfo.k8mars_dir / "wzlk8toolkit" / "Scripts" / "mgFileEditor.py"
        output = os.popen("python {} {} {}".format(script_path, provided_file, generated_file)).readlines()

        return output[0].strip()


    # def __run_photogrammetry(self):
    #     cmd = 'exec -n ggr {} -- ' \
    #           'meshroom_batch ' \
    #           '-i /tmp/{} ' \
    #           '--compute no ' \
    #           '--save /tmp/{}/generatedMgTemplate.mg -o /tmp/MeshroomCache'.format(fullPodName, folder, folder)

class YamlFileModelPVC:
    def __init__(self):
        """
        Read yaml file as a dict.
        """
        with open(PathInfo.yaml_templates_dir / 'pvc.yaml') as f:
            self.text = yaml.safe_load(f)

    def __set_pvc_name(self, param):
        self.text['metadata']['name'] = param

class YamlFileModelMinioClient:
    def __init__(self):
        """
        Read yaml file as a dict.
        """
        # super().__init__()
        # with open(PathInfo.yaml_templates_dir / 'job_minioclient.yaml') as f:
        #     self.text = yaml.safe_load(f)
        self.text = yaml_templates.yaml_minio_client
        self.pod_name = self.__set_pod_name()

    def __set_pod_name(self, param):
        # self.text['metadata']['name'] = param
        return param

    def __set_container_name(self, param):
        self.text['spec']['template']['spec']['containers'][0]['name'] = param

    def __set_image(self, param):
        self.text['spec']['template']['spec']['containers'][0]['image'] = param

    def __set_commands(self, param):
        """
        Set the bash commands which will be executed in the container when creating.
        :param param: list, a list of commands, each command is a str element in the list.
        :return:
        """
        self.text['spec']['template']['spec']['containers'][0]['command'] = param

class YamlFileModelWzlk8toolkit:
    """
    Only called by K8marsJobCreator class.
    """
    def __init__(self):
        """
        Read yaml file as a dict.
        """
        with open(PathInfo.yaml_templates_dir / 'job_wzlk8toolkit.yaml') as f:
            self.text = yaml.safe_load(f)

    def set_pod_name(self, param):
        self.text['metadata']['name'] = param

    def set_pvc_name(self, param):
        self.text['spec']['template']['spec']['volumes'][0]['persistentVolumeClaim']['claimName'] = param

    def __set_container_name(self, param):
        self.text['spec']['template']['spec']['containers'][0]['name'] = param

    def __set_image(self, param):
        self.text['spec']['template']['spec']['containers'][0]['image'] = param

    def set_commands(self, param):
        """
        Set the bash commands which will be executed in the container when creating.
        :param param: list, a list of commands, each command is a str element in the list.
        :return:
        """
        self.text['spec']['template']['spec']['containers'][0]['command'] = list(param)

class YamlFileModelMinioService:
    busy_ports = []
    busy_nodeports = []
    def __init__(self):
        """
        Read yaml file as a dict.
        """
        with open(PathInfo.yaml_templates_dir / 'svc_minio.yaml') as f:
            self.text = yaml.safe_load(f)

    def set_minio_service_name(self, param):
        self.text['metadata']['name'] = param

    def set_nodeport(self, param):
        self.text['spec']['ports'][0]['nodeport'] = param

    def set_port(self, param):
        self.text['spec']['ports'][0]['port'] = param

    def set_targetport(self, param):
        self.text['spec']['ports'][0]['targetPort'] = param

class YamlFileController:
    pass

class DockerFileSession:
    pass

class DockerFileController:
    pass
