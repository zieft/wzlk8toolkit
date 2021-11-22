import subprocess
import time
from ruamel import yaml
import sys
import os
import re
sys.path.append('..')
from wzlk8toolkit import PathInfo


class K8marsJobCreator:
    """
    Create yaml file.
    """
    def __init__(self, job_name):
        self.__unique_id = self.__add_unique_id()
        self.__name = job_name + '-' + self.unique_id
        self.__pvc_name = 'ggr-pvc-for-' + self.name
        self.__dataset_name = self.__get_dataset_name()
        self.__mg_file_name = self.__get_mg_file_name()
        self.__notation = self.__get_notation()
        self.__job_dir = self.__create_job_dir()
        self.__yaml_dir = self.__create_yaml_dir()
        self.__data_dir = self.__create_data_dir()
        self.__yaml_file_generated = self.__create_new_yaml_file()
        self.__yaml_pvc = self.__generate_yaml_pvc()
        self.__yaml_minio_client = YamlFileModelMinioClient()
        self.__yaml_wzlk8toolkit = self.__generate_yaml_wzlk8toolkit()
        self.__yaml_minio_server = YamlFileModelMinioService() # TODO: unique minio service port
        # self.__cmd_photogrammetry = CMDPhotogrammetry(self.dataset_name, self.mg_file_name).commands

    def __str__(self):
        return self.__name

    # def show_task_menu(self):
    #     print('----------------------------------------------------')
    #     print('Please select entry point:')
    #     print('1) Photogrammetry')
    #     print('2) Mesh Post processing')
    #     print('3) UV unwrap')
    #     print('4) Crack detection')
    #     print('5) Coordinate transformation to CAD')
    #
    # def select_task_menu(self):
    #     select = input('Please Select: ')
    #     if select == 1:
    #         photogrametry = CMDPhotogrammetry(self.dataset_name, self.mg_file_name)
    #         self.__yaml_wzlk8toolkit.set_commands(photogrametry.commands)
    #
    #     if select == 2:
    #         pass
    #     if select == 3:
    #         pass
    #     if select == 4:
    #         pass
    #     if select == 5:
    #         pass

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

    # @property
    # def cmd_photogrammetry(self):
    #     return self.__cmd_photogrammetry

    @property
    def pvc_name(self):
        return self.__pvc_name

    # @dataset_name.setter
    # def dataset_name(self, value):
    #     self.__dataset_name = value

    def __add_unique_id(self):
        time_stamp = time.time()
        time_local = time.localtime(time_stamp)
        time_id = time.strftime("%m%d%H%M%S",time_local)
        return time_id

    def __get_dataset_name(self):
        dataset_name = input('Please enter [dataset] name (the name of the dataset folder in S3): ')
        return dataset_name

    def __get_mg_file_name(self):
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
                yaml_pvc_string = yaml.dump(self.__yaml_pvc.text)
                yaml_job_string = yaml.dump(self.__yaml_wzlk8toolkit.text)
                f.writelines(yaml_pvc_string)
                f.write('---\n')
                f.writelines(yaml_job_string)

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
        kubectl_controller_obj = KubectlController(selected_job)
        kubectl_controller_obj.show_options_for_job()
        kubectl_controller_obj.select_option_for_job()


class ClusterPod:
    def __init__(self, name='', ):
        self.name = name

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
            pass # TODO
        if selected == '2':
            pass # TODO
        if selected == '3':
            pass # TODO
        if selected == '4':
            pass # TODO

    def __kubectl_run(self, cmd: str):
        """
        Run kubectl command through git bash using kubectl.exe in k8mars folder.
        :param cmd: str, kubectl command without kubectl
        :return:
        """
        cmd = './kubectl.exe ' + cmd
        current_dir = os.getcwd()
        work_dir = PathInfo.k8mars_dir
        if current_dir != work_dir:
            os.chdir(work_dir)
        splited_cmd = cmd.split()
        process = subprocess.Popen(splited_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        os.chdir(current_dir)

        return stdout


    def __get_pod_name(self):
        output = self.__kubectl_run('get pods -n ggr')
        fullName = re.findall(r'{}-.....'.format(self.job.name), output)

        return fullName[0]

    def __get_container_status(self, job_obj):
        # 获取容器状态，ready or not
        pod_name = self.__get_pod_name()
        description = str(self.__kubectl_run('-n ggr describe pod {}'.format(pod_name)))
        container_status_list = re.findall(r'ContainersReady(.*?)\\n', description)
        container_status = container_status_list[0].strip()
        print('ContainerReady:', container_status)

        return container_status


    def __get_svc_ip(self):
        output = self.__kubectl_run('describe svc -n ggr {}'.format(self.job.yaml_minio_server))
        svcIP = re.findall(r'10.*?9000', output)

        return svcIP

    def __run_photogrammetry(self):
        cmd = 'exec -n ggr {} -- ' \
              'meshroom_batch ' \
              '-i /tmp/{} ' \
              '--compute no ' \
              '--save /tmp/{}/generatedMgTemplate.mg -o /tmp/MeshroomCache'.format(fullPodName, folder, folder)

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
        super().__init__()
        with open(PathInfo.yaml_templates_dir / 'job_minioclient.yaml') as f:
            self.text = yaml.safe_load(f)

    def __set_pod_name(self, param):
        self.text['metadata']['name'] = param

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
