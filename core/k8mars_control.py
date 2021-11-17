import subprocess
import time
import yaml
import sys
import os
sys.path.append('..')
from wzlk8toolkit import PathInfo

class K8marsJobCreator:
    """
    Create yaml file.
    """
    def __init__(self, job_name):
        self.__unique_id = self.__add_uniqe_id()
        self.__name = job_name + '-' + self.unique_id
        self.__dataset_name = self.__get_dataset_name()
        self.__mg_file_name = self.__get_mg_file_name()
        self.__job_dir = self.__create_job_dir()
        self.__yaml_dir = self.__create_yaml_dir()
        self.__data_dir = self.__create_data_dir()
        self.__yaml_file_generated = self.__create_new_yaml_file()
        self.__yaml_pvc = YamlFileModelPVC()
        self.__yaml_minio_client = YamlFileModelMinioClient()
        self.__yaml_wzlk8toolkit = YamlFileModelWzlk8toolkit()
        self.__yaml_minio_server = YamlFileModelMinioService()

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

    # @dataset_name.setter
    # def dataset_name(self, value):
    #     self.__dataset_name = value

    def __add_uniqe_id(self):
        time_stamp = time.time()
        time_local = time.localtime(time_stamp)
        time_id = time.strftime("%m%d%H%M%S",time_local)
        return time_id

    def __get_dataset_name(self):
        dataset_name = input('Please enter dataset name (the name of the dataset folder in S3): ')
        return dataset_name

    def __get_mg_file_name(self):
        mg_file_name = input('Please enter .mg file name (without .mg): ')
        return mg_file_name

    def __create_job_dir(self):
        job_dir = PathInfo.cache_folder / 'job_{}/'.format(self.name)
        try:
            os.mkdir(job_dir)
            print('Job_directory {} created!'.format(job_dir))
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
        self.__yaml_pvc.__set_pvc_name(self.name)

    def __show_task_menu(self):
        print('----------------------------------------------------')
        print('Please select entry point:')
        print('1) Photogrammetry')
        print('2) Mesh Post processing')
        print('3) UV unwrap')
        print('4) Crack detection')
        print('5) Coordinate transformation to CAD')

    def __select_task_menu(self):
        select = input('Please Select: ')
        if select == 1:
            photogrametry = NodePhotogrammetry(self.dataset_name, self.mg_file_name)
        if select == 2:
            pass
        if select == 3:
            pass
        if select == 4:
            pass
        if select == 5:
            pass

    def write_yaml_file(self, what_to_write='all'):
        if what_to_write == 'all':
            with open(self.__yaml_file_generated, 'w') as f:
                yaml_pvc_string = yaml.safe_dump(self.__yaml_pvc.text)
                yaml_job_string = yaml.safe_load(self.__yaml_wzlk8toolkit.text)
                f.writelines(yaml_pvc_string)
                f.write('---')
                f.writelines(yaml_job_string)

class NodePhotogrammetry:
    """
    Only called by K8marsJonCreate instance.
    """
    def __init__(self, dataset_name, mg_file_name):
        self.__dataset_name = dataset_name # without .mg
        self.__mg_file_name = mg_file_name
        self.commands = self.__generate_commands()

    @property
    def dataset_name(self):
        return self.__dataset_name

    @property
    def mg_file_name(self):
        return self.__mg_file_name

    def __generate_commands(self):
        cmds = ['sh',
                    '-c',
                    '--',
                    ]

    def __cmd_create_mg_template(self):
        cmd = "meshroom_batch -i " \
              "/storage/{} " \
              "--compute no " \
              "--save /storage/{}/generatedMgTemplate.mg " \
              "-o /storage/MeshroomCache".format(self.dataset_name, self.dataset_name)

    def __cmd_run_mgFileEditor(self):
        cmd = "python /opt/scripts/wzlk8toolkit/Scripts/mgFileEditor.py /storage/{}/{}".format(self.dataset_name, self.mg_file_name)

class K8marsJobManager:
    """
    Communicate with the cluster.
    """
    def __init__(self):
        self.__job_list = []

    @property
    def job_list(self):
        return self.__job_list

    def __add_new_job(self, K8marsJobCreator_obj):
        self.__job_list.append(K8marsJobCreator_obj)

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
    def __init__(self):
        """
        Read yaml file as a dict.
        """
        with open(PathInfo.yaml_templates_dir / 'job_wzlk8toolkit.yaml') as f:
            self.text = yaml.safe_load(f)

    def __set_pod_name(self, param):
        self.text['metadata']['name'] = param

    def __set_pvc_name(self, param):
        self.text['spec']['template']['spec']['volumes'][0]['persistentVolumeClaim']['claimName'] = param

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

class YamlFileModelMinioService:
    busy_ports = []
    busy_nodeports = []
    def __init__(self):
        """
        Read yaml file as a dict.
        """
        with open(PathInfo.yaml_templates_dir / 'svc_minio.yaml') as f:
            self.text = yaml.safe_load(f)

    def __set_minio_service_name(self, param):
        self.text['metadata']['name'] = param

    def __set_nodeport(self, param):
        self.text['spec']['ports'][0]['nodeport'] = param

    def __set_port(self, param):
        self.text['spec']['ports'][0]['port'] = param

    def __set_targetport(self, param):
        self.text['spec']['ports'][0]['targetPort'] = param

class YamlFileController:
    pass

class DockerFileSession:
    pass

class DockerFileController:
    pass
