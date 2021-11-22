from core.k8mars_control import *
import os
from pathlib import Path

class PathInfo:
    """
    Path of directory always ends with '/'
    """
    work_dir = Path(os.getcwd())
    k8mars_dir = Path(os.path.abspath(os.path.join(os.getcwd(), '..')))
    yaml_templates_dir = work_dir / 'yaml_templates/'
    cache_folder = work_dir / 'k8marsCache/'

class Wzlk8toolkitMain:
    def __init__(self):
        self.__manager = K8marsJobManager()

    def __display_main_menu(self):
        print('1) Create a new job.')
        print('2) Existing job.')
        print('3) Exit.')

    def __select_main_menu(self):
        item = input('please select: ')
        if item == '1':
            self.__create_job()
        if item == '2':
            self.__manager.show_jobs_by_index()
            self.__manager.select_job()
            # show list of jobs with index
            # menu for entry points,
            # edit detail of yaml file
            # send kubectl cmd to cluster
            # copy dataset into persistent volume
            pass
        if item == '3': 
            exit()

    def main(self):
        while True:
            self.__display_main_menu()
            self.__select_main_menu()

    def __create_job(self):
        name = input('Please enter [job] name: ')
        K8marsJobCreator_obj = K8marsJobCreator(name)
        K8marsJobCreator_obj.write_yaml_file()
        self.__manager.add_new_job(K8marsJobCreator_obj)
        # K8marsJobCreator_obj.show_task_menu()
        # K8marsJobCreator_obj.select_task_menu()



if __name__ == '__main__':
    main = Wzlk8toolkitMain()
    main.main()
