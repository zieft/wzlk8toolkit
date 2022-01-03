from core.k8mars_control import *
import os
from pathlib import Path

class PathInfo:
    """
    Path of directory always ends with '/'
    """
    ### Working Directory ###
    work_dir = Path(os.getcwd())
    k8mars_dir = Path(os.path.abspath(os.path.join(os.getcwd(), '..')))
    yaml_templates_dir = work_dir / 'yaml_templates/'
    cache_folder = work_dir / 'k8marsCache/'

    ### data location ###


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
            selected_job_obj = self.__manager.select_job()
            kubectl_controller_obj = KubectlController(selected_job_obj)
            kubectl_controller_obj.show_options_for_job()
            kubectl_controller_obj.select_option_for_job()
            # show list of jobs with index : done
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
        K8marsJobCreator_obj = K8marsJobCreator()
        K8marsJobCreator_obj.write_yaml_file()
        self.__manager.add_new_job(K8marsJobCreator_obj)




if __name__ == '__main__':
    main = Wzlk8toolkitMain()
    main.main()
