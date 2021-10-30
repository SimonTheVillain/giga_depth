import os


os.system("ls /home/simon/pycharm")

command = "/home/simon/pycharm/GigaDepth/venv/bin/python /home/simon/pycharm/GigaDepth/train.py --dataset_path=/media/simon/ssd_datasets/datasets/structure_core_unity_sequences --config_file configs/full_66_lcn_j4.yaml --gpu_list 0"
os.system(command)
command = "/home/simon/pycharm/GigaDepth/venv/bin/python /home/simon/pycharm/GigaDepth/train.py --dataset_path=/media/simon/ssd_datasets/datasets/structure_core_unity_sequences --config_file configs/full_66_j4.yaml --gpu_list 0"
os.system(command)