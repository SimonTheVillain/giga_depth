python3 train.py --dataset_path=/media/simon/T7/datasets/DepthInSpace/rendered_default --config_file=configs/dis_def_lcn_c960.yaml -b 1
python3 train.py --dataset_path=/media/simon/T7/datasets/DepthInSpace/rendered_kinect --config_file=configs/dis_kin_lcn_c960.yaml -b 1
python3 train.py --dataset_path=/media/simon/T7/datasets/DepthInSpace/rendered_real --config_file=configs/dis_real_lcn_c960.yaml -b 1
python3 train.py --dataset_path=/media/simon/T7/datasets/DepthInSpace/rendered_real --config_file=configs/dis_real_lcn_j2_c960.yaml -b 1
