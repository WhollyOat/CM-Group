from yacs.config import CfgNode

dataset_cfg = CfgNode()

# config for dataset
dataset_cfg.cmgroup_crop = CfgNode()
dataset_cfg.cmgroup_crop.num_id = 233
dataset_cfg.cmgroup_crop.num_cam = 6
dataset_cfg.cmgroup_crop.data_root = "/media/data2/CM-Group-crop"

