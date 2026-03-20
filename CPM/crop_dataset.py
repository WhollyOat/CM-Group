import os
import json
from PIL import Image


origin_data_root = './CM-Group'
save_data_root = './CM-Group-crop'

isExist = os.path.exists(save_data_root)
if not isExist:
    os.makedirs(save_data_root) 


for camera_id in range(1,7):
    camera_dir = os.path.join(save_data_root, 'camera'+ str(camera_id))
    isExist = os.path.exists(camera_dir)
    if not isExist:
        os.makedirs(camera_dir) 

    for gid in range(1,429):
        if gid==170:
            continue 
            
        if gid<10:
            str_gid= "00"+ str(gid)
        elif gid<100:
            str_gid= "0"+ str(gid)
        else:
            str_gid= str(gid)
        
        gid_dir = os.path.join(origin_data_root, 'camera'+ str(camera_id), str_gid)

        for img_name in os.listdir(gid_dir):
            json_path = os.path.join(origin_data_root, 'json', str_gid, img_name.split('.')[0]+ '.json')
            with open(json_path, 'r') as jsonFile:
             labelFile = json.load(jsonFile)
             jsonFile.close()

            bboxs = labelFile['bbox']
            pids = labelFile['pid']
            for pid in pids:
                save_data_dir = os.path.join(camera_dir, str_gid+'_'+str(pid))
                isExist = os.path.exists(save_data_dir)
                if not isExist:
                    os.makedirs(save_data_dir) 
                
                bbox= bboxs[pids.index(pid)]
                img = Image.open(os.path.join(gid_dir, img_name))
                img = img.crop(bbox)
                img.save(os.path.join(save_data_dir, img_name.split('.')[0].split('_')[-1]+ '.jpeg'))





