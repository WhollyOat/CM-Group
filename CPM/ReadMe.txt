Here are a few steps to run CPM on your devices:
1. Please visit our repository on 'https://github.com/WhollyOat/CM-Group', sign the release agreement and request for the original dataset CM-Group by email. 
2. Put CM-Group in a proper directory and run crop_dataset.py to get the cropped images CM-Group-crop.
3. Edit the file '/configs/default/dataset.py' and replace with your data_root.
4. Input 'export CUDA_VISIBLE_DEVICES=0' to select a gpu you would like to run.
5. Input 'python train.py --cfg ./configs/CMGroup.yml' to start training and testing.

The code is based on MPANet (Wu et al. 2021), you can refer to the original paper and code to see more details.

Please contact us if you have any problems with CM-Group and the code.

The code is for ACADEMIC use only.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Citation:
Please cite the following paper if you use this repository in your research work.

@inproceedings{DBLP:conf/cvpr/XiongL23,
  author       = {Jianghao Xiong and
                  Jianhuang Lai},
  title        = {Similarity Metric Learning For RGB-Infrared Group Re-Identification},
  booktitle    = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                  {CVPR} 2023},
  pages        = {13662--13671},
  year         = {2023},
  doi          = {10.1109/CVPR52729.2023.01313}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Reference:
@INPROCEEDINGS{9577837,
  author={Wu, Qiong and Dai, Pingyang and Chen, Jie and Lin, Chia-Wen and Wu, Yongjian and Huang, Feiyue and Zhong, Bineng and Ji, Rongrong},
  booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Discover Cross-Modality Nuances for Visible-Infrared Person Re-Identification}, 
  year={2021},
  volume={},
  number={},
  pages={4328-4337},
  doi={10.1109/CVPR46437.2021.00431}}



