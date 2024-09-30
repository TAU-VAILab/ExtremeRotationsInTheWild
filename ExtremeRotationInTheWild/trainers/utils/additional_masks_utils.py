import torch
import numpy as np
import cv2
from torchvision import transforms
from einops import rearrange

from .path_to_submodules import *
from src.loftr import LoFTR, default_cfg
from semseg.models import *
from semseg.datasets import *

class SegmenationMaskCreator:
    def __init__(self):
        self.segmodel = eval('SegFormer')(
            backbone='MiT-B3',
            num_classes=150
        )

        try:
            self.segmodel.load_state_dict(torch.load(SEGFORMER_MODEL_PATH, map_location='cpu'))
        except:
            print("Download a pretrained model's weights from the result table.")
        self.segmodel.to('cuda:0')
        self.segmodel.eval()
        print('Loaded Model')
        self.palette = eval('ADE20K').PALETTE
        self.labels = eval('ADE20K').CLASSES
        valid_classes = ["sky","streetlight","road","sidewalk","building"]
        valid_labels = [np.where(np.asarray(self.labels)==v_class)[0][0] for v_class in valid_classes]
        valid_classes_dict = dict(zip(valid_classes, valid_labels))
        self.valid_labels = valid_classes_dict
        self.road_like_labels = [valid_classes_dict["road"],valid_classes_dict["sidewalk"]]
    
    def post_process_seg_map(self, seg, height_feat):
        seg = seg.softmax(1).argmax(1).to(int).unsqueeze(1)
        mask_valid = torch.isin(seg, torch.tensor(list(self.valid_labels.values())).cuda())
        seg = torch.where(mask_valid, seg, torch.zeros_like(seg).cuda())
        mask_road = torch.isin(seg, torch.tensor(self.road_like_labels).cuda())
        seg = torch.where(mask_road, torch.ones_like(seg)*self.valid_labels["road"],seg)
        from_values = [self.valid_labels["road"], self.valid_labels["streetlight"]]
        to_values = [3, 4]
        for from_val, to_val in zip(from_values, to_values):
            seg = torch.where(seg == from_val, torch.tensor(to_val), seg)
        seg_new = transforms.functional.resize(seg,(height_feat,height_feat),antialias=True,interpolation=transforms.InterpolationMode.NEAREST_EXACT)
        return seg_new
    
    def create_pair_seg_mask(self,img1,img2,height_feat):
        with torch.no_grad():
            
            seg1 = self.segmodel(img1)
            seg1 = self.post_process_seg_map(seg1,height_feat)
            seg2 = self.segmodel(img2)
            seg2 = self.post_process_seg_map(seg2,height_feat)
            return seg1,seg2
            

'''This code partially taken from doppelgangers - 
    doppelgangers/datasets/pairwise_disambiguation_dataset.py.
    https://github.com/RuojinCai/doppelgangers.git
    Reference:
    [1] Ruojin Cai, Joseph Tung, Qianqian Wang, Hadar Averbuch-Elor, Bharath Hariharan, Noah Snavely
        Doppelgangers: Learning to Disambiguate Images of Similar Structures. arXiv:2309.02420
'''
class KPandMatchesMasksCreator:
    def __init__(self): 
        matcher = LoFTR(config=default_cfg)
        matcher.load_state_dict(torch.load(LOFTR_MODEL_PATH)['state_dict'])
        self.matcher = matcher.eval().cuda()
           
    def create_masks_from_keypoints(self,kp0,kp1,confidence,batch_indexes,batch_size,feat_size,scale = 8.0):
        result = np.zeros((batch_size,2,2*feat_size,feat_size))
        for i in range(batch_size):
            rel_kp = batch_indexes== i
            kp0_i = kp0[rel_kp].cpu().numpy()
            kp1_i = kp1[rel_kp].cpu().numpy()
            conf_i = confidence[rel_kp].cpu().numpy()
            if np.sum(conf_i>0.8) == 0:
                matches = None
            else:
                F, mask = cv2.findFundamentalMat(kp0_i[conf_i>0.8],kp1_i[conf_i>0.8],cv2.FM_RANSAC, 1, 0.99)
                if mask is None or F is None:
                    matches = None
                else: 
                    matches = np.array(np.ones((kp0_i.shape[0], 2)) * np.arange(kp0_i.shape[0]).reshape(-1,1)).astype(int)[conf_i>0.8][mask.ravel()==1]
            kp_mask_0,kp_mask_1 = np.zeros((1,feat_size,feat_size)),np.zeros((1,feat_size,feat_size))
            kp_mask_0[0,(kp0_i[:,1]/scale).astype(int),(kp0_i[:,0]/scale).astype(int)] = 1
            kp_mask_1[0,(kp1_i[:,1]/scale).astype(int),(kp1_i[:,0]/scale).astype(int)] = 1
            kp_mask =  np.concatenate((kp_mask_0,kp_mask_1),axis=1)
            matches_mask_0,matches_mask_1 = np.zeros((1,feat_size,feat_size)),np.zeros((1,feat_size,feat_size))
            if matches is not None:
                matches_mask_0[0,(kp0_i[matches,1]/scale).astype(int),(kp0_i[matches,0]/scale).astype(int)] = 1
                matches_mask_1[0,(kp1_i[matches,1]/scale).astype(int),(kp1_i[matches,0]/scale).astype(int)] = 1
            matches_mask =  np.concatenate((matches_mask_0,matches_mask_1),axis=1)
            result[i] = np.concatenate((kp_mask,matches_mask),axis=0)
        return result   
    
    def create_pair_loftr_masks(self,batch):
       with torch.no_grad():
            self.matcher(batch)
            
            image_feature_map1 = batch['feat_c0']
            image_feature_map2 = batch['feat_c1']
            height_feat = int(batch['image0'].shape[2]/8)
            image_feature_map1= rearrange(image_feature_map1, 'b (h w) c -> b c h w', h= height_feat)
            image_feature_map2= rearrange(image_feature_map2, 'b (h w) c -> b c h w', h= height_feat)

            kp0 = batch['mkpts0_f']
            kp1 = batch['mkpts1_f']
            confidence = batch['mconf']
            batch_indexes = batch['b_ids']
            mask = self.create_masks_from_keypoints(kp0,kp1,confidence,batch_indexes,batch['bs'],image_feature_map1.shape[2],scale = 8.0)
            return image_feature_map1,image_feature_map2,mask
            