import torch
import numpy as np
from torch.utils import data
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
from torchvision import transforms
from PIL import ImageFile
from trainers.utils.compute_utils import *
from datasets.dataset_utils import *
from torch.utils.data.sampler import WeightedRandomSampler
from datasets.different_aspect_ratio_utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

class RotationDataset(VisionDataset):
    def __init__(self, root, loader, extensions=None, height=None, pairs_file=None, transform=None,
                 target_transform=None, Train=True,data_type="panorama",stage_type = "90_fov",to_gray_transform=None):
        super(RotationDataset, self).__init__(root, transform=transform,
                                           target_transform=target_transform),
        self.random_ar = BatchRandomAspectRatio()
        self.to_gray_transform=to_gray_transform
        self.pairs = np.load(pairs_file, allow_pickle=True).item()
        self.loader = loader
        self.extensions = extensions
        self.train = Train
        self.height = height
        self.stage = stage_type
        self.data_type = data_type


    def __getitem__(self, index):

        img1 = self.pairs[index]['img1']
        img2 = self.pairs[index]['img2']
        path = os.path.join(self.root, img1['path'])
        if self.data_type == "colmap":
            overlap_amount = self.pairs[index]['overlap_amount']
            scene = self.pairs[index]['scene']
            q1 = [img1['qw'],img1['qx'],img1['qy'],img1['qz']]
            size1 = [img1['width'],img1['height']]
            fov1 = np.degrees(2*np.arctan(min(size1[0],size1[1])/(2*img1['fx'])))
        else:
            rotation_x1, rotation_y1 = img1['x'], img1['y']
            fov1 = np.rad2deg(img1['fov'])
            overlap_amount = self.pairs[index]['overlap_amount']
        ###
        image1 = self.loader(path)
        if self.stage in ["d_fov", "d_im"]:
            self.random_ar()
            image1 = imread_cv2(path)
            image1 = self.transform(get_cropped_image_w_ar(image1,fov1,self.random_ar.chosen_resolution,self.random_ar.portrait))
        elif self.stage in ["90_fov", "ELP"]:
            image1 = self.loader(path)
            image1,_ = get_resized_and_pad(image1, img_size=self.height)
        else:
            raise NotImplementedError
            
        grayimg1 = self.to_gray_transform(image1)
        if self.target_transform is not None:
            image1 = self.target_transform(image1)
        
        path2 = os.path.join(self.root, img2['path'])

        if self.data_type == "colmap":
            q2 = [img2['qw'],img2['qx'],img2['qy'],img2['qz']]
            size2 = [img2['width'],img2['height']]
            fov2 = np.degrees(2*np.arctan(min(size2[0],size2[1])/(2*img2['fx'])))
        else:
            rotation_x2, rotation_y2 = img2['x'], img2['y']
            fov2 = np.rad2deg(img2['fov'])
 
        
        if self.stage in ["d_fov", "d_im"]:
            image2 = imread_cv2(path2)
            image2 = self.transform(get_cropped_image_w_ar(image2,fov2,self.random_ar.chosen_resolution,self.random_ar.portrait))
        elif self.stage in ["90_fov", "ELP"]:
            image2 = self.loader(path2)
            image2,_ = get_resized_and_pad(image2, img_size=self.height)
        else:
            raise NotImplementedError


        grayimg2 = self.to_gray_transform(image2)
        if self.target_transform is not None:
            image2 = self.target_transform(image2)

        if self.data_type == "colmap":
            out =  {
                'img1': image1,
                'q1': q1,
                'img2': image2,
                'q2': q2,
                'path': path,
                'path2': path2,
                'overlap_amount': overlap_amount,
                'scene' : scene,
                'grayimg1': grayimg1,
                'grayimg2': grayimg2,
            }
            return out
            
        else:
            return {
                'img1': image1,
                'rotation_x1': rotation_x1,
                'rotation_y1': rotation_y1,
                'img2': image2,
                'rotation_x2': rotation_x2,
                'rotation_y2': rotation_y2,
                'path': path,
                'path2': path2,
                'grayimg1': grayimg1,
                'grayimg2': grayimg2,
                'overlap_amount': overlap_amount,
            }

    def __len__(self):
        #if len(self.pairs) > 1000 and not self.train:
        #    return 1000
        return len(self.pairs)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_datasets(cfg):
    tr_dataset_second= None
    te_dataset_second= None
    transform = transforms.Compose([transforms.ToTensor()])
    target_transform=transforms.Compose([transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    gray_transform=transforms.Compose([transforms.Grayscale()])
    tr_dataset = RotationDataset(cfg.train.path, default_loader, '.png', height=cfg.height,
                              pairs_file=cfg.train.pairs_file,
                              transform=transform,
                              target_transform=target_transform,data_type = cfg.train.data_type, stage_type = cfg.stage_type,
                              to_gray_transform=gray_transform
                              )
    if hasattr(cfg.train,"additional_pairs_file"):
        tr_dataset_second = RotationDataset(cfg.train.additional_path, default_loader, '.png', height=cfg.height,
                              pairs_file=cfg.train.additional_pairs_file,
                              transform=transform,
                              target_transform=target_transform,data_type = cfg.train.data_type_second, stage_type = cfg.stage_type,
                              to_gray_transform=gray_transform
                              )
        
    te_dataset = RotationDataset(cfg.val.path, default_loader, '.png', height=cfg.height, pairs_file=cfg.val.pairs_file,
                              transform=transform,
                              target_transform=target_transform,
                              Train=False,data_type = cfg.val.data_type, stage_type = cfg.stage_type,
                              to_gray_transform=gray_transform)
    if hasattr(cfg.val,"additional_pairs_file"):
        te_dataset_second = RotationDataset(cfg.val.additional_path, default_loader, '.png', height=cfg.height, pairs_file=cfg.val.additional_pairs_file,
                              transform=transform,
                              target_transform=target_transform,
                              Train=False,data_type = cfg.val.data_type_second, stage_type = cfg.stage_type,
                              to_gray_transform=gray_transform)
        
    return tr_dataset, te_dataset, tr_dataset_second ,te_dataset_second
def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)
    
def get_angle(data_type,pair):
    img1 = pair['img1']
    img2 = pair['img2']
    if data_type == "colmap":
        q1 = [img1['qw'],img1['qx'],img1['qy'],img1['qz']]
        q2 = [img2['qw'],img2['qx'],img2['qy'],img2['qz']]
        relmat = compute_gt_rmat_colmap(torch.tensor(q1,requires_grad=False), torch.tensor(q2,requires_grad=False), 1)
    else:
        rotation_x1, rotation_y1 = img1['x'], img1['y']
        rotation_x2, rotation_y2 = img2['x'], img2['y']
        relmat = compute_gt_rmat(torch.tensor(rotation_x1,requires_grad=False), torch.tensor(rotation_y1,requires_grad=False), 
                                 torch.tensor(rotation_x2,requires_grad=False), torch.tensor(rotation_y2,requires_grad=False), 1)
    angle = compute_angle_from_r_matrices(relmat.cuda())
    return angle
    

def make_weights_for_balanced_classes(dataset, nclasses=360):                        
    angles = [int(torch.floor(get_angle(dataset.data_type,dataset.pairs[i])*180/pi).item()) 
              for i in range(len(dataset.pairs))]                                                
    class_counts = [angles.count(i) for i in range(nclasses)]                                                                            
    num_samples = float(sum(class_counts))                                                   
    class_weights = [num_samples/class_counts[i] if class_counts[i] > 0 else 0 
                     for i in range(len(class_counts))]     
                           
    weights = [class_weights[angles[i]] for i in range(len(dataset.pairs))]                                                                        
    return weights,num_samples

def get_data_loaders(cfg):
    tr_dataset, te_dataset ,tr_dataset_second, te_dataset_second = get_datasets(cfg)
    if hasattr(cfg.train,"sampler") :
        if cfg.train.sampler == "weighted_random":
            weights,num_samples = make_weights_for_balanced_classes(tr_dataset)
            sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))  

            train_loader = data.DataLoader(
                dataset=tr_dataset, batch_size=cfg.train.batch_size,
                shuffle=False, num_workers=cfg.num_workers, drop_last=True,
                worker_init_fn=init_np_seed,sampler=sampler)
        else:
            raise NotImplementedError
    else:
        train_loader = data.DataLoader(
            dataset=tr_dataset, batch_size=cfg.train.batch_size,
            shuffle=True, num_workers=cfg.num_workers, drop_last=True,
            worker_init_fn=init_np_seed)
        
    test_loader = data.DataLoader(
        dataset=te_dataset, batch_size=cfg.val.batch_size,
        shuffle=False, num_workers=cfg.num_workers, drop_last=False,
        worker_init_fn=init_np_seed)
    
    loaders = {
            "test_loader": test_loader,
            'train_loader': train_loader,
        }
    
    if te_dataset_second is not None:
        test_loader_second = data.DataLoader(
            dataset=te_dataset_second, batch_size=cfg.val.batch_size,
            shuffle=False, num_workers=cfg.num_workers, drop_last=False,
            worker_init_fn=init_np_seed)
        loaders["test_loader_second"] =  test_loader_second
        
    if tr_dataset_second is not None:
        if hasattr(cfg.train,"second_sampler") :
            if cfg.train.second_sampler == "weighted_random":
                weights,num_samples = make_weights_for_balanced_classes(tr_dataset)
                sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
                train_loader_second = data.DataLoader(
                    dataset=tr_dataset_second, batch_size=cfg.train.batch_size,
                    shuffle=False, num_workers=cfg.num_workers, drop_last=True,
                    worker_init_fn=init_np_seed,sampler=sampler)
            else:
                raise NotImplementedError 
        else: 
            train_loader_second = data.DataLoader(
                dataset=tr_dataset_second, batch_size=cfg.train.batch_size,
                shuffle=True, num_workers=cfg.num_workers, drop_last=True,
                worker_init_fn=init_np_seed)
        loaders["train_loader_second"] =  train_loader_second
 
    return loaders



if __name__ == "__main__":
    pass
