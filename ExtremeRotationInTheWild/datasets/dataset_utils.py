####from doppelgangers
import numpy as np
import torch
import torchvision.transforms as T

def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new

def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    if w_new == 0:
        w_new = df
    if h_new == 0:
        h_new = df
    return w_new, h_new

def get_resized_and_pad(img_raw, img_size=256, df=8, padding=True):
    if not isinstance(img_raw, torch.Tensor):
        img_raw = T.ToTensor()(img_raw)
    w, h ,c = img_raw.shape[2], img_raw.shape[1], img_raw.shape[0]
    w_new, h_new = get_resized_wh(w, h, img_size)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    if padding:  # padding
        pad_to = max(h_new, w_new)    
        mask = np.zeros((1,pad_to, pad_to), dtype=bool)
        cen = int(pad_to/2)
        mask[:,cen-int(h_new/2):cen+int(h_new/2),cen-int(w_new/2):cen+int(w_new/2)] = True
        mask = mask[:,::8,::8]
    transform = T.Resize((h_new,w_new),antialias=True)
    image = transform(img_raw)
    pad_image = torch.zeros((1,c, pad_to, pad_to), dtype=torch.float32)
    
    pad_image[0,:,cen-int(h_new/2):cen+int(h_new/2),cen-int(w_new/2):cen+int(w_new/2)]=image

    return pad_image.squeeze(0), mask.squeeze(0)

def get_crop_square_and_resize(img_raw, img_size=256, df=8, padding=True):
    if not isinstance(img_raw, torch.Tensor):
        img_raw = T.ToTensor()(img_raw)
    w, h = img_raw.shape[2], img_raw.shape[1]
    new_size = min(w,h)
    transform = T.Compose([T.CenterCrop((new_size,new_size)),T.Resize((img_size,img_size),antialias=True)])
    image = transform(img_raw)
    return image

