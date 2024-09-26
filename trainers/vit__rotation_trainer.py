import os
import tqdm
import torch
import importlib
import numpy as np
from trainers.base_trainer import BaseTrainer
from trainers.utils.loss_utils import *
from evaluation.evaluation_metrics import *
from trainers.utils.additional_masks_utils import *
from trainers.utils.out_utils import *


class Trainer(BaseTrainer):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args

        dn_lib = importlib.import_module(cfg.models.rotationnet.type)
        self.rotation_net = dn_lib.RotationNet(cfg.models.rotationnet)
        self.rotation_net.cuda()
        print("rotationnet:")
        print(self.rotation_net)

        # The optimizer
        if not hasattr(self.cfg.trainer, "opt_dn"):
            self.cfg.trainer.opt_dn = self.cfg.trainer.opt

        if getattr(self.cfg.trainer.opt_dn, "scheduler", None) is not None:
            self.opt_dn, self.scheduler_dn = get_opt(
                list(self.rotation_net.parameters()), self.cfg.trainer.opt_dn)
        else:
            self.opt_dn = get_opt(
                list(self.rotation_net.parameters()), self.cfg.trainer.opt_dn)
            self.scheduler_dn = None
        
        self.classification = getattr(self.cfg.trainer, "classification", True)
        self.pairwise_type = getattr(self.cfg.trainer, "pairwise_type", "concat")
        self.rotation_parameterization = getattr(self.cfg.trainer, "rotation_parameterization", False)
        self.w_translation = getattr(self.cfg.trainer, "w_translation", False)
        # Prepare save directory
        os.makedirs(os.path.join(cfg.save_dir, "checkpoints"), exist_ok=True)
        ###
        self.data_type = getattr(self.cfg.data,"data_type","panorama")
        
        self.kpmodel = KPandMatchesMasksCreator()
        self.segmodel = SegmenationMaskCreator()
        self.randomization = getattr(self.cfg.trainer, "randomization", 1.)
        self.randomization_kp = getattr(self.cfg.trainer, "randomization_kp", 1.)
        self.top_picks = getattr(self.cfg.trainer, "top_picks", 0)
        ###
 
    def epoch_end(self, epoch, writer=None):
        if self.scheduler_dn is not None:
            self.scheduler_dn.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_dn_lr', self.scheduler_dn.get_lr()[0], epoch)
                
    def update(self, data_full, no_update=False,data_type = "panorama"):

        if data_type == "colmap":
                q1 = data_full['q1']
                q2 = data_full['q2']
        else:
            rotation_x1 = data_full['rotation_x1']
            rotation_y1 = data_full['rotation_y1']
            rotation_x2 = data_full['rotation_x2']
            rotation_y2 = data_full['rotation_y2']
            
        if 'mask1' in data_full:
            batch = {'image0':  data_full['grayimg1'].cuda(), 'image1': data_full['grayimg2'].cuda(), 'mask0': data_full['mask1'].cuda(), 'mask1': data_full['mask2'].cuda()}
        else:
            batch = {'image0':  data_full['grayimg1'].cuda(), 'image1': data_full['grayimg2'].cuda()}
            
        image_feature_map1,image_feature_map2,kp_and_matches = self.kpmodel.create_pair_loftr_masks(batch)
        
        seg1,seg2 = self.segmodel.create_pair_seg_mask(data_full['img1'].cuda(),data_full['img2'].cuda(),height_feat = int(batch['image0'].shape[2]/8))
            
        batch_size = batch['image0'].size(0)
        if self.randomization < 1.0:           
            random_masks = np.random.uniform(size=batch_size)<=self.randomization
            seg1[random_masks] = torch.zeros_like(seg1[0])
            seg2[random_masks] = torch.zeros_like(seg2[0])    
        if self.randomization_kp < 1.0:
            random_masks = np.random.uniform(size=batch_size)<=self.randomization
            kp_and_matches[random_masks] = torch.zeros_like(kp_and_matches[0])
            
            
        if not no_update:
            self.rotation_net.float().train()
            self.opt_dn.zero_grad()

        
        ###
        if data_type == "colmap":
            gt_rmat =compute_gt_rmat_colmap(q1,q2,batch_size)
        else:
            gt_rmat = compute_gt_rmat(rotation_x1, rotation_y1, rotation_x2, rotation_y2, batch_size)
        
        if not self.classification or self.rotation_parameterization:
                    print("Not Implemented!")
        else:
            angle_x, angle_y, angle_z = compute_euler_angles_from_rotation_matrices(gt_rmat)
        
        input_img = torch.cat([image_feature_map1, image_feature_map2], dim=2)
        input_img = torch.cat([input_img.cuda(), torch.from_numpy(kp_and_matches).cuda()], dim=1)
        seg = torch.cat([seg1, seg2], dim=2)
        input_img = torch.cat([input_img, seg.cuda(),torch.zeros_like(seg).cuda()], dim=1)

        # loss type
        if not self.classification:
            # regression loss
            print("Not Implement!")
        else:
            # classification loss
            out_rotation_x, out_rotation_y, out_rotation_z = self.rotation_net(input_img.float())
            ###
            loss_x = rotation_loss_class(out_rotation_x, angle_x)
            loss_y = rotation_loss_class(out_rotation_y, angle_y)
            loss_z = rotation_loss_class(out_rotation_z, angle_z)

            loss = loss_x + loss_y + loss_z
            res1 = {"loss": loss, "loss_x": loss_x, "loss_y": loss_y, "loss_z": loss_z}



        if not no_update:
            loss.backward()
            self.opt_dn.step()
        else:
            self.opt_dn.zero_grad()
        train_info = {}
        train_info.update(res1)
        train_info.update({"loss": loss})
        return train_info

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False,train_info_second=None,train_info_third=None,trainers_names = None):
        if writer is not None:
            for k, v in train_info.items():
                if not ('loss' in k) and not ('Error' in k):
                    continue
                if step is not None:
                    if train_info_second is not None:
                        if train_info_third is not None:
                            writer.add_scalars(k, {trainers_names[0] : v,
                                               trainers_names[1] : train_info_second[k],
                                               trainers_names[2] : train_info_third[k]} , step)
                        else:
                            writer.add_scalars(k, {trainers_names[0] : v,
                                                   trainers_names[1] : train_info_second[k]} , step)
                    else:
                        writer.add_scalar('train/' + k, v, step)
                else:
                    assert epoch is not None
                    writer.add_scalar('train/' + k, v, epoch)
                    


    def validate(self, test_loader, epoch, val_angle=False,save_pictures=False):
        print("Validation")
        out_rmat_array = None
        gt_rmat_array = None
        gt_rmat1_array = None
        out_rmat1_array = None
        overlap_amount_array = None
        scene_array = None 
        
        with torch.no_grad():
            self.rotation_net.float().eval()
            
            for i,data_full in enumerate(tqdm.tqdm(test_loader)):
                if test_loader.dataset.data_type == "colmap":
                    q1 = data_full['q1']
                    q2 = data_full['q2']
                else:
                    rotation_x1 = data_full['rotation_x1']
                    rotation_y1 = data_full['rotation_y1']
                    rotation_x2 = data_full['rotation_x2']
                    rotation_y2 = data_full['rotation_y2']

                batch = {'image0':  data_full['grayimg1'].cuda(), 'image1': data_full['grayimg2'].cuda()}
                    
                image_feature_map1,image_feature_map2,kp_and_matches = self.kpmodel.create_pair_loftr_masks(batch)
        
                seg1,seg2 = self.segmodel.create_pair_seg_mask(data_full['img1'].cuda(),data_full['img2'].cuda(),height_feat = int(batch['image0'].shape[2]/8))

                batch_size = batch['image0'].size(0)
                if 'overlap_amount' in data_full:
                    overlap_amount = data_full['overlap_amount']
                    if overlap_amount_array is None:
                        overlap_amount_array = overlap_amount
                    else:
                        overlap_amount_array = overlap_amount_array + overlap_amount
                    
                if test_loader.dataset.data_type == "colmap":
                    gt_rmat =compute_gt_rmat_colmap(q1,q2,batch_size)
                        
                    scene = data_full['scene']
                    if scene_array is None:
                        scene_array = scene
                    else:
                        scene_array = scene_array + scene
                else:
                    gt_rmat = compute_gt_rmat(rotation_x1, rotation_y1, rotation_x2, rotation_y2, batch_size)

                input_img = torch.cat([image_feature_map1, image_feature_map2], dim=2)
                input_img = torch.cat([input_img.cuda(), torch.from_numpy(kp_and_matches).cuda()], dim=1)
                seg = torch.cat([seg1, seg2], dim=2)
                input_img = torch.cat([input_img, seg.cuda(),torch.zeros_like(seg).cuda()], dim=1)
                if not self.classification or self.rotation_parameterization:
                    print("Not Implemented!")
                else:
                    out_rotation_x, out_rotation_y, out_rotation_z = self.rotation_net(input_img.float())
                    if self.top_picks == 0:                     
                        out_rmat, out_rmat1 = compute_out_rmat_from_euler(out_rotation_x, out_rotation_y, out_rotation_z, batch_size)
                    else:
                        out_rmat, _, _ = compute_out_rmat_from_euler_top_k(out_rotation_x, out_rotation_y, out_rotation_z, batch_size,gt_rmat,top_picks = self.top_picks)
                
                if gt_rmat_array is None:
                    gt_rmat_array = gt_rmat
                else:
                    gt_rmat_array = torch.cat((gt_rmat_array, gt_rmat))
                if out_rmat_array is None:
                    out_rmat_array = out_rmat
                else:
                    out_rmat_array = torch.cat((out_rmat_array, out_rmat))
 
                if val_angle:
                    gt_rmat1 = compute_rotation_matrix_from_viewpoint(rotation_x1, rotation_y1, batch_size).view(batch_size, 3, 3).cuda()
                    if gt_rmat1_array is None:
                        gt_rmat1_array = gt_rmat1
                    else:
                        gt_rmat1_array = torch.cat((gt_rmat1_array, gt_rmat1))
                    if out_rmat1_array is None:
                        out_rmat1_array = out_rmat1
                    else:
                        out_rmat1_array = torch.cat((out_rmat1_array, out_rmat1))
            if overlap_amount_array is None:
                res_error = evaluation_metric_rotation(out_rmat_array, gt_rmat_array)
            else:
                res_error = evaluation_metric_rotation(out_rmat_array, gt_rmat_array,overlap_amount_array)
            if val_angle:
                angle_error = evaluation_metric_rotation_angle(out_rmat_array, gt_rmat_array, gt_rmat1_array, out_rmat1_array)
                res_error.update(angle_error)

            # mean, median, max, std, 10deg
            all_res = statistics_from_res(res_error)
            print("Validation Epoch:%d " % epoch, all_res)
        return all_res

    def log_val(self, val_info, writer=None, step=None, epoch=None,val_info_second=None,tests_names = None):
        if writer is not None:
            for k, v in val_info.items():
                if step is not None:
                    if 'vis' in k:
                        writer.add_image(k, v, step)
                    else:
                        if val_info_second is not None:
                            writer.add_scalars(k, {tests_names[0] : v,
                                                   tests_names[1] : val_info_second[k]} , step)
                        else:
                            writer.add_scalar(k, v, step)
                else:
                    if 'vis' in k:
                        writer.add_image(k, v, epoch)
                    else:
                        if val_info_second is not None:
                            writer.add_scalars(k, {tests_names[0] : v,
                                                   tests_names[1] : val_info_second[k]} , step)
                        else:
                            writer.add_scalar(k, v, step)

    def save(self, epoch=None, step=None, appendix=None):
        d = {
            'opt_dn': self.opt_dn.state_dict(),
            'dn': self.rotation_net.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s.pt" % (epoch)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)
        remove_name = "epoch_%s.pt" % (epoch-1)
        remove_path = os.path.join(self.cfg.save_dir, "checkpoints", remove_name)
        #if os.path.exists(remove_path) and (epoch-1)!=9:
        #    os.remove(remove_path)

    def resume(self, path, strict=True, resume_encoder=False, resume_dn_opt=False, **args):
        ckpt = torch.load(path)
        if not resume_encoder:
            from collections import OrderedDict    
            new_state_dict  = OrderedDict()
            for k, v in ckpt['dn'].items():
                new_state_dict[k.replace("module.", "",1)] = v
            self.rotation_net.load_state_dict(new_state_dict,strict=strict)
            if resume_dn_opt:
                self.opt_dn.load_state_dict(ckpt['opt_dn'])
            start_epoch = ckpt['epoch']
        else:
            start_epoch = 0
        return start_epoch

    def test(self, opt, *arg, **kwargs):
        raise NotImplementedError("Trainer [test] not implemented.")