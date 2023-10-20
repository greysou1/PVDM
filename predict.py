# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import warnings
warnings.filterwarnings('ignore')
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from tensorfn import load_config
import numpy as np
import torchvision.transforms as transforms
import torchvision
import os, glob, cv2, time, shutil
import torch.distributed as dist

from config.diffconfig import DiffusionConfig
from models.unet_autoenc import BeatGANsAutoencConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps


def get_conf():

    return BeatGANsAutoencConfig(image_size=256, 
    in_channels=3+20, 
    model_channels=128, 
    out_channels=3*2,  # also learns sigma
    num_res_blocks=2, 
    num_input_res_blocks=None, 
    embed_channels=512, 
    attention_resolutions=(32, 16, 8,), 
    time_embed_channels=None, 
    dropout=0.1, 
    channel_mult=(1, 1, 2, 2, 4, 4), 
    input_channel_mult=None, 
    conv_resample=True, 
    dims=2, 
    num_classes=None, 
    use_checkpoint=False,
    num_heads=1, 
    num_head_channels=-1, 
    num_heads_upsample=-1, 
    resblock_updown=True, 
    use_new_attention_order=False, 
    resnet_two_cond=True, 
    resnet_cond_channels=None, 
    resnet_use_zero_module=True, 
    attn_checkpoint=False, 
    enc_out_channels=512, 
    enc_attn_resolutions=None, 
    enc_pool='adaptivenonzero', 
    enc_num_res_block=2, 
    enc_channel_mult=(1, 1, 2, 2, 4, 4, 4), 
    enc_grad_checkpoint=False, 
    latent_net_conf=None)

class Predictor():
    def __init__(self, npy_dir=None, ckpt_path=None, start_frame_idx=0, end_frame_idx=-1, step_frame_idx=1):
        """Load the model into memory to make running multiple predictions efficient"""
        torch.manual_seed(12)
        torch.cuda.manual_seed(12)
        np.random.seed(12)
        random.seed(12)

        torch.backends.cudnn.deterministic=True
        
        self.start_frame_idx = start_frame_idx
        
        conf = load_config(DiffusionConfig, "config/fashion.conf", show=False)

        if ckpt_path is None:
            ckpt = torch.load("checkpoints/last.pt")
        else:
            ckpt = torch.load(ckpt_path)
        self.model = get_conf().make_model()
        self.model.load_state_dict(ckpt["ema"], strict=False)
        self.model = self.model.cuda()
        self.model.eval()

        self.betas = conf.diffusion.beta_schedule.make()
        self.diffusion = create_gaussian_diffusion(self.betas, predict_xstart = False)#.to(device)
        
        if npy_dir is None:
            self.pose_list = glob.glob('data/deepfashion_256x256/target_pose/*.npy')
        else:
            pose_list = glob.glob(npy_dir+'*.npy')
            try:
                self.pose_list = sorted(pose_list, key=lambda x: int(x.split('/')[-1].replace('frame', '').split('.')[0]))[start_frame_idx:end_frame_idx:step_frame_idx]
            except:
                self.pose_list = sorted(pose_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))[start_frame_idx:end_frame_idx:step_frame_idx]
        
        self.transforms = transforms.Compose([transforms.Resize((256,256), interpolation=Image.BICUBIC),
                            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])

    def predict_pose_exp(
        self,
        image,
        num_poses=1,
        sample_algorithm='ddim',
        nsteps=100,

        ):
        """Run a single prediction on the model"""

        src = Image.open(image)
        src = self.transforms(src).unsqueeze(0).cuda()
        tgt_pose = torch.stack([transforms.ToTensor()(np.load(ps)).cuda() for ps in np.random.choice(self.pose_list, num_poses)], 0)

        print("tgt_pose.shape = ", tgt_pose.shape)
        src = src.repeat(num_poses,1,1,1)




        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, tgt_pose], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src.shape).cuda()
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, tgt_pose])
            samples = xs[-1].cuda()


        samples_grid = torch.cat([src[0],torch.cat([samps for samps in samples], -1)], -1)
        samples_grid = (torch.clamp(samples_grid, -1., 1.) + 1.0)/2.0
        pose_grid = torch.cat([torch.zeros_like(src[0]),torch.cat([samps[:3] for samps in tgt_pose], -1)], -1)
        
        print("pose_grid.shape = ", pose_grid.shape)
        output = torch.cat([1-pose_grid, samples_grid], -2)

        print("numpy_imgs.shape = ", numpy_imgs)
        fake_imgs = (255*numpy_imgs).astype(np.uint8)
        print("fake_imgs[0].shape = ", fake_imgs[0].shape)
        Image.fromarray(fake_imgs[0]).save('output.png')

    def predict_pose(
        self,
        image=None,
        pil_image=None,
        num_poses=1,
        sample_algorithm='ddim',
        output_filename="output",
        save_dir='outputs',
        nsteps=100,

        ):
        """Run a single prediction on the model"""

        if image is not None:
            src = Image.open(image)
        else:
            src = pil_image

        src = self.transforms(src).unsqueeze(0).cuda()
        tgt_pose = torch.stack([transforms.ToTensor()(np.transpose(np.load(ps), (1,2,0))).cuda() for ps in self.pose_list[:num_poses]], 0)

        src = src.repeat(num_poses,1,1,1)

        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, tgt_pose], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src.shape).cuda()
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, tgt_pose])
            samples = xs[-1].cuda()


        samples_grid = torch.cat([src[0],torch.cat([samps for samps in samples], -1)], -1)
        samples_grid = (torch.clamp(samples_grid, -1., 1.) + 1.0)/2.0
        pose_grid = torch.cat([torch.zeros_like(src[0]),torch.cat([samps[:3] for samps in tgt_pose], -1)], -1)

        output = torch.cat([1-pose_grid, samples_grid], -2)

        numpy_imgs = output.unsqueeze(0).permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs = (255*numpy_imgs).astype(np.uint8)
        Image.fromarray(fake_imgs[0]).save(save_dir+"/"+output_filename.split('.')[0]+".png")

    def predict_pose_save_npy(
        self,
        image=None,
        pil_image=None,
        num_poses=1,
        sample_algorithm='ddim',
        output_filename="output",
        save_dir='outputs',
        nsteps=100,

        ):
        """Run a single prediction on the model"""

        torch.manual_seed(12)
        torch.cuda.manual_seed(12)
        np.random.seed(12)
        random.seed(12)

        torch.backends.cudnn.deterministic=True
        
        if image is not None:
            src = Image.open(image)
        else:
            src = pil_image

        src = self.transforms(src).unsqueeze(0).cuda()
        tgt_pose = torch.stack([transforms.ToTensor()(np.transpose(np.load(ps), (1,2,0))).cuda() for ps in self.pose_list[:num_poses]], 0)

        src = src.repeat(num_poses,1,1,1)

        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, tgt_pose], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src.shape).cuda()
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, tgt_pose])
            samples = xs[-1].cuda()

        save_npy_dir = save_dir+output_filename.split('.')[0]+"/samples.npy"
        if not os.path.exists(save_npy_dir): os.makedirs(save_npy_dir)

        samples_np = samples.detach().cpu().numpy() # convert to numpy array
        np.save(save_npy_dir, samples_np) # save as .npy file

        print("saved npy at " + save_dir+"/"+output_filename.split('.')[0]+"/samples.npy")
        samples_grid = torch.cat([src[0],torch.cat([samps for samps in samples], -1)], -1)
        samples_grid = (torch.clamp(samples_grid, -1., 1.) + 1.0)/2.0
        pose_grid = torch.cat([torch.zeros_like(src[0]),torch.cat([samps[:3] for samps in tgt_pose], -1)], -1)

        output = torch.cat([1-pose_grid, samples_grid], -2)

        numpy_imgs = output.unsqueeze(0).permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs = (255*numpy_imgs).astype(np.uint8)
        Image.fromarray(fake_imgs[0]).save(save_dir+"/"+output_filename.split('.')[0]+".png")
        print("Saved at "+save_dir+"/"+output_filename.split('.')[0]+".png")

    def predict_pose_save_imgs(
        self,
        image=None,
        pil_image=None,
        num_poses=1,
        sample_algorithm='ddim',
        output_filename="output",
        save_dir='outputs',
        nsteps=100
        ):
        """Run a single prediction on the model"""

        if image is not None:
            src = Image.open(image)
        else:
            src = pil_image

        src = self.transforms(src).unsqueeze(0).cuda()
        tgt_pose = torch.stack([transforms.ToTensor()(np.transpose(np.load(ps), (1,2,0))).cuda() for ps in self.pose_list[:num_poses]], 0)

        src = src.repeat(num_poses,1,1,1)

        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, tgt_pose], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src[0].shape)
            noise = noise.repeat(num_poses, 1, 1, 1).cuda()
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, tgt_pose])
            samples = xs[-1].cuda()

        print(samples.shape)
        save_path = os.path.join(save_dir, output_filename.split('.')[0])

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i, sample in enumerate(samples):
            # Convert the tensor image to a numpy array
            numpy_img = sample.permute(1, 2, 0).cpu().numpy()
            
            # Scale the pixel values from [-1, 1] to [0, 255]
            numpy_img = ((numpy_img + 1) / 2.0 * 255).astype(np.uint8)
            
            filename = "frame_{}.png".format(i + self.start_frame_idx)
            # Save the image as PNG
            filepath = os.path.join(save_path, filename)
            Image.fromarray(numpy_img).save(filepath)

        samples_grid = torch.cat([src[0],torch.cat([samps for samps in samples], -1)], -1)
        samples_grid = (torch.clamp(samples_grid, -1., 1.) + 1.0)/2.0
        pose_grid = torch.cat([torch.zeros_like(src[0]),torch.cat([samps[:3] for samps in tgt_pose], -1)], -1)

        output = torch.cat([1-pose_grid, samples_grid], -2)

        numpy_imgs = output.unsqueeze(0).permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs = (255*numpy_imgs).astype(np.uint8)
        Image.fromarray(fake_imgs[0]).save(save_dir+"/"+output_filename.split('.')[0]+".png")
    
    def predict_pose_from_pose_list(
        self,
        image,
        num_poses=1,
        sample_algorithm='ddim',
        output_filename="output",
        nsteps=100,
        pose_list=None
        ):
        """Run a single prediction on the model"""

        if pose_list is None:
            pose_list = self.pose_list
        src = Image.open(image)
        src = self.transforms(src).unsqueeze(0).cuda()
        tgt_pose = torch.stack([transforms.ToTensor()(np.load(ps)).cuda() for ps in pose_list[:num_poses]], 0)

        src = src.repeat(num_poses,1,1,1)

        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, tgt_pose], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src.shape).cuda()
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, tgt_pose])
            samples = xs[-1].cuda()


        samples_grid = torch.cat([src[0],torch.cat([samps for samps in samples], -1)], -1)
        samples_grid = (torch.clamp(samples_grid, -1., 1.) + 1.0)/2.0
        pose_grid = torch.cat([torch.zeros_like(src[0]),torch.cat([samps[:3] for samps in tgt_pose], -1)], -1)

        output = torch.cat([1-pose_grid, samples_grid], -2)

        numpy_imgs = output.unsqueeze(0).permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs = (255*numpy_imgs).astype(np.uint8)
        Image.fromarray(fake_imgs[0]).save("outputs/"+output_filename.split('.')[0]+'.png')
    
    def predict_pose_from_npy(
        self,
        image,
        npy_file_path='',
        num_poses=1,
        sample_algorithm='ddim',
        save_dir='outputs',
        output_filename="output",
        nsteps=100,

        ):
        """Run a single prediction on the model"""


        # src = Image.open(image)
        src = image
        src = self.transforms(src).unsqueeze(0).cuda()
        tgt_pose = torch.stack([transforms.ToTensor()(np.transpose(np.load(npy_file_path), (1,2,0))).cuda()], 0)

        src = src.repeat(num_poses,1,1,1)

        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, tgt_pose], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            torch.manual_seed(12)
            torch.cuda.manual_seed(12)
            np.random.seed(12)
            random.seed(12)

            torch.backends.cudnn.deterministic=True
            
            print(torch.randn(5))
            noise = torch.randn(src.shape).cuda()
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, tgt_pose])
            samples = xs[-1].cuda()


        samples_grid = torch.cat([src[0],torch.cat([samps for samps in samples], -1)], -1)
        samples_grid = (torch.clamp(samples_grid, -1., 1.) + 1.0)/2.0
        pose_grid = torch.cat([torch.zeros_like(src[0]),torch.cat([samps[:3] for samps in tgt_pose], -1)], -1)

        output = torch.cat([1-pose_grid, samples_grid], -2)

        numpy_imgs = output.unsqueeze(0).permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs = (255*numpy_imgs).astype(np.uint8)
        Image.fromarray(fake_imgs[0]).save(save_dir+"/"+output_filename.split('.')[0]+".png")

    def predict_pose_loopsave(self,
        image,
        num_poses=1,
        sample_algorithm='ddim',
        nsteps=100,
        save_folder="exp"
        ):
        """ call predict_pose_loopsave_temp nsteps number of times"""

        save_folder = "outputs/loops/" + save_folder
        if not os.path.exists(save_folder): os.makedirs(save_folder)

        # for i in range(1, nsteps+1):
        #     self.predict_pose_loopsave_singlerun(image=image, sample_algorithm=sample_algorithm, num_poses=num_poses, nsteps=i, save_folder=save_folder)
        self.predict_pose_loopsave_singlerun(image=image, sample_algorithm=sample_algorithm, num_poses=num_poses, nsteps=1000, save_folder=save_folder)

    def predict_pose_loopsave_singlerun(
        self,
        image,
        num_poses=1,
        sample_algorithm='ddim',
        nsteps=100,
        save_folder="outputs/loops/"
        ):
        """Run a single prediction on the model"""
        src = Image.open(image)
        src = self.transforms(src).unsqueeze(0).cuda()
        tgt_pose = torch.stack([transforms.ToTensor()(np.load(ps)).cuda() for ps in np.random.choice(self.pose_list, num_poses)], 0)

        src = src.repeat(num_poses,1,1,1)

        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, tgt_pose], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src.shape).cuda()
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, tgt_pose], save_folder=save_folder)
            samples = xs[-1].cuda()


        samples_grid = torch.cat([src[0],torch.cat([samps for samps in samples], -1)], -1)
        samples_grid = (torch.clamp(samples_grid, -1., 1.) + 1.0)/2.0
        pose_grid = torch.cat([torch.zeros_like(src[0]),torch.cat([samps[:3] for samps in tgt_pose], -1)], -1)

        output = torch.cat([1-pose_grid, samples_grid], -2)

        numpy_imgs = output.unsqueeze(0).permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs = (255*numpy_imgs).astype(np.uint8)
        
        # Image.fromarray(fake_imgs[0]).save(f'{save_folder}/nsteps_{nsteps}.png')

    def predict_appearance(
        self,
        image,
        ref_img,
        ref_mask,
        ref_pose,
        sample_algorithm='ddim',
        nsteps=100,

        ):
        """Run a single prediction on the model"""

        src = Image.open(image)
        src = self.transforms(src).unsqueeze(0).cuda()
        
        ref = Image.open(ref_img)
        ref = self.transforms(ref).unsqueeze(0).cuda()

        mask = transforms.ToTensor()(Image.open(ref_mask)).unsqueeze(0).cuda()
        pose =  transforms.ToTensor()(np.load(ref_pose)).unsqueeze(0).cuda()


        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, pose, ref, mask], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src.shape).cuda()
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, pose, ref, mask], diffusion=self.diffusion)
            samples = xs[-1].cuda()


        samples = torch.clamp(samples, -1., 1.)

        output = (torch.cat([src, ref, mask*2-1, samples], -1) + 1.0)/2.0

        numpy_imgs = output.permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs = (255*numpy_imgs).astype(np.uint8)
        Image.fromarray(fake_imgs[0]).save('output.png')

    def show_pose(
        self,
        image,
        num_poses=1,
        sample_algorithm='ddim',
        nsteps=100,
        i=0
        ):
        """Run a single prediction on the model"""
        src = Image.open(image)
        src = self.transforms(src).unsqueeze(0).cuda()

        tgt_pose = torch.stack([transforms.ToTensor()(np.load(ps)).cuda() for ps in self.pose_list[:10]], 0)

        pose_grid = torch.cat([torch.zeros_like(src[0]),torch.cat([samps[:3] for samps in tgt_pose], -1)], -1)
        
        output = pose_grid #torch.cat([1-pose_grid], -2)

        numpy_imgs = output.unsqueeze(0).permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs = (255*numpy_imgs).astype(np.uint8)
        Image.fromarray(fake_imgs[0]).save('pose_imgs/pose_output_{}.png'.format(i))

# ref_img = "data/deepfashion_256x256/target_edits/reference_img_0.png"
# ref_mask = "data/deepfashion_256x256/target_mask/lower/reference_mask_0.png"
# ref_pose = "data/deepfashion_256x256/target_pose/reference_pose_0.npy"

# obj = Predictor()

# #obj.predict_pose(image='test.jpg', num_poses=4, sample_algorithm = 'ddim',  nsteps = 50)

# #obj.predict_appearance(image='test.jpg', ref_img = ref_img, ref_mask = ref_mask, ref_pose = ref_pose, sample_algorithm = 'ddim',  nsteps = 50)
