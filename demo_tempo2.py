#@title Pose Control using PIDM
# source env/bin/activate

print("loading packages ...")
from predict import Predictor
# from tqdm import tqdm
import cv2
import os
import random
from PIL import Image

video_numbers = [str(i) for i in range(10, 30)]
video_numbers = ["498"]

for video_number in video_numbers:
    # ------------------------------------------------------------------
    num_of_poses = 4 # num of frames
    nsteps = 1000    # num of diffusion timesteps

    dataset_path = "/home/prudvik/PIDM/dataset/fashionvideo/"
    first_frame_path = dataset_path + "train_frames/{}/frame0.jpg".format(video_number)
    npy_dir = dataset_path + "npy_frames/{}/".format(video_number)

    pil_image = Image.open(first_frame_path)
    video_length = len(os.listdir(npy_dir))

    start_frame_idx =  (video_length//2) + 0 #random.randint(video_length//2, video_length)
    end_frame_idx = -1
    step_frame_idx = 1

    ckpt_path = '/home/prudvik/PIDM/pidm-train/checkpoints/pidm_deepfashion/model_100000.pt'
    # ckpt_path = '/home/prudvik/PIDM/pidm-train/checkpoints/pidm_deepfashion/last.pt'

    # ------------------------------------------------------------------

    save_dir = "outputs/"+ckpt_path.split('/')[-1].split('.')[0]

    output_filename = f"video_{video_number}_{ckpt_path.split('/')[-1].split('.')[0]}_batch_{str(num_of_poses)}_start{start_frame_idx-int(video_length/2)}_step{step_frame_idx}"

    # output_filename = 'last_model_batch_4'

    print("====================================================\n")
    print('npy_dir : ', npy_dir)
    print('ckpt_path : ', ckpt_path)
    print('num_of_poses : ', num_of_poses)
    print('save_dir : ', save_dir)
    print('output_filename : ', output_filename)

    obj = Predictor(npy_dir=npy_dir, ckpt_path=ckpt_path, start_frame_idx=start_frame_idx, end_frame_idx=end_frame_idx, step_frame_idx=step_frame_idx)
    try:
        obj.predict_pose_save_imgs(pil_image=pil_image, 
                        sample_algorithm='ddim', 
                        num_poses=num_of_poses, 
                        nsteps=nsteps, 
                        save_dir=save_dir,
                        output_filename=output_filename)
    except:
        continue

    print("====================================================\n\n")

### Experiments ###
# _start0_step20
# _start20_step5

# 1. Generate 4 frame with step size 30
# 2. Generate first 4 consecutive frames
# 3. Generate random 4 consecutive frames

# and for 1 video each
# 1. 5 second full clip