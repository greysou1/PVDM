#@title Pose Control using PIDM
# source env/bin/activate

print("loading packages ...")
from predict import Predictor
# from tqdm import tqdm
import cv2
import os
from PIL import Image

def get_frame_PIL(video_path, frame_idx=0):
    # Load the video file
    cap = cv2.VideoCapture(video_path)

    i = 0
    while i <= frame_idx:
        # Read the first frame of the video
        ret, frame = cap.read()
        if not ret: break
        i += 1

    # Convert the frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a PIL image
    return Image.fromarray(frame)

part1_video_dir = '/home/prudvik/PIDM/videos/fashionvideo_part1.mp4'
part2_npy_dir = '/home/prudvik/PIDM/npy_files/fashionvideo_part2/'
ckpt_path = '/home/prudvik/PIDM/pidm-train/checkpoints/pidm_deepfashion/model_100000.pt'
# ckpt_path = '/home/prudvik/PIDM/pidm-train/checkpoints/pidm_deepfashion/last.pt'
# ckpt_path = None

num_of_poses = 20
nsteps = 1000

save_dir = 'outputs/demo_tempo/'
output_filename = ckpt_path.split('/')[-1].split('.')[0] + '_batch_' +str(num_of_poses) + '_same_batch_noise'

print("====================================================\n\n")
print('part1_video_dir : ', part1_video_dir)
print('part2_npy_dir : ', part2_npy_dir)
print('ckpt_path : ', ckpt_path)
print('num_of_poses : ', num_of_poses)
print('save_dir : ', save_dir)
print('output_filename : ', output_filename)

for i in range(0, 150, num_of_poses):
    pil_image = get_frame_PIL(part1_video_dir, frame_idx=0)

    obj = Predictor(npy_dir=part2_npy_dir, ckpt_path=ckpt_path, start_frame_idx=i)

    obj.predict_pose_save_imgs(pil_image=pil_image,
                    sample_algorithm='ddim',
                    num_poses=num_of_poses,
                    nsteps=nsteps,
                    save_dir=save_dir,
                    output_filename=output_filename)


print("====================================================")

# checkpoints_dir = '/home/prudvik/PIDM/pidm-train/checkpoints/pidm_deepfashion/'
# checkpoints = sorted(os.listdir(checkpoints_dir))
# checkpoints = [checkpoints[i] for i in range(0, len(checkpoints), 25)]
# for checkpoint in checkpoints: