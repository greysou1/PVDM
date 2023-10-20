#@title Pose Control using PIDM
# source env/bin/activate

print("loading packages ...")
import os
import cv2
from PIL import Image
from predict import Predictor
# from tqdm import tqdm

obj = Predictor()

# first_frame = cv2.imread('/home/prudvik/PIDM/02_4_full.jpg')
# first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

f = cv2.VideoCapture('/home/prudvik/PIDM/fashionvideo_part2.mp4')
 
ret, first_frame = f.read()
first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

# Convert the frame to a PIL Image object
pil_image = Image.fromarray(first_frame)

print("start")

npy_dir = '/home/prudvik/PIDM/npy_files/fashionvideo_part2/'
npy_list = os.listdir(npy_dir)

npy_list = sorted(npy_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

for i, npy_file in enumerate(npy_list):
    npy_file = npy_dir+npy_file
    obj.predict_pose_from_npy(image=pil_image, 
                                npy_file_path=npy_file, 
                                sample_algorithm='ddim', 
                                num_poses=4, 
                                nsteps=100, 
                                save_dir="outputs/fashionvideo_part2/",
                                output_filename='generated_frame_'+str(i)) 
