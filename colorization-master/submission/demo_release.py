
import argparse
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cv2 import cv2
import os
import shutil
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()


colorizer_siggraph17 = siggraph17(pretrained=True).eval()

#colorizer_siggraph17 = siggraph17(pretrained=True).train()

#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(colorizer_siggraph17.parameters(), lr=0.001, momentum=0.9)

if(opt.use_gpu):
	colorizer_eccv16.cuda()
	colorizer_siggraph17.cuda()

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
if(opt.use_gpu):
	tens_l_rs = tens_l_rs.cuda()

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
"""
plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16)
plt.imsave('%s_siggraph17.png'%opt.save_prefix, out_img_siggraph17)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
#plt.imshow(img)
#plt.imsave('imgs_out/img.png', img)
plt.title('Original')
plt.axis('off')

plt.subplot(2,2,2)
#plt.imshow(img_bw)
#plt.imsave('imgs_out/img_bw.png', img_bw)
plt.title('Input')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(out_img_eccv16)
#plt.imsave('imgs_out/out_img_eecsv16.png', out_img_eccv16)
plt.title('Output (ECCV 16)')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(out_img_siggraph17)
#plt.imsave('imgs_out/out_img_siggraph17.png', out_img_siggraph17)
plt.title('Output (SIGGRAPH 17)')
plt.axis('off')
#plt.show()
"""

# PROCESS ww1 frames

ww1_processed = os.path.join(os.getcwd(), 'ww1_processed')


if os.path.exists(ww1_processed):
    shutil.rmtree(ww1_processed)

os.makedirs(ww1_processed)

vidcap = cv2.VideoCapture('ww1.mp4')
success, image = vidcap.read()
count = 0
success = True

while success:
	success, image = vidcap.read()

	filepath = os.path.join(ww1_processed, "frame" + str(count) + ".jpg")

	if success:
		gray_img = rgb2gray_approx(image)
		cv2.imwrite(filepath, gray_img)     # save frame as JPEG file

	count += 1

# Run eccv16 on ww1 frames

c_ecc_ww1 = os.path.join(os.getcwd(), 'c_ecc_ww1')

if os.path.exists(c_ecc_ww1):
    shutil.rmtree(c_ecc_ww1)

os.makedirs(c_ecc_ww1)

for image in os.listdir(ww1_processed):

	img_path = os.path.join(ww1_processed, image)

	# default size to process images is 256x256
	# grab L channel in both original ("orig") and resized ("rs") resolutions
	img = load_img(img_path)
	(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
	if(opt.use_gpu):
		tens_l_rs = tens_l_rs.cuda()

	# colorizer outputs 256x256 ab map
	# resize and concatenate to original L channel
	out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())

	filepath = os.path.join(c_ecc_ww1, "color_" + image)
	plt.imsave(filepath, out_img_eccv16)

# Create new video of eccv16 colorized ww1

os.system("ffmpeg -f image2 -r 25 -i ./c_ecc_ww1/color_frame%01d.jpg -vcodec mpeg4 -y c_ecc_ww1.mp4")

# Run siggraph17 on ww1 frames

c_sig_ww1 = os.path.join(os.getcwd(), 'c_sig_ww1')

if os.path.exists(c_sig_ww1):
    shutil.rmtree(c_sig_ww1)

os.makedirs(c_sig_ww1)

for image in os.listdir(ww1_processed):

	img_path = os.path.join(ww1_processed, image)

	# default size to process images is 256x256
	# grab L channel in both original ("orig") and resized ("rs") resolutions
	img = load_img(img_path)
	(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
	if(opt.use_gpu):
		tens_l_rs = tens_l_rs.cuda()

	# colorizer outputs 256x256 ab map
	# resize and concatenate to original L channel
	out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

	filepath = os.path.join(c_sig_ww1, "color_" + image)
	plt.imsave(filepath, out_img_siggraph17)

# Create new video of siggraph17 colorized ww1

os.system("ffmpeg -f image2 -r 25 -i ./c_sig_ww1/color_frame%01d.jpg -vcodec mpeg4 -y c_sig_ww1.mp4")



cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(cv2.CAP_V4L2)
#if not cap.isOpened():
	#raise IOError("Cannot open webcam")

#video = v4l2capture.Video_device("/dev/video0")

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

    # Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#gray_img = rgb2gray_approx(frame)

    # Display the resulting frame
	cv2.imshow('frame', gray)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



#####


# PROCESS garden frames

garden_processed = os.path.join(os.getcwd(), 'garden_processed')


if os.path.exists(garden_processed):
    shutil.rmtree(garden_processed)

os.makedirs(garden_processed)

vidcap = cv2.VideoCapture('garden.mp4')
success, image = vidcap.read()
count = 0
success = True

while success:
	success, image = vidcap.read()

	filepath = os.path.join(garden_processed, "frame" + str(count) + ".jpg")

	if success:
		gray_img = rgb2gray_approx(image)
		cv2.imwrite(filepath, gray_img)     # save frame as JPEG file

	count += 1

# Run eccv16 on garden frames

c_ecc_garden = os.path.join(os.getcwd(), 'c_ecc_garden')

if os.path.exists(c_ecc_garden):
    shutil.rmtree(c_ecc_garden)

os.makedirs(c_ecc_garden)

for image in os.listdir(garden_processed):

	img_path = os.path.join(garden_processed, image)

	# default size to process images is 256x256
	# grab L channel in both original ("orig") and resized ("rs") resolutions
	img = load_img(img_path)
	(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
	if(opt.use_gpu):
		tens_l_rs = tens_l_rs.cuda()

	# colorizer outputs 256x256 ab map
	# resize and concatenate to original L channel
	out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())

	filepath = os.path.join(c_ecc_garden, "color_" + image)
	plt.imsave(filepath, out_img_eccv16)

# Create new video of eccv16 colorized garden

os.system("ffmpeg -f image2 -r 25 -i ./c_ecc_garden/color_frame%01d.jpg -vcodec mpeg4 -y c_ecc_garden.mp4")

# Run siggraph17 on garden frames

c_sig_garden = os.path.join(os.getcwd(), 'c_sig_garden')

if os.path.exists(c_sig_garden):
    shutil.rmtree(c_sig_garden)

os.makedirs(c_sig_garden)

for image in os.listdir(garden_processed):

	img_path = os.path.join(garden_processed, image)

	# default size to process images is 256x256
	# grab L channel in both original ("orig") and resized ("rs") resolutions
	img = load_img(img_path)
	(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
	if(opt.use_gpu):
		tens_l_rs = tens_l_rs.cuda()

	# colorizer outputs 256x256 ab map
	# resize and concatenate to original L channel
	out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

	filepath = os.path.join(c_sig_garden, "color_" + image)
	plt.imsave(filepath, out_img_siggraph17)

# Create new video of siggraph17 colorized garden

os.system("ffmpeg -f image2 -r 25 -i ./c_sig_garden/color_frame%01d.jpg -vcodec mpeg4 -y c_sig_garden.mp4")

#####

# PROCESS dictator frames

dictator_processed = os.path.join(os.getcwd(), 'dictator_processed')


if os.path.exists(dictator_processed):
    shutil.rmtree(dictator_processed)

os.makedirs(dictator_processed)

vidcap = cv2.VideoCapture('dictator.mp4')
success, image = vidcap.read()
count = 0
success = True

while success:
	success, image = vidcap.read()

	filepath = os.path.join(dictator_processed, "frame" + str(count) + ".jpg")

	if success:
		gray_img = rgb2gray_approx(image)
		cv2.imwrite(filepath, gray_img)     # save frame as JPEG file

	count += 1

# Run eccv16 on dictator frames

c_ecc_dictator = os.path.join(os.getcwd(), 'c_ecc_dictator')

if os.path.exists(c_ecc_dictator):
    shutil.rmtree(c_ecc_dictator)

os.makedirs(c_ecc_dictator)

for image in os.listdir(dictator_processed):

	img_path = os.path.join(dictator_processed, image)

	# default size to process images is 256x256
	# grab L channel in both original ("orig") and resized ("rs") resolutions
	img = load_img(img_path)
	(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
	if(opt.use_gpu):
		tens_l_rs = tens_l_rs.cuda()

	# colorizer outputs 256x256 ab map
	# resize and concatenate to original L channel
	out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())

	filepath = os.path.join(c_ecc_dictator, "color_" + image)
	plt.imsave(filepath, out_img_eccv16)

# Create new video of eccv16 colorized dictator

os.system("ffmpeg -f image2 -r 25 -i ./c_ecc_dictator/color_frame%01d.jpg -vcodec mpeg4 -y c_ecc_dictator.mp4")

# Run siggraph17 on dictator frames

c_sig_dictator = os.path.join(os.getcwd(), 'c_sig_dictator')

if os.path.exists(c_sig_dictator):
    shutil.rmtree(c_sig_dictator)

os.makedirs(c_sig_dictator)

for image in os.listdir(dictator_processed):

	img_path = os.path.join(dictator_processed, image)

	# default size to process images is 256x256
	# grab L channel in both original ("orig") and resized ("rs") resolutions
	img = load_img(img_path)
	(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
	if(opt.use_gpu):
		tens_l_rs = tens_l_rs.cuda()

	# colorizer outputs 256x256 ab map
	# resize and concatenate to original L channel
	out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

	filepath = os.path.join(c_sig_dictator, "color_" + image)
	plt.imsave(filepath, out_img_siggraph17)

# Create new video of siggraph17 colorized dictator

os.system("ffmpeg -f image2 -r 25 -i ./c_sig_dictator/color_frame%01d.jpg -vcodec mpeg4 -y c_sig_dictator.mp4")


# Pre-process evaluation images

evaluation = os.path.join(os.getcwd(), 'evaluation')

evaluation_bw = os.path.join(os.getcwd(), 'evaluation_bw')



if os.path.exists(evaluation_bw):
    shutil.rmtree(evaluation_bw)

os.makedirs(evaluation_bw)

for image in os.listdir(evaluation):

	img_path = os.path.join(evaluation, image)
	img = load_img(img_path)

	filepath = os.path.join(evaluation_bw, image)

	gray_img = rgb2gray_approx(img)
	cv2.imwrite(filepath, gray_img)     # save frame as JPEG file


# Evaluate siggraph17

sig_psnr_score_total = 0.0
sig_ssim_score_total = 0.0
count = 0

for image in os.listdir(evaluation):

	img_path = os.path.join(evaluation, image)

	bw_img_path = os.path.join(evaluation_bw, image)

	# default size to process images is 256x256
	# grab L channel in both original ("orig") and resized ("rs") resolutions
	img = load_img(img_path)

	img_bw = load_img(bw_img_path)

	(tens_l_orig, tens_l_rs) = preprocess_img(img_bw, HW=(256,256))
	if(opt.use_gpu):
		tens_l_rs = tens_l_rs.cuda()


	# colorizer outputs 256x256 ab map
	# resize and concatenate to original L channel
	out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

	psnr_score = psnr(img, out_img_siggraph17)
	ssim_score = ssim(img, out_img_siggraph17, multichannel=True)

	sig_psnr_score_total += psnr_score
	sig_ssim_score_total += ssim_score
	count += 1

sig_psnr_avg = sig_psnr_score_total/count
sig_ssim_avg = sig_ssim_score_total/count

print("siggraph17 evaluation:")
print("psnr_avg: " + str(sig_psnr_avg))
print("ssim_avg: " + str(sig_ssim_avg))


# Evaluate eccv16

eccv_psnr_score_total = 0.0
eccv_ssim_score_total = 0.0
count = 0

for image in os.listdir(evaluation):

	img_path = os.path.join(evaluation, image)

	bw_img_path = os.path.join(evaluation_bw, image)

	# default size to process images is 256x256
	# grab L channel in both original ("orig") and resized ("rs") resolutions
	img = load_img(img_path)

	img_bw = load_img(bw_img_path)

	(tens_l_orig, tens_l_rs) = preprocess_img(img_bw, HW=(256,256))
	if(opt.use_gpu):
		tens_l_rs = tens_l_rs.cuda()


	# colorizer outputs 256x256 ab map
	# resize and concatenate to original L channel
	out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())

	psnr_score = psnr(img, out_img_eccv16)
	ssim_score = ssim(img, out_img_eccv16, multichannel=True)

	eccv_psnr_score_total += psnr_score
	eccv_ssim_score_total += ssim_score
	count += 1

eccv_psnr_avg = eccv_psnr_score_total/count
eccv_ssim_avg = eccv_ssim_score_total/count

print("eccv16 evaluation:")
print("psnr_avg: " + str(eccv_psnr_avg))
print("ssim_avg: " + str(eccv_ssim_avg))