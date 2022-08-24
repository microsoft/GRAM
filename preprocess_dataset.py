import os
import numpy as np
from PIL import Image
import glob
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import argparse
import shutil


# 3D landmarks of a template cat head
cat_lm3D = np.array([
    [-4.893227, 0.255504, 3.936153],
    [4.893227, 0.255504, 3.936153],
    [0.000000, -5.859148, 8.948051],
    [-11.579516, 3.353250, -6.676847],
    [-12.895623, 15.929962, -4.881758],
    [-5.203006, 9.292290, -3.132928],
    [5.203006, 9.292290, -3.132928],
    [12.895623, 15.929962, -4.881758],
    [11.579516, 3.353250, -6.676847],
])

# 3D landmarks of a template human face
face_lm3D = np.array([
    [-0.31148657,  0.09036078,  0.13377953],
    [ 0.30979887,  0.08972035,  0.13179526],
    [ 0.0032535 , -0.24617933,  0.55244243],
    [-0.25216928, -0.5813392 ,  0.22405732],
    [ 0.2484662 , -0.5812824 ,  0.22235769],
])


# calculating least squres problem between 3D landmarks and 2D landmarks for image alignment
def POS(xp,x,cate=None):
    npts = xp.shape[0]

    A = np.zeros([2*npts,8])
    A[0:2*npts-1:2,0:3] = x
    A[0:2*npts-1:2,3] = 1
    A[1:2*npts:2,4:7] = x
    A[1:2*npts:2,7] = 1
    b = np.reshape(xp,[2*npts,1])

    if cate=='cats':
        weight = np.array([[4]] * 4 + [[2]] * 2 + [[0.5]] * 2 + [[0.2]] * 2 + [[0.2]] * 2 + [[0.2]] * 2 + [[0.2]] * 2 + [[0.5]] * 2) # set different importances for different landmarks
    else:
        weight = 1

    A = A * weight
    b = b * weight

    k,_,_,_ = np.linalg.lstsq(A,b)

    R1 = k[0:3].squeeze()
    R2 = k[4:7].squeeze()
    sTx = k[3]
    sTy = k[7]

    cz = np.cross(R1, R2)
    y = np.array([0, 1, 0])
    cx = np.cross(y, cz)
    cy = np.cross(cz, cx)
    cx = cx / np.linalg.norm(cx)
    cy = cy / np.linalg.norm(cy)
    cz = cz / np.linalg.norm(cz)

    yaw = np.arctan2(-cz[0], cz[2]) + 0.5 * np.pi
    pitch = np.arctan(-cz[1] / np.linalg.norm(cz[::2])) + 0.5 * np.pi
    roll1 = (np.sign(np.dot(cz, np.cross(cx, R1))) * np.arccos(np.dot(R1, cx) / np.linalg.norm(R1)) + np.sign(np.dot(cz, np.cross(cy, R2))) * np.arccos(np.dot(R2, cy) / np.linalg.norm(R2))) / 2
    roll2 = np.arctan2(-xp[1, 1] + xp[0, 1], xp[1, 0] - xp[0, 0])
    roll = roll2 + np.sign(roll1 - roll2) * np.log(np.abs(roll1 - roll2)/np.pi*180)*np.pi/180

    if cate=='cats':
        scale = 0.75 * np.linalg.norm(R1) + 0.25 * np.linalg.norm(R2) # for cats, we try to ensure the head scales along x-axis are similar for different subjects
    else:
        scale = 0.5 * np.linalg.norm(R1) + 0.5 * np.linalg.norm(R2)

    translate = np.stack([sTx, sTy],axis = 0)

    return yaw, pitch, roll, translate, scale


def align_img_ffhq(img,pos,target_size=256):
	_, _, _, translate, scale = pos
	w0,h0 = img.size
	scale = scale/target_size*224

	w = (w0/scale*95).astype(np.int32)
	h = (h0/scale*95).astype(np.int32)
	img = img.resize((w,h),resample = Image.LANCZOS)

	left = (w/2 - target_size/2 + float((translate[0] - w0/2)*95/scale)).astype(np.int32)
	right = left + target_size
	up = (h/2 - target_size/2 + float((h0/2 - translate[1])*95/scale)).astype(np.int32)
	below = up + target_size

	padding_len = max([abs(min(0,left)),abs(min(0,up)),max(right-w,0),max(below-h,0)])
	if padding_len > 0:
		img = np.array(img)
		img = np.pad(img,pad_width=((padding_len,padding_len),(padding_len,padding_len),(0,0)),mode='reflect')
		img = Image.fromarray(img)

	crop_img = img.crop((left+padding_len,up+padding_len,right+padding_len,below+padding_len))

	return crop_img


def align_img_cats(img, pos, target_size=256):
    _, _, roll, translate, scale = pos
    img = np.array(img)
    translate[1] = img.shape[0] - translate[1]
    cos_ = np.cos(roll)
    sin_ = np.sin(roll)
    rotate = np.array([[cos_, -sin_], [sin_, cos_]])
    crop = 15 * scale * np.array([[1, 1, -1, -1, 1], [1, -1, -1, 1, 1]])
    crop = rotate @ crop + translate.reshape((2, 1))
    padding = int(15 * scale)
    translate = translate + padding
    img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), 'constant')
    crop_img = Image.fromarray(img)

    # we eliminate roll angles for cat heads
    crop_img = crop_img\
        .rotate(roll/np.pi*180, resample=Image.BICUBIC, center=(translate[0], translate[1]))\
        .resize((target_size, target_size), Image.LANCZOS, box=(translate[0] - 15 * scale, translate[1] - 15 * scale, translate[0] + 15 * scale, translate[1] + 15 * scale))
    
    return crop_img

def preprocess_ffhq(img_path,lm_path,save_path,target_size=256,cate=None):
    img_name = img_path.split('/')[-1]

    shutil.copy(os.path.join(img_path.replace(img_name,''),'poses',img_name.replace('png','mat')),os.path.join(save_path,'poses',img_name.replace('png','mat')))
    
    img, lm = load_data_ffhq(img_path,lm_path)
    pos = POS(lm,face_lm3D,cate=cate)
    crop_img = align_img_ffhq(img,pos,target_size=target_size)
    crop_img.save(os.path.join(save_path,img_name))

def preprocess_cats(img_path,lm_path,save_path,target_size=256,cate=None):
    img_name = img_path.split('/')[-1]

    img, lm = load_data_cats(img_path,lm_path)
    pos = POS(lm,cat_lm3D,cate=cate)
    crop_img = align_img_cats(img,pos,target_size=target_size)
    crop_img.save(os.path.join(save_path,img_name.replace('jpg','png')))
    yaw, pitch, _, _, _ = pos
    np.save(os.path.join(save_path,'poses',img_name.replace('.jpg','_pose.npy')), np.array([float(pitch), float(yaw)])) 

def preprocess_carla(img_path,lm_path,save_path,target_size=128,cate=None):
    img_name = img_path.split('/')[-1]

    shutil.copy(os.path.join(img_path.replace(img_name,''),'poses',img_name.replace('.png','_extrinsics.npy')),os.path.join(save_path,'poses',img_name.replace('.png','_extrinsics.npy')))
    img = Image.open(img_path)
    img = img.resize((target_size,target_size),resample = Image.LANCZOS)
    img.save(os.path.join(save_path,img_name))

def load_data_ffhq(img_path,lm_path):
    img = Image.open(img_path)

    lm = np.loadtxt(lm_path)
    lm[:,1] = img.size[1] - 1 - lm[:,1]  #flip y-axis for detected landmarks

    return img, lm

def load_data_cats(img_path,lm_path):
    img = Image.open(img_path)

    with open(lm_path) as lm_file:
        lm = lm_file.read()
    lm = lm.split()[1:]
    lm = np.array([float(i) for i in lm])
    lm = lm.reshape((-1, 2))
    lm[:,1] = img.size[1] - lm[:,1]  #flip y-axis for provided landmarks

    return img, lm

def preprocess_data(raw_dataset_path, cate='ffhq'):

    if cate == 'ffhq':
        all_img_path = sorted(glob.glob(os.path.join(raw_dataset_path,'*.png')))
        all_lm_path = [os.path.join(raw_dataset_path,'lm5p',f.split('/')[-1].replace('png','txt')) for f in all_img_path]
        preprocess_func = preprocess_ffhq
    elif cate == 'cats':
        all_img_path = sorted(glob.glob(os.path.join(raw_dataset_path,'*.jpg')))
        all_lm_path = [f+'.cat' for f in all_img_path]
        preprocess_func = preprocess_cats
    elif cate == 'carla':
        all_img_path = sorted(glob.glob(os.path.join(raw_dataset_path,'*.png')))
        all_lm_path = all_img_path
        preprocess_func = preprocess_carla
    else:
        raise Exception("Invalid dataset type") 
    
    print('Number of images: %d'%len(all_img_path))
    
    save_path = os.path.join('datasets',cate)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path,'poses'), exist_ok=True)

    for img_path, lm_path in tqdm(zip(all_img_path,all_lm_path)):
        try:
            preprocess_func(img_path,lm_path,save_path,cate=cate) # skip a raw image if it does not have corresponding landmarks or poses
        except:
            print('skip invalid data...')
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dataset_path', type=str, default='./raw_data/ffhq', help='raw dataset path')
    parser.add_argument('--cate', type=str, default='ffhq', help='dataset type [ffhq | cats | carla]')
    opt = parser.parse_args()

    preprocess_data(opt.raw_dataset_path,cate=opt.cate)
