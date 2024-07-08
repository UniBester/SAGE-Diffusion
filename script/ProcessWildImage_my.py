
import os
import os.path as osp
import numpy as np

from PIL import Image
import scipy
import scipy.ndimage
import argparse
from utils import load_file_from_url, align_face, color_parse_map, pil2tensor, tensor2pil

import torchvision.transforms as transforms

from models.parsenet import ParseNet
import torch
import cv2
from tqdm import tqdm


try:
    import dlib
except ImportError:
    print('Please install dlib by running:' 'conda install -c conda-forge dlib')
    


'''
For in the wild face image: CUDA_VISIBLE_DEVICES=0 python ./script/ProcessWildImage.py -i ./test_data/in_the_wild -o ./test_data/in_the_wild_Result -n -s
For aligned face image: CUDA_VISIBLE_DEVICES=0 python ./script/ProcessWildImage.py -i ./test_data/aligned_face -o ./test_data/aligned_face_Result -s
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', type=str, default='/home/dingguanqi/codes/2d/HAT/results/HAT_GAN_Real_SRx4/visualization/flowers')
    parser.add_argument('-o', '--out_dir', type=str, default='test_data/fl')
    parser.add_argument('-m', '--min_size', type=int, default=160)
    parser.add_argument('-n', '--need_alignment', action='store_true', help='input face image needs alignment like FFHQ')
    parser.add_argument('-c', '--cnn_detector', action='store_true', help='do not use cnn face detector in dlib.')
    parser.add_argument('-s', '--sr', action='store_true', help='using blind face restoration method')
    args = parser.parse_args()
    
    '''
    Step 1: Face Crop and Alignment
    '''
    # img_list = os.listdir(args.in_dir)
    # img_list.sort()
    # test_img_num = len(img_list)

    # if args.out_dir.endswith('/'):  # solve when path ends with /
    #     args.out_dir = args.out_dir[:-1]



    '''
    Step 3: Get w from Cropped Face
    '''
    from models.psp import pSp
    

    device = 'cuda'
    e4e_path = '/home/dingguanqi/codes/2d/encoder4editing/e4e_ckpts/fl_e4e.pt'

    e4e_ckpt = torch.load(e4e_path, map_location='cpu')
    latent_avg = e4e_ckpt['latent_avg'].to(device)
    e4e_opts = e4e_ckpt['opts']
    e4e_opts['checkpoint_path'] = e4e_path
    e4e_opts['device'] = device
    opts = argparse.Namespace(**e4e_opts)
    e4e = pSp(opts).to(device)
    e4e.eval()


    save_path_step3 = args.out_dir + '-Step3-BFR-e4e'
    save_path_step3_rec = args.out_dir + '-Step3-BFR-e4e-rec'

    os.makedirs(save_path_step3, exist_ok=True)
    os.makedirs(save_path_step3_rec, exist_ok=True)

    img_step2_list = os.listdir(args.in_dir)
    img_step2_list.sort()
    i=0
    for img_name in tqdm(img_step2_list):
        # print(f'[{i+1}/{len(img_step2_list)}] Processing: {img_name}')

        image = Image.open(os.path.join(args.in_dir, img_name)).convert('RGB')
        image = pil2tensor(image)
        image = (image - 127.5) / 127.5     # Normalize

        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            latents_psp = e4e.encoder(image)
        if latents_psp.ndim == 2:
            latents_psp = latents_psp + latent_avg.repeat(latents_psp.shape[0], 1, 1)[:, 0, :]
        else:
            latents_psp = latents_psp + latent_avg.repeat(latents_psp.shape[0], 1, 1)

        torch.save(latents_psp, osp.join(save_path_step3, img_name[:-4]+'.pth'))
        if i < 100:
            with torch.no_grad():
                imgs, _ = e4e.decoder([latents_psp[0].unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
            imgs_of = tensor2pil(imgs)
            imgs_of = imgs_of.resize((256,256))
            imgs_of.save(osp.join(save_path_step3_rec, img_name[:-4]+'.jpg'))
        i+=1

    print('#'*10 + 'Step 3: Get w Done! You can check the e4e reconstruction from {}'.format(save_path_step3_rec) + '#'*10)
