
import sys
sys.path.append('/home/dingguanqi/codes/2d/w-plus-adapter/script')
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from train_sage import Ax
import w_plus_adapter
from script.utils_direction import *
import os.path as osp
from tqdm import tqdm
import torch.nn.functional as F
import argparse
from models.psp import pSp
from script.utils import load_file_from_url, align_face, color_parse_map, pil2tensor, tensor2pil
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='experiments/af_2024-06-01/checkpoint-39000/pytorch_model.bin')
    parser.add_argument('--wp_ckpt', type=str, default='ckpt-af/save-checkpoint-33950/wplus_adapter.pth')
    parser.add_argument('--e4e_path', type=str, default='home/dingguanqi/codes/2d/encoder4editing/e4e_ckpts/af_e4e.pt')
    parser.add_argument('--save_path', type=str, default='experiments/af_2024-06-01/inference')
    parser.add_argument('--alpha', type=float, default=1.5)
    parser.add_argument('--cate', type=str, default='basenji')
    parser.add_argument('--embeddings_path', type=str, default='test_data/af-Step3-BFR-e4e/')
    parser.add_argument('--input_path', type=str, default='/home/dingguanqi/codes/2d/data/animal_faces/test_name_size_30/basenji/n02110806_298.JPEG_45_159_358_491.jpg')
    parser.add_argument('--class_embeddings_path', type=str, default='test_data/af-class_embeddings.pt')

    return parser.parse_args()

args = parse_args()

'''
model parameter settings
'''
base_model_path = "runwayml/stable-diffusion-v1-5"
# base_model_path = 'dreamlike-art/dreamlike-anime-1.0' #animate model
# base_model_path = 'darkstorm2150/Protogen_x3.4_Official_Release'

vae_model_path = "stabilityai/sd-vae-ft-mse"
device = "cuda"
wp_ckpt = args.wp_ckpt
# wp_ckpt = 'experiments_stage1_2024-05-29/save-checkpoint-18450/wplus_adapter.pth'

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

# if not osp.exists('./pretrain_models/wplus_adapter_stage1.bin'):
#     download_url = 'https://github.com/csxmli2016/w-plus-adapter/releases/download/v1/wplus_adapter_stage1.bin'
#     load_file_from_url(url=download_url, model_dir='./pretrain_models/', progress=True, file_name=None)

wp_model = w_plus_adapter.WPlusAdapter(pipe, wp_ckpt, device)


A_length = 100
ax = Ax(A_length)
ckpt = torch.load(args.ckpt, map_location='cpu')
ax.load_state_dict(ckpt)
ax=ax.to(device=device)
ax.eval()

e4e_path = args.e4e_path
e4e_ckpt = torch.load(e4e_path, map_location='cpu')
latent_avg = e4e_ckpt['latent_avg'].to(device)
e4e_opts = e4e_ckpt['opts']
e4e_opts['checkpoint_path'] = e4e_path
e4e_opts['device'] = device
opts = argparse.Namespace(**e4e_opts)
e4e = pSp(opts).to(device)
e4e.eval()


'''
Parameter settings
'''
alpha=args.alpha
seed = 18
cate = args.cate
prompt = 'a photo of a ' + cate
embeddings_path=args.embeddings_path
# e4e_id_path = 'test_data/af-Step3-BFR-e4e/Afghan_hound-n02088094_1035.JPEG_55_11_213_171_HAT_GAN_Real_SRx4.pth'
class_embeddings = torch.load(args.class_embeddings_path, map_location=torch.device('cpu'))
input_path = args.input_path
output_path = os.path.join(args.save_path, cate)
os.makedirs(output_path, exist_ok=True)

def calc_distribution(ax, cates):
    
    embed_list = []
    
    for embed in os.listdir(embeddings_path):
        cate = embed.split('-')[0]
        if cate in cates:
            class_embed = class_embeddings[cate].to(device)
            origin_embed = torch.load(osp.join(embeddings_path, embed), map_location='cpu').to(device=device) # for w+ B*18*512
            dw = origin_embed[:, :6] - class_embed[:, :6] 
            with torch.no_grad():
                dw_pred, A, x_pred = ax(dw)
                embed_list.append(torch.stack(x_pred))
    codes=torch.stack(embed_list).squeeze(2).squeeze(2).permute(1,0,2).cpu().numpy()
    mean=np.mean(codes,axis=1)
    covs=[]
    for i in range(codes.shape[0]):
        covs.append(np.cov(codes[i].T))

    return mean, covs

def sampler(A, means, covs, beta=0.8):
    one = torch.ones_like(torch.from_numpy(means[0]))
    zero = torch.zeros_like(torch.from_numpy(means[0]))
    dws=[]
    groups=[[0,1,2],[3,4,5]]
    for i in range(means.shape[0]):
        x=torch.from_numpy(np.random.multivariate_normal(mean=means[i], cov=covs[i], size=1)).float().to(device)
        mask = torch.where(torch.from_numpy(np.abs(means[i]))>beta, one, zero).to(device)
        x=x*mask
        for g in groups[i]:
            dw=torch.matmul(A[g], x.transpose(0,1)).squeeze(-1)
            dws.append(dw)
    dws=torch.stack(dws)
    return dws


def torchOrth(A, r=10):
    u,s,v = torch.svd(A)
    return v.T[:r]

def get_similar_cate(class_embeddings, ce, k=30):
    keys=list(class_embeddings.keys())
    distances={}
    for key in keys:
        distances[key]=torch.sum(F.pairwise_distance(ce, class_embeddings[key].cuda(), p=2))
    cates=sorted(distances.items(), key=lambda x: x[1])[:k]
    cates=[i[0] for i in cates] 
    return cates

def get_local_distribution(ax, latents, cr_directions, class_embeddings):
    ce = get_ce(latents.squeeze(0), cr_directions)
    cates = get_similar_cate(class_embeddings, ce)
    return calc_distribution(ax, cates)

def get_crdirections(class_embeddings, r=30):
    class_embeddings=torch.stack(list(class_embeddings.values()))
    class_embeddings=class_embeddings.squeeze(1).permute(1,0,2).cuda()
    cr_directions=[]
    for i in range(class_embeddings.shape[0]):
        cr_directions.append(torchOrth(class_embeddings[i], r))
    cr_directions=torch.stack(cr_directions)
    return cr_directions

def get_ce(latents, cr_directions):
    ce=[]
    for i in range(latents.shape[0]):
        cr_code=torch.zeros_like(latents[0])
        for j in range(cr_directions.shape[1]):
            cr_code=cr_code+torch.dot(latents[i],cr_directions[i][j])*cr_directions[i][j]
        ce.append(cr_code)
    ce=torch.stack(ce)
    return ce




residual_att_scale = 0.8
use_freeu = True
cr_directions = get_crdirections(class_embeddings, r=30)
cr_dictionary = get_crdirections(class_embeddings, r=-1)
image = Image.open(input_path).convert('RGB')
image = pil2tensor(image)
image = (image - 127.5) / 127.5     # Normalize

image = image.unsqueeze(0).to(device)

with torch.no_grad():
    latents_psp = e4e.encoder(image)
if latents_psp.ndim == 2:
    img_embed = latents_psp + latent_avg.repeat(latents_psp.shape[0], 1, 1)[:, 0, :]
else:
    img_embed = latents_psp + latent_avg.repeat(latents_psp.shape[0], 1, 1)

means, covs = get_local_distribution(ax, img_embed, cr_directions, class_embeddings)
dw = sampler(ax.A, means, covs, beta=0.0)

ce = get_ce(img_embed.squeeze(0), cr_dictionary).unsqueeze(0)
# ce = img_embed

img_embed = torch.cat(((alpha*dw.unsqueeze(0)+ ce[:, :6]), ce[:, 6:]), dim=1)

images = wp_model.generate_idnoise(prompt=prompt, w=img_embed.repeat(1, 1, 1).to(device, torch.float16), scale=residual_att_scale, num_samples=1, num_inference_steps=50, seed=seed, use_freeu=use_freeu, negative_prompt=None)

images[0].save(os.path.join(output_path, cate+'-generated.png'))

