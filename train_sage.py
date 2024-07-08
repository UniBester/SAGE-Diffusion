import os
import random
import argparse
from pathlib import Path
import itertools
import time

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from w_plus_adapter.w_plus_adapter import WProjModel
from w_plus_adapter.utils import is_torch2_available

if is_torch2_available():
    from w_plus_adapter.attention_processor import WPlusAttnProcessor2_0 as WPlusAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from w_plus_adapter.attention_processor import WPlusAttnProcessor, AttnProcessor

import os.path as osp
from diffusers import AutoencoderKL
from PIL import Image


# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file=None, tokenizer=None, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, imgs_path="", noise_path="", class_embedding_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate

        self.templates = [
            "{}",
            "a {}",
            "a photo of a {}",
            "a cropped photo of a {}",
            "the photo of a {}",
            "a depiction of a {}",
            "a close-up photo of a {}",
            "a good photo of a {}",
            "a photography of a {}",
            "a cropped photography of a {}",
            "a close-up photography of a {}",
            "a bright photography of a {}",
            "a good photography of a {}",
        ]
        self.noise = []
        self.image = []
        self.mask = []
        self.caption = []



        img_lists = os.listdir(imgs_path)
        
        for img_name in img_lists:
            self.image.append(osp.join(imgs_path, img_name))
            self.noise.append(osp.join(noise_path, img_name[:-4]+'.pth'))
        


        self.transform = transforms.Compose([
            transforms.Resize((self.size,self.size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.clip_image_processor = CLIPImageProcessor()
        self.data = self.image
        self.class_embedding = torch.load(class_embedding_path, map_location=torch.device('cpu'))
        
    def __getitem__(self, idx):

        ind = random.randint(0, len(self.image) - 1)
        image_file = self.image[ind]
        noise_file = self.noise[ind]


        
        # read noise
        noise = torch.load(noise_file, map_location=torch.device('cpu'))
        noise.requires_grad = False
        # read image
        raw_image = Image.open(image_file)
        image = self.transform(raw_image.convert("RGB"))
        
        select_template = random.choice(self.templates)
        word = image_file.split('/')[-1].split('-')[0]
        text = select_template.format(word)
        
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        class_embedding = self.class_embedding[word]

        _, id_name = osp.split(image_file)
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text": text,
            "id_noise": noise,
            'id_name':id_name[:-4],
            'class_embedding': class_embedding
        }

    def __len__(self):
        return len(self.image)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    idnoises = torch.stack([example["id_noise"] for example in data])
    idnames = [example["id_name"] for example in data]
    text = [example["text"] for example in data]
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    class_embeddings = torch.stack([example["class_embedding"] for example in data])

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "id_noises": idnoises,
        'id_names': idnames,
        'text': text,
        'class_embeddings': class_embeddings
    }



class WPlusAdapter(torch.nn.Module):
    """WPlus-Adapter"""
    def __init__(self, unet, w_proj_model, adapter_modules):
        super().__init__()
        self.unet = unet
        self.w_proj_model = w_proj_model
        self.adapter_modules = adapter_modules

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        wplus_tokens = self.w_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, wplus_tokens], dim=1)
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim):
        super(EqualLinear, self).__init__()
        self.out_dim=out_dim
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_dim*3, in_dim*2)
        self.fc2 = nn.Linear(in_dim*2, in_dim*2)
        self.fc3 = nn.Linear(in_dim*2, in_dim)
        self.fc4 = nn.Linear(in_dim, out_dim)
        self.fc5 = nn.Linear(out_dim, out_dim)
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, input):
        out = self.flat(input)
        out = self.fc1(out)
        out = self.nonlinearity(out)
        out = self.fc2(out)
        out = self.nonlinearity(out)
        out = self.fc3(out)
        out = self.nonlinearity(out)
        out = self.fc4(out)
        out = self.nonlinearity(out)
        out = self.fc5(out)
        return out


def sparse_loss(X, theta0=0.5, theta1=-1):
    bsz=X[0].shape[0]
    x0 = torch.sigmoid(theta0*X[0].abs()+theta1)
    x1 = torch.sigmoid(theta0*X[1].abs()+theta1)
    return x0.sum()/bsz+x1.sum()/bsz


def orthogonal_loss(A, B):
	dim = A.shape[-1]
	return torch.sum(torch.abs(torch.matmul(A.transpose(1,2), B)))/(dim*B.shape[-1]*6)

class Ax(nn.Module):
	def __init__(self, dim):
		super(Ax, self).__init__()
		self.A=nn.Parameter(torch.randn(6, 512, dim), requires_grad=True)
		self.encoder0=EqualLinear(512, dim)
		self.encoder1=EqualLinear(512, dim)
	def forward(self, dw):
		x0=self.encoder0(dw[:, :3])
		x0=x0.unsqueeze(-1).unsqueeze(1)
		x1=self.encoder1(dw[:, 3:6])
		x1=x1.unsqueeze(-1).unsqueeze(1)
		x=[x0.squeeze(-1),x1.squeeze(-1)]
		output_dw0=torch.matmul(self.A[:3], x0).squeeze(-1)
		output_dw1=torch.matmul(self.A[3:6], x1).squeeze(-1)
		output_dw=torch.cat((output_dw0,output_dw1),dim=1)
		return output_dw, self.A, x

    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='runwayml/stable-diffusion-v1-5',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--w_proj_path",
        type=str,
        default='/home/dingguanqi/codes/2d/w-plus-adapter/ckpt-fl/save-checkpoint-18450/wplus_adapter.pth',
        help="w_proj_path",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default=None,
        help="Training data root path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="tb_logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=600)
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--class_embedding_path",
        type=str,
        default='test_data/fl-class_embeddings.pt'
    )
    parser.add_argument(
        "--latent_path",
        type=str,
        default='test_data/fl-Step3-BFR-e4e'
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default='/home/dingguanqi/codes/2d/HAT/results/HAT_GAN_Real_SRx4/visualization/flowers'
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    


def tensor2image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)


def print_networks(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params / 1e6


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    

    #wplus-adapter
    w_proj_model = WProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        embeddings_dim=512,
        clip_extra_context_tokens=4,
    )
    w_proj_model.requires_grad_(False)

    if accelerator.is_main_process:
        accelerator.init_trackers("wplus-adapter", config=vars(args))
        print(['WProjModel contains: {} M parameters'.format(print_networks(w_proj_model))])

    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_wplus.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_wplus.weight": unet_sd[layer_name + ".to_v.weight"],
                "to_q_wplus.weight": unet_sd[layer_name + ".to_q.weight"],
            }
            attn_procs[name] = WPlusAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    wplus_adapter = WPlusAdapter(unet, w_proj_model, adapter_modules)
    pretrain_state_dict = torch.load(args.w_proj_path, map_location="cpu")
    wplus_adapter.w_proj_model.load_state_dict(pretrain_state_dict["w_proj"], strict=False)
    wplus_adapter.adapter_modules.load_state_dict(pretrain_state_dict["wplus_adapter"], strict=False)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    
    A_length = 100
    ax = Ax(A_length)
    
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    wplus_adapter.to(accelerator.device, dtype=weight_dtype)
    ax.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    params_to_opt = itertools.chain(ax.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, size=args.resolution, imgs_path=args.data_root_path, noise_path=args.latent_path, class_embedding_path=args.class_embedding_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    ax, optimizer, train_dataloader = accelerator.prepare(ax, optimizer, train_dataloader)

    '''
    you can continue the training by loading the model from the previous epoch
    '''
    # accelerator.load_state('./experiments_stage1_2023-12-01/checkpoint-10000')
    # or
    # learnable_parameter_path = './checkpoint-5000/wplus_adapter.bin'
    # pretrain_state_dict = torch.load(learnable_parameter_path, map_location="cpu")
    # accelerator.unwrap_model(wplus_adapter).w_proj_model.load_state_dict(pretrain_state_dict["image_proj"], strict=False)
    # accelerator.unwrap_model(wplus_adapter).adapter_modules.load_state_dict(pretrain_state_dict["ip_adapter"], strict=False)

    B = torch.stack(list(torch.load(args.class_embedding_path, map_location=accelerator.device).values())).squeeze(1).permute(1,2,0)[:6]

    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(wplus_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
  
                origin_embeds = batch['id_noises'][:,0,:,:].to(device=accelerator.device, dtype=weight_dtype) # for w+ B*18*512
                
                class_embeds = batch['class_embeddings'][:,0,:,:].to(accelerator.device, dtype=weight_dtype) # for w+ B*18*512

                dw = origin_embeds[:, :6] - class_embeds[:, :6] 

                dw_pred, A, x_pred = ax(dw)
                image_embeds = torch.cat((dw_pred + class_embeds[:, :6], origin_embeds[:, 6:]), dim=1).to(device=accelerator.device, dtype=weight_dtype)
   
            
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]
                
                noise_pred = wplus_adapter(noisy_latents, timesteps, encoder_hidden_states, image_embeds)
        
                loss_ldm = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                loss_orth = 0.0005*orthogonal_loss(A, B)

                loss_sparse = 0.005*sparse_loss(x_pred)

                loss_latent = F.mse_loss(image_embeds.float(), origin_embeds.float(), reduction="mean")

                loss = loss_ldm + loss_orth + loss_sparse + loss_latent
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, global step {}, data_time: {}, time: {}, step_loss: {}, lr: {}".format(
                        epoch, step, global_step, load_data_time, time.perf_counter() - begin, avg_loss, args.learning_rate))
            
            global_step += 1
            if accelerator.is_main_process:
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    accelerator.print('saving checkpoint {}'.format(global_step))
            
            logs = {"loss": loss.detach().item(), "lr": args.learning_rate, "loss_ldm": loss_ldm.detach().item(), "loss_orth": loss_orth.detach().item(), "loss_sparse": loss_sparse.detach().item(), "loss_latent": loss_latent.detach().item()}
            accelerator.log(logs, step=global_step)

            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    
