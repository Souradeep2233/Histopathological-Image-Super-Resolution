import torch, torchvision
from torch import nn
from torch.nn import init
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from einops import rearrange, repeat
from tqdm.notebook import tqdm
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math, os, copy
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

from unet import UNetModel, SuperResModel, EncoderUNetModel
# Encoding noise intensity at per with dimension


class Diffusion(nn.Module):
    def __init__(self, model, device, img_size, LR_size, channels=3):
        super().__init__()
        self.channels = channels
        self.model = model.to(device)
        self.img_size = img_size
        self.LR_size = LR_size
        self.device = device

    def set_loss(self, loss_type):
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError()

    def make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
        if schedule == 'linear':
            betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        elif schedule == 'warmup':
            warmup_frac=0.1
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
            warmup_time = int(n_timestep * warmup_frac)
            betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        elif schedule == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(schedule)
        return betas

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = self.make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        self.num_timesteps = int(len(betas))
        # Coefficient for forward diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('pred_coef1', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('pred_coef2', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # Coefficient for reverse diffusion posterior q(x_{t-1} | x_t, x_0)
        variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('variance', to_torch(variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
    def predict_start(self, x_t, t, noise):
        return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise

    # Compute mean and log variance of posterior(reverse diffusion process) distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    # Note that posterior q for reverse diffusion process is conditioned Gaussian distribution q(x_{t-1}|x_t, x_0)
    # Thus to compute desired posterior q, we need original image x_0 in ideal, 
    # but it's impossible for actual training procedure -> Thus we reconstruct desired x_0 and use this for posterior
    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        x_recon = self.predict_start(x, t, noise=self.model(torch.cat([condition_x, x], dim=1), noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return mean, posterior_log_variance

    # Progress single step of reverse diffusion process
    # Given mean and log variance of posterior, sample reverse diffusion result from the posterior
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        mean, log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return mean + noise * (0.5 * log_variance).exp()

    # Progress whole reverse diffusion process
    @torch.no_grad()
    def super_resolution(self, x_in):
        img = torch.rand_like(x_in, device=x_in.device)
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, i, condition_x=x_in)
        return img

    # Compute loss to train the model
    def p_losses(self, x_in):
        x_start = x_in
        lr_imgs = transforms.Resize(self.LR_size)(x_in)
        b, c, h, w = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        sqrt_alpha = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t-1], self.sqrt_alphas_cumprod_prev[t], size=b)).to(x_start.device)
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1, 1)

        noise = torch.randn_like(x_start).to(x_start.device)
        # Perturbed image obtained by forward diffusion process at random time step t
        x_noisy = sqrt_alpha * x_start + (1 - sqrt_alpha**2).sqrt() * noise
        # The model predict actual noise added at time step t
        pred_noise = self.model(torch.cat([lr_imgs, x_noisy], dim=1), noise_level=sqrt_alpha)
        
        return self.loss_func(noise, pred_noise)

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)


# Class to train & test desired model
class SR3():
    def __init__(self, device, img_size, LR_size, loss_type, dataloader, testloader, 
                    schedule_opt, save_path, load_path=None, load=False, 
                    in_channel=6, out_channel=3, inner_channel=32, norm_groups=8, 
                    channel_mults=(1, 2, 4, 8, 8), res_blocks=3, dropout=0, lr=1e-5, distributed=False):
        super(SR3, self).__init__()
        self.dataloader = dataloader
        self.testloader = testloader
        self.device = device
        self.save_path = save_path
        self.img_size = img_size
        self.LR_size = LR_size

        model = UNet(in_channel, out_channel, inner_channel, norm_groups, channel_mults, res_blocks, dropout, img_size)
        self.sr3 = Diffusion(model, device, img_size, LR_size, out_channel)

        # Apply weight initialization & set loss & set noise schedule
        self.sr3.apply(self.weights_init_orthogonal)
        self.sr3.set_loss(loss_type)
        self.sr3.set_new_noise_schedule(schedule_opt)

        if distributed:
            assert torch.cuda.is_available()
            self.sr3 = nn.DataParallel(self.sr3)

        self.optimizer = torch.optim.Adam(self.sr3.parameters(), lr=lr)

        params = sum(p.numel() for p in self.sr3.parameters())
        print(f"Number of model parameters : {params}")

        # if load:
        #     self.load(load_path)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    def train(self, epoch, verbose):
        fixed_imgs = copy.deepcopy(next(iter(self.testloader)))
        fixed_imgs = fixed_imgs[0].to(self.device)
        # Transform to low-resolution images
        print("test:"+f"{fixed_imgs.shape}")
        fixed_imgs = transforms.Resize(self.img_size)(fixed_imgs)

        for i in tqdm(range(epoch)):
            train_loss = 0
            for _, imgs in enumerate(self.dataloader):
                # Initial imgs are high-resolution
                imgs = imgs[0].to(self.device)
                b, c, h, w = imgs.shape

                self.optimizer.zero_grad()
                loss = self.sr3(imgs)
                loss = loss.sum() / int(b*c*h*w)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * b

            if (i+1) % verbose == 0:
                self.sr3.eval()
                test_imgs = next(iter(self.testloader))
                test_imgs = test_imgs[0].to(self.device)
                b, c, h, w = test_imgs.shape

                with torch.no_grad():
                    val_loss = self.sr3(test_imgs)
                    val_loss = val_loss.sum() / int(b*c*h*w)
                self.sr3.train()

                train_loss = train_loss / len(self.dataloader)
                print(f'Epoch: {i+1} / loss:{train_loss:.3f} / val_loss:{val_loss.item():.3f}')

                
                # Save example of test images to check training
                plt.figure(figsize=(15,10))
                
                plt.subplot(1,2,1)
                plt.axis("off")
                plt.title("Low-Resolution Inputs")
                print("fixed_imgs:"+f"{test_imgs.shape}")
                grid_img = torchvision.utils.make_grid(test_imgs, nrow=2, padding=1, normalize=True).cpu().numpy()
                
                transposed_grid_img=np.transpose(grid_img, (1, 2, 0))
                print(transposed_grid_img.shape)
                plt.imshow(transposed_grid_img)
                plt.show()
                
                plt.subplot(1,2,2)
                plt.axis("off")
                plt.title("Super-Resolution Results")
                
                plt.imshow(np.transpose(torchvision.utils.make_grid(self.test(test_imgs).detach().cpu(), 
                                                                    nrow=2, padding=1, normalize=True),(1,2,0)))
                plt.show()
                plt.savefig('SuperResolution_Result.jpg')
                plt.close()

                # Save model weight
                self.save(self.save_path)

    def test(self, imgs):
        imgs_lr = imgs
        print("testlr:"+f"{imgs_lr.shape}")
        self.sr3.eval()
        with torch.no_grad():
            if isinstance(self.sr3, nn.DataParallel):
                result_SR = self.sr3.module.super_resolution(imgs_lr)
            else:
                result_SR = self.sr3.super_resolution(imgs_lr)
                print("testSR:"+f"{result_SR.shape}")
        self.sr3.train()
        return result_SR

    def save(self, save_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path))
        print("Model loaded successfully")


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_folder = "LR"
        self.low_resolution_path = os.path.join(self.root_dir, self.image_folder)
        self.high_resolution_path = os.path.join(self.root_dir, "HR")
        self.image_filenames_LR = os.listdir(self.low_resolution_path)
        self.image_filenames_HR = os.listdir(self.high_resolution_path)

    def __len__(self):
        return len(self.image_filenames_LR)

    def __getitem__(self, idx):
        image_name_LR = self.image_filenames_LR[idx]
        image_name_HR = self.image_filenames_HR[idx]

        low_res_path = os.path.join(self.low_resolution_path, image_name_LR)
        high_res_path = os.path.join(self.high_resolution_path, image_name_HR)

        low_res_image = Image.open(low_res_path).convert("RGB")
        high_res_image = Image.open(high_res_path).convert("RGB")

        if self.transform:
            low_res_image = self.transform(low_res_image)
            # print(low_res_image)
            high_res_image = self.transform(high_res_image)

        return low_res_image, high_res_image

if __name__ == "__main__":
    batch_size = 1
    LR_size = 96
    img_size = 384
    # root = 'Histopathology\Dataset\Training Data'
    root = 'Histopathology\SR_GAN_Prac\Data'
    # testroot = './data/celeba_hq'

    transforms_ = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    dataloader=CustomImageDataset(root_dir=root,transform=transforms_)
    
    
    
    train_dataloader = DataLoader(dataloader,batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(dataloader,batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # testloader = DataLoader(torchvision.datasets.ImageFolder(testroot, transform=transforms_), 
    #                         batch_size=4, shuffle=True, num_workers=8, pin_memory=True)

    # Save train data example
    # imgs, _ = next(iter(train_dataloader))
    # LR_imgs = transforms.Resize(img_size)(transforms.Resize(LR_size)(imgs))
    # plt.figure(figsize=(15,10))
    # plt.subplot(1,2,1)
    # plt.axis("off")
    # plt.title("Low-Resolution Images")
    # plt.imshow(np.transpose(torchvision.utils.make_grid(LR_imgs[:4], padding=1, normalize=True).cpu(),(1,2,0)))

    # plt.subplot(1,2,2)
    # plt.axis("off")
    # plt.title("High-Resolution Images")
    # plt.imshow(np.transpose(torchvision.utils.make_grid(imgs[:4], padding=1, normalize=True).cpu(),(1,2,0)))
    # plt.savefig('Train_Examples.jpg')
    # plt.close()
    # print("Example train images were saved")

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(device)
    schedule_opt = {'schedule':'linear', 'n_timestep':20, 'linear_start':1e-4, 'linear_end':0.05}

    sr3 = SR3(device, img_size=img_size, LR_size=LR_size, loss_type='l1', 
                dataloader=train_dataloader, testloader=test_dataloader, schedule_opt=schedule_opt, 
                save_path='./SR3.pt', load_path=None, load=True, inner_channel=96, 
                norm_groups=16, channel_mults=(1, 2, 2, 2), dropout=0.2, res_blocks=2, lr=1e-5, distributed=False)
    sr3.train(epoch=1, verbose=1)
#Time step, batch size, epoch ,verbose and dataset has been modified !!
# Configure the metrics ,  and tune the hyper paramter appropriately  !!!!