from Overlapping_patch_generator_test import *
from load_SR3 import *
import torch 
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

image_path = 'Histopathology\Dataset\Training_data_40X_100X\\001-40x.tif'
image = Image.open(image_path)
patches=[]
patch_size = image.size[0]//16
overlap_ratio = 0.0

batch_size = 1
LR_size = 32
img_size = 32
device="cuda"

patches= divide_image_into_patches(image,patch_size,overlap_ratio)
lr_patches=[]
hr_patches=[]

transforms_ = transforms.Compose([
                                transforms.Resize((32,32)),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
for _, img in enumerate(patches):
    img_16 = transforms_(img)
    img_16=img_16.unsqueeze(0)
    img_16.fill_(batch_size)
    lr_patches.append(img_16)
# Model Instance
schedule_opt = {'schedule':'linear', 'n_timestep':2000, 'linear_start':1e-4, 'linear_end':0.05}
sr3_model=SR3(device="cuda", img_size=img_size, LR_size=LR_size, loss_type='l1', 
                dataloader=None, testloader=None, schedule_opt=schedule_opt, 
                save_path=None, load_path="SR3_16_32.pt", load=True, inner_channel=96, 
                norm_groups=16, channel_mults=(1, 2, 2, 2), dropout=0.2, res_blocks=4, lr=1e-5, distributed=True)

# for img in lr_patches:
#     # img=transforms.ToTensor()(img)
#     img=img.unsqueeze(0)
#     img.fill_(1)

x=0
print("Model Implementation in process:")

for img in lr_patches:
    # print(type(img))
    # print(img.shape) # Dim mismatch , should be [1,3,32,16]
    hr=sr3_model.test(img)
    x+=1    
    hr=hr.detach().cpu()
    print(x,hr.shape)
    hr_patches.append(hr)
    
hr_image= blend_patches(hr_patches,image_size=image.size[0]*2.5,patch_size=patch_size,overlap_ratio=0)
hr_image.show()



    

    
## In my 4GB GPU it takes almost 128s for model implementation , and just 70s in lab's 8GB GPU !