import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


image_path = 'Histopathology\Dataset\Training_data_40X_100X\\001-40x.tif'

# Patch Divider
def divide_image_into_patches(image, patch_size, overlap_ratio):
    width, height = image.size
    stride = int(patch_size * (1 - overlap_ratio))

    patches = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

    return patches


# Patch Blender

def blend_patches(patches, image_size, patch_size, overlap_ratio):
    canvas = Image.new("RGB", image_size)

    width, height = image_size
    stride = int(patch_size * (1 - overlap_ratio))

    patch_counts = [[1 for _ in range(width)] for _ in range(height)]

    for i, patch in enumerate(patches):
        x_start = (i % (width // stride)) * stride
        y_start = (i // (width // stride)) * stride

        for y in range(patch_size):
            for x in range(patch_size):
                r, g, b = patch.getpixel((x, y))
                r_prev, g_prev, b_prev = canvas.getpixel((x_start + x, y_start + y))
                r_avg = (r + r_prev) / patch_counts[y_start + y][x_start + x]
                g_avg = (g + g_prev) / patch_counts[y_start + y][x_start + x]
                b_avg = (b + b_prev) / patch_counts[y_start + y][x_start + x]
                canvas.putpixel((x_start + x, y_start + y), (int(r_avg), int(g_avg), int(b_avg)))
                patch_counts[y_start + y][x_start + x] += 1

    return canvas


###Testing Functions:

# Load the image using Pillow (replace 'path_to_your_image.jpg' with the actual path)
image = Image.open(image_path)


## Patch Divider working Successfully !


# Define patch size and overlap ratio
patch_size = image.size[0]//16
overlap_ratio = 0.0
# Divide the image into overlapping patches
patches = divide_image_into_patches(image, patch_size, overlap_ratio)
# Display the image patches
num_patches = len(patches)
rows = (num_patches - 1) // 5 + 1
print(patches[0].size)
# for i, patch in enumerate(patches):
#     plt.subplot(rows, 5, i + 1)
#     plt.imshow(patch)
#     # plt.title(f'Patch {i+1}')
#     plt.axis('off')
# plt.show()


## Blending Successful
# stitched_img=blend_patches(patches=patches,image_size=image.size,patch_size=16,overlap_ratio=0)
# plt.imshow(stitched_img)
# plt.show()


# Model Implementation
from SR3_pytorch1_copy import *

model=SR3(device="cuda", img_size=32, LR_size=32, loss_type='l1', 
                dataloader=train_dataloader, testloader=test_dataloader, schedule_opt=schedule_opt, 
                save_path=None, load_path="SR3_16_32.pt", load=True, inner_channel=96, 
                norm_groups=16, channel_mults=(1, 2, 2, 2), dropout=0.2, res_blocks=4, lr=1e-5, distributed=True)

checkpoint=torch.load("SR3_16_32.pt")
model.load_state_dict(checkpoint["model_state_dict"])
