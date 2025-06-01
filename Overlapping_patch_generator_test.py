import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


#image_path = 'Histopathology\Dataset\Testing Data\LR\\001-100x.tif'

# Patch Divider
def divide_image_into_patches(image, patch_size, overlap_ratio):
    width, height = 256,256
    stride = int(patch_size * (1 - overlap_ratio))

    patches = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

    return patches

# Load the image using Pillow (replace 'path_to_your_image.jpg' with the actual path)

#image = Image.open(image_path)

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

# Load the image using Pillow (replace 'path_to_your_image.jpg' with the actual path)
#image = Image.open(image_path)


# Define patch size and overlap ratio
#patch_size = image.size[0]//10
#overlap_ratio = 0.0

# Divide the image into overlapping patches
#patches = divide_image_into_patches(image, patch_size, overlap_ratio)
# Display the image patches
# num_patches = len(patches)
# rows = (num_patches - 1) // 5 + 1
# for i, patch in enumerate(patches):
#     plt.subplot(rows, 5, i + 1)
#     plt.imshow(patch)
#     # plt.title(f'Patch {i+1}')
#     plt.axis('off')
# plt.show()



# Blend the overlapping patches to create a new image
#new_image = blend_patches(patches, image.size, patch_size, overlap_ratio)
# Display the new blended image
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import matplotlib.pyplot as plt


# image_path = 'Histopathology\Dataset\Testing Data\LR\\001-100x.tif'

# Patch Divider

def divide_image_into_patches(image, patch_size, overlap_ratio):
    width, height = image.size
    stride = int(patch_size * (1 - overlap_ratio))

    patches = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patch=transforms.ToTensor()(patch)
            patch=torch.unsqueeze(patch,0)
            patches.append(patch)
            # print(patch.shape)

    return patches

# Load the image using Pillow (replace 'path_to_your_image.jpg' with the actual path)

# image = Image.open(image_path)

# Patch Blender

def blend_patches(patches, image_size, patch_size, overlap_ratio):
    image_height, image_width = image_size

    # Unpack patch size
    patch_height = patch_width = patch_size

    # Calculate overlap pixels
    overlap_pixels_y = int(patch_height * overlap_ratio)
    overlap_pixels_x = int(patch_width * overlap_ratio)

    # Calculate number of patches along each dimension
    num_patches_y = int(np.ceil((image_height - overlap_pixels_y) / (patch_height - overlap_pixels_y)))
    num_patches_x = int(np.ceil((image_width - overlap_pixels_x) / (patch_width - overlap_pixels_x)))

    # Initialize a blank complete image
    complete_image = np.zeros((patches[0].shape[1], image_height, image_width))

    # Reconstruct complete image by blending patches
        for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate coordinates for the current patch
            start_y = i * (patch_height - overlap_pixels_y)
            end_y = min(start_y + patch_height, image_height)
            start_x = j * (patch_width - overlap_pixels_x)
            end_x = min(start_x + patch_width, image_width)

            # Extract the current patch
            patch = patches[i * num_patches_x + j]

            # Reshape the patch to match the dimensions of the complete image
            patch_reshaped = np.zeros((patch.shape[0], patch_height, patch_width))
            patch_reshaped[:, :min(patch.shape[1], patch_height), :min(patch.shape[2], patch_width)] = patch[:, :min(patch.shape[1], patch_height), :min(patch.shape[2], patch_width)]

            # Blend the patch into the complete image
            complete_image[:, start_y:end_y, start_x:end_x] += patch_reshaped

    return complete_image

# Load the image using Pillow (replace 'path_to_your_image.jpg' with the actual path)
image_path="Histopathology\\Dataset\\Training_data_40X_100X\\4.0.0.7.png"
image = Image.open(image_path)


# Define patch size and overlap ratio
patch_size = image.size[0]//10
overlap_ratio = 0.0

# Divide the image into overlapping patches
patches = divide_image_into_patches(image, patch_size, overlap_ratio)
# Display the image patches
num_patches = len(patches)
rows = (num_patches - 1) // 5 + 1
# for i, patch in enumerate(patches):
#     plt.subplot(rows, 5, i + 1)
#     plt.imshow(patch)
#     # plt.title(f'Patch {i+1}')
#     plt.axis('off')
# plt.show()



# # Blend the overlapping patches to create a new image
new_image = blend_patches(patches, image.size, patch_size, overlap_ratio)
# # Display the new blended image
# # new_image.show()
# plt.imshow(new_image)
# plt.axis('off')  # Turn off axis
# plt.show()