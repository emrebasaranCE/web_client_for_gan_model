import os
import argparse
import json
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from models import CompletionNetwork
from utils import poisson_blend


parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('config')
parser.add_argument('input_img')
parser.add_argument('tensor_img')  # now used for loading mask
parser.add_argument('output_img')
parser.add_argument('--max_holes', type=int, default=5)  # unused now
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--hole_min_w', type=int, default=24)  # unused now
parser.add_argument('--hole_max_w', type=int, default=48)  # unused now
parser.add_argument('--hole_min_h', type=int, default=24)  # unused now
parser.add_argument('--hole_max_h', type=int, default=48)  # unused now


def main(args):

    args.model = os.path.expanduser(args.model)
    args.config = os.path.expanduser(args.config)
    args.input_img = os.path.expanduser(args.input_img)
    args.tensor_img = os.path.expanduser(args.tensor_img)
    args.output_img = os.path.expanduser(args.output_img)

    # =============================================
    # Load model
    # =============================================
    with open(args.config, 'r') as f:
        config = json.load(f)
    mpv = torch.tensor(config['mpv']).view(1, 3, 1, 1)
    model = CompletionNetwork()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    # =============================================
    # Load input image
    # =============================================
    img = Image.open(args.input_img)
    img = transforms.Resize(args.img_size)(img)
    img = transforms.RandomCrop((args.img_size, args.img_size))(img)
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, dim=0)  # shape: [1, 3, H, W]

    # =============================================
    # Load precomputed mask
    # =============================================
    mask_array = np.load(args.tensor_img)
    print(f"Original mask shape: {mask_array.shape}")
    
    # Based on your mask creation code, we know the mask is in format [1, 1, h, w]
    # We need to resize it to match the img_size
    
    if mask_array.shape[2:] != (args.img_size, args.img_size):
        # Get the original mask content (remove batch and channel dimensions)
        mask_content = mask_array[0, 0]
        
        # Convert to PIL for resizing
        mask_pil = Image.fromarray((mask_content * 255).astype(np.uint8))
        
        # Resize to match input image dimensions
        mask_pil = mask_pil.resize((args.img_size, args.img_size), Image.NEAREST)
        
        # Convert back to numpy array
        resized_mask = np.array(mask_pil).astype(np.float32) / 255.0
        
        # Reshape back to [1, 1, h, w]
        mask_array = np.expand_dims(np.expand_dims(resized_mask, 0), 0)
        
        print(f"Resized mask to: {mask_array.shape}")
    
    # Convert to torch tensor
    mask = torch.from_numpy(mask_array).float()
    
    print(f"Input image shape: {x.shape}")
    print(f"Final mask shape: {mask.shape}")

    # =============================================
    # Inpainting
    # =============================================
    model.eval()
    with torch.no_grad():
        x_mask = x - x * mask + mpv * mask
        input = torch.cat((x_mask, mask), dim=1)
        output = model(input)
        inpainted = poisson_blend(x_mask, output, mask)
        imgs = torch.cat((x, x_mask, inpainted), dim=0)
        for i, img in enumerate(imgs):

            save_image(img, os.path.join(os.path.dirname(args.output_img) + "/" + os.path.basename(args.output_img)))

    print('Output image was saved as %s.' % args.output_img)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)