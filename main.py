import cv2
import numpy as np
from PIL import Image
import time
import argparse
import cv2
from sklearn.cluster import KMeans
import torch,math,os,torchvision,random, einops
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import torchvision.transforms as tfms
import torchvision.transforms.functional as tfF
from torchvision.utils import *
from utils import load_pretrained_weights
import vision_transformer as models
import time
from torchvision import datasets

def get_args(*args):
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(add_help="bin/clusters.py --arch vit_base --pretrained_weights ../SiT/outputs/imagenet/sit-ViT_B/checkpoint.pth --img imgs/sample4.png")
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    
    parser.add_argument(
        '--img',
        type=str,
        default='./imgs/sample1.JPEG',
        help='Input image path')
    
    args = parser.parse_args(*args)
    
    return args


# Declare global variables
clicked = False
r = g = b = xpos = ypos = 0
threshold = 0.9


        
def calculate_similarity(f_query, f_key_list):
        
    sim = f_query.dot(f_key_list.transpose(0,2,1))
    return sim 
    
class PatchEmbedding:
    """
    该类加载了预训练的VIT_Base模型，可以对输入图像生成图像的patch token。
     Args:
        pretrained_weights (str): 预训练权重文件的路径。
        arch (str, optional): 模型使用的体系结构。默认为“vit_base”。
        patch_size (int, optional): 图像中提取的patch的大小。默认值为16。
     Attributes:
        model: 图像嵌入模型。
        embed_dim (int): 图像嵌入的维度。
     Methods:
        load_pretrained_weights(pretrained_weights): 载入预训练的权重到模型中。
        get_representations(image_path, tfms, denormalize): 为输入图像生成patch token。
    """
    def __init__(self, pretrained_weights, arch='vit_base', patch_size=16):
        self.model = models.__dict__[arch](patch_size=patch_size, num_classes=0)
        self.embed_dim = self.model.embed_dim
        self.model.eval().requires_grad_(False)
        self.load_pretrained_weights(pretrained_weights)
        
        
    def load_pretrained_weights(self, pretrained_weights):
        load_pretrained_weights(self.model, pretrained_weights)
        
    def get_representations(self, image_path, tfms):
        """
        生成输入图像的patch token。
         Args:
            image_path (str): 输入图像的路径。
            tfms (Transformation): 用于对输入图像进行变换的变换器。
         Returns:
            patch_tokens (ndarray): 表示生成的patch token的数组: N, C。
         """
        image = Image.open(image_path)
        img = tfms(image)
        x = img[None,:]
        patch_tokens = self.model.forward_features(x)[0,1:] # N - 1, C
        patch_tokens = nn.functional.normalize(patch_tokens, dim=-1, p=2).numpy()
        return patch_tokens
        
    def __call__(self, x) :
        patch_tokens = self.model.forward_features(x)[:,1:] # B, N - 1, C
        patch_tokens = nn.functional.normalize(patch_tokens, dim=-1, p=2).numpy()
        return patch_tokens


if __name__ == '__main__':
    args = get_args()
    ## load model
    from torchvision import datasets, transforms
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    common_tf = transforms.Compose([
        transforms.Resize(args.image_size),
        ])

    tfms=transforms.Compose([
        common_tf,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    def denormalize(imagesToPrint):
        imagesToPrint = imagesToPrint.clone()
        imagesToPrint *= torch.Tensor(IMAGENET_DEFAULT_STD,device=imagesToPrint.device).reshape(3,1,1)
        imagesToPrint += torch.Tensor(IMAGENET_DEFAULT_MEAN,device=imagesToPrint.device).reshape(3,1,1)
        return imagesToPrint.clamp(0,1)
        
    ## init model
    embedding = PatchEmbedding(args.pretrained_weights)
    
    # read image
    image = Image.open(args.img)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,args.image_size)
    
    ## get representations
    query = embedding.get_representations(args.img,tfms)
    
    keys = query[None,:]
    sim = calculate_similarity(query, keys) # QN, KB,KN
    p_sim = sim[:,0,:]
    
    nh,hw = args.image_size[0]//args.patch_size, args.image_size[1]//args.patch_size
    def mouse_callback(event, x, y, flags, param):
        """
        Mouse callback function
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            global clicked, xpos, ypos, r, g, b
            clicked = True
            xpos, ypos = x, y
            # Get the color of the selected pixel
            b, g, r = image[y, x]
            print("Selected color (BGR format): ", b, g, r, threshold)

    # Define the trackbar callback function
    
    def threshold_callback(value):
        global threshold
        threshold = value / 100
        print("New trackbar value:", threshold)
        
    cv2.namedWindow('Input image')
    cv2.setMouseCallback('Input image', mouse_callback)
    cv2.createTrackbar('Threshold', 'Input image', 0, 100, threshold_callback)
    cv2.namedWindow('Segmented image')
    while True:
        # Display the image
        cv2.imshow('Input image', image,)

        # Exit loop if ESC key is pressed
        if cv2.waitKey(1) == 27:
            break

        # If a pixel has been clicked, segment the surrounding object
        if clicked:
            # find the patch token
            p_x, p_y = xpos//args.patch_size, ypos//args.patch_size
            p_sim = sim[p_x+p_y*hw].reshape(hw,nh)
            
            # Create a mask for the selected color within a given threshold
            
            mask = (p_sim>=threshold).astype(np.float)
            mask = cv2.resize(mask,args.image_size)

            # Apply the mask to the original image
            alpha = 0.2
            segmented_image = (image*alpha + (1-alpha)*image * mask[:,:,None]).astype(np.uint8)

            # Display the resulting segmented image
            cv2.imshow('Segmented image', segmented_image,)

            # Reset the clicked flag
            clicked = False

    cv2.destroyAllWindows()


