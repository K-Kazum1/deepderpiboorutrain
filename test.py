import os
import requests
import argparse
import torch
import numpy as np
from PIL import Image
from lorafy import lorafy,loraweights,loadlora,DepthwiseSeparableConv
import open_clip
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize
from safetensors.torch import load_file, save_file

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default = 'jar/model.cookie')
parser.add_argument("--model_size", type=int, default = 1280)
parser.add_argument("--device", choices = ['cpu', 'cuda'], default = 'cpu')

args = parser.parse_args()

device = args.device

taglist = [i for i in open("tags",'r').read().split('\n') if i]


model, _, preprocess = open_clip.create_model_and_transforms(
  model_name="ViT-bigG-14",
  pretrained="laion2b_s39b_b160k"
)

model.to("cpu")
model=model.visual
model.to("cuda")
model.to(torch.float32)
model.train(False)
model.requires_grad_(False)

conv1 = DepthwiseSeparableConv(    3,     104, kernel_size=5, stride=5)
conv2 = DepthwiseSeparableConv(  104,   416, kernel_size=4, stride=4)
conv3 = DepthwiseSeparableConv(416, 1664, kernel_size=4, stride=4)

model.conv1 = torch.nn.Sequential(conv1,conv2,conv3)

convs = [conv1,conv2,conv3]

preprocess.transforms[0] = Resize(size=args.model_size, interpolation=Image.BICUBIC)
preprocess.transforms[1] = CenterCrop(size=args.model_size)

w=load_file(args.model_path)
loadlora(model,w=w)

conv1.depthwise.weight.data = w['conv1-depth']
conv2.depthwise.weight.data = w['conv2-depth']
conv3.depthwise.weight.data = w['conv3-depth']
conv1.pointwise.weight.data= w['conv1-point']
conv2.pointwise.weight.data= w['conv2-point']
conv3.pointwise.weight.data = w['conv3-point']

linear = torch.nn.Linear(1,1)
linear.weight.data = w['linear-weight']
linear.bias.data = w['linear-bias']

model = torch.nn.Sequential(model,linear).to(device)

def test(image):
    if os.path.exists(image):
        image = Image.open(image)
    else:
        image = Image.open(requests.get(image, stream=True).raw)
    with torch.no_grad():
        o = model(torch.unsqueeze(preprocess(image),0).to(device))[0]
    return ', '.join([taglist[i] for i in sorted(range(len(taglist)),key = lambda x:o[x],reverse=True)[:20]])

while image:=input(">>> "):
    print(test(image))
    
