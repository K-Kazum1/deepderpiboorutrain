import json
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
parser.add_argument("--image_dir", default = "image")
parser.add_argument("--model_path", default = 'jar/model.cookie')
parser.add_argument("--wandb", default = '')
parser.add_argument("--train_base", action = 'store_true')
parser.add_argument("--train_conv", action = 'store_true')
parser.add_argument("--load", action = 'store_true')
parser.add_argument("--model_name", default = 'ViT-bigG-14')
parser.add_argument("--model_pretrained", default = 'laion2b_s39b_b160k')
parser.add_argument("--silent", action = 'store_true')
parser.add_argument("--skip_high_loss", action = 'store_true')
parser.add_argument("--lr", default = 3e-4,type=float)

args = parser.parse_args()

if args.wandb:
    import wandb
    wandb.init(project=args.wandb)


device = "cuda" if torch.cuda.is_available() else "cpu"

if device == 'cpu':
    print('no gpu detected, you reaaaaly should be using gpu for training')

imagedir =  args.image_dir
print('loading tags')
taglist = open("../tags",'r').read().strip().split('\n')
tags = json.load(open('../tags.json'))


def get_params(layers):
    l=[]
    for i in layers:
        for j in i.parameters():
            l.append(j)
    return l

def surgery(model,t):# expands the final linear layer with more tags without affecting the rest of the linear layer
    model,linear = model.children()

    weight = linear.weight

    weight = torch.cat((weight,torch.normal(weight.mean().expand(t-weight.shape[0],weight.shape[1]),weight.std()).to(device)),0)

    linear = torch.nn.Linear(linear.in_features,t).to('cuda')

    linear.weight= torch.nn.Parameter(weight)

    model = torch.nn.Sequential(model,linear)

    return model     
    
def _convert_to_rgb(image):
    return image.convert('RGB')

def filter(tag):
    tag = tag.split(',')
    a = open('ids','w')
    c = 0
    for i in tags:
        if not any(j in tag for j in tags[i]):
            a.write(i)
            c+=1
    print(c)
    a.close()

def test(image):
    if os.path.exists(image):
        image = Image.open(image)
    else:
        image = Image.open(requests.get(image, stream=True).raw)
    with torch.no_grad():
        o = model(torch.unsqueeze(preprocess(image),0).to(device))[0]
    return ', '.join([taglist[i] for i in sorted(range(len(taglist)),key = lambda x:o[x],reverse=True)[:20]])


                     
def clip_tanh(opt):
    # Gradient clipping
    CLIP_MAX = 3 # standard deviations
    with torch.no_grad():
        for group in opt.param_groups:
            for p in group['params']:
                st = opt.state[p]
                if p.grad is not None and 'exp_avg_sq' in st:
                    beta2 = group['betas'][1]
                    std = (st['exp_avg_sq'] / (1 - beta2**st['step'])).sqrt() + 1e-8
                    bound = std * CLIP_MAX
                    p.grad = torch.tanh(p.grad / bound) * bound
                    
def tag2tensor(id):
    tag = tags[str(id)]
    return torch.tensor([i in tag for i in taglist], dtype = torch.float32)

class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, *patterns):
        self.filenames = []
        self.preprocess = preprocess
        for pat in patterns:
            
            if os.path.exists(f:=f'{imagedir}/{pat}.png'):
                self.filenames.append(f)
            elif os.path.exists(f:=f'{imagedir}/{pat}.jpg'):
                self.filenames.append(f)

    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        if isinstance(idx, list) or isinstance(idx, tuple):
            return [self[j] for j in idx]
        if isinstance(idx, str):
            if os.path.exists(f:=f'{imagedir}/{idx}.png'):
                filename = f'{imagedir}/{idx}.png'
            else:
                filename = f'{imagedir}/{idx}.jpg'
        else:
            filename = self.filenames[idx]
        try:
            image = Image.open(filename).convert('RGB')
        except Exception as e:
            print(filename,e)
            #raise e

        id = filename[len(imagedir)+1:-4]

        return preprocess(image).to(device), tag2tensor(id).to(device),id

print('loading model')

model, _, preprocess = open_clip.create_model_and_transforms(
  model_name=args.model_name,
  pretrained=args.pretrained
)

model.to("cpu")
model=model.visual # clip comes with a language embedding transformer, only need the vision transformer
model.to("cuda")
model.to(torch.float32)
model.train(False)
model.requires_grad_(False)# want to turn off training for the base weights and only train the lora/convolution/linear layers

modelconv =  model.trunk.patch_embed.proj if isinstance(model,open_clip.timm_model.TimmModel) else model.conv1

if args.train_conv:
    model.to("cpu")
    modelconv.to("cuda")
    conv1 = DepthwiseSeparableConv(    3,     104, kernel_size=5, stride=5)
    conv2 = DepthwiseSeparableConv(  104,   416, kernel_size=4, stride=4)
    conv3 = DepthwiseSeparableConv(416, modelconv.out_channels, kernel_size=4, stride=4,bias = isinstance(modelconv.bias,torch.nn.parameter.Parameter))
    weights = get_params([conv1,conv2,conv3])
    conv = modelconv

    preprocess.transforms[0] = Resize(size=1280, interpolation=Image.BICUBIC)
    preprocess.transforms[1] = CenterCrop(size=1280)
else:
    if args.train_base:
        
        convs = [modelconv]
    else:
        conv1 = DepthwiseSeparableConv(    3,     104, kernel_size=5, stride=5)
        conv2 = DepthwiseSeparableConv(  104,   416, kernel_size=4, stride=4)
        conv3 = DepthwiseSeparableConv(416, modelconv.out_channels, kernel_size=4, stride=4)

        modelconv = torch.nn.Sequential(conv1,conv2,conv3)

        convs = [conv1,conv2,conv3]

        preprocess.transforms[0] = Resize(size=1280, interpolation=Image.BICUBIC)
        preprocess.transforms[1] = CenterCrop(size=1280)

        if isinstance(model,open_clip.timm_model.TimmModel):
            model.trunk.patch_embed.img_size  = (1280, 1280)
        

    if args.load:
        w=load_file(args.model_path)
        #w = {i:torch.nn.Parameter(w[i]) for i in w}
        loadlora(model,w=w)
        if not args.train_base:
            conv1.depthwise.weight.data = w['conv1-depth']
            conv2.depthwise.weight.data = w['conv2-depth']
            conv3.depthwise.weight.data = w['conv3-depth']
            conv1.pointwise.weight.data= w['conv1-point']
            conv2.pointwise.weight.data= w['conv2-point']
            conv3.pointwise.weight.data = w['conv3-point']

        linear = torch.nn.Linear(1,1)#weight size replaces the input and output shape
        linear.weight.data = w['linear-weight']
        linear.bias.data = w['linear-bias']
        
        model = torch.nn.Sequential(model,linear)

    else:

        lorafy(model,4)

        if isinstance(model,open_clip.timm_model.TimmModel):
            output_dim = model.trunk.head.LoRAout.out_features
        else:
            output_dim = model.output_dim

        linear = torch.nn.Linear(output_dim,len(taglist))

        model = torch.nn.Sequential(model,linear)

    model.to('cuda')
    weights = loraweights(model[0])

    if model[1].out_features < len(taglist):
        model = surgery(model,len(taglist))

    weights = list(weights.values()) + get_params(convs+[linear])    

    if isinstance(model[0],open_clip.timm_model.TimmModel):
        model[0].trunk.patch_embed.proj = modelconv
    else:
        model[0].conv1 = modelconv
    model.to('cuda')



print('model ready')
dataset = ImagesDataset(*open('../ids').read()[:-1].split(','))
print('dataset ready')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

if os.path.exists(f:='jar/mean'):
    mean = load_file(f)['mean']
else:
    print('getting tag frequency')
    with torch.no_grad():
        mean = torch.stack(tuple([tag2tensor(i) for i in tags])).float().mean(0)
    save_file({'mean':mean},f)

mean = mean.to(device)



def forever(dl):
    while True:
        for item in dl:
            yield item
train_it = forever(dataloader)

losses = []
i=0

print('train start')

def approx():#tries to make the large image convolution output the same features as the base convolution

    base_preprocess = Compose( [# preprocess used in clip originally
                                Resize(size=224, interpolation=Image.BICUBIC),
                                CenterCrop(size=224),
                                _convert_to_rgb,
                                ToTensor(),
                                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                            ])

    model = torch.nn.Sequential(conv1,conv2,conv3).to(device)

    lossf = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),args.lr)
    optimizer.zero_grad()
    losses = []

    def im(id):
        id = id[0]
        if os.path.exists(f:=f'{imagedir}/{id}.png'):
            filename = f'{imagedir}/{id}.png'
        else:
            filename = f'{imagedir}/{id}.jpg'
        return Image.open(filename).convert('RGB')

    i = 0
    while True:
        i+=1
        try:
            x,y,ids = next(train_it)
        except:
            train_it = forever(dataloader)
            x,y,ids = next(train_it)

        loss = lossf(model(preprocess(im(ids)).to(device)),conv(base_preprocess(im(ids)).to(device)))
        loss.backward()

        losses.append(loss.item())

        if i%400==0:
            clip_tanh(optimizer)
            optimizer.step()
            optimizer.zero_grad()
            if not args.silent:
                print(i//400, np.mean(losses),np.std(losses),max(losses),min(losses))
            losses=[]

def train():

    lossf = torch.nn.BCEWithLogitsLoss(pos_weight = 1/mean-1)
    optimizer = torch.optim.Adam(weights, lr=args.lr)
    optimizer.zero_grad()
    losses = []
    i=0
    while True:
        i+=1
        
        try:
            x,y,ids = next(train_it)
        except:
            train_it = forever(dataloader)
            x,y,ids = next(train_it)

        loss = lossf(x:=model(x),y)

        if args.wandb:
            wandb.log({"loss": loss.item(),'epoch':i})
        
        if loss>5:
            if not args.silent:
                print('\t',int(ids[0]),loss.item())
            if args.skip_high_loss:
                loss *=0
        else:
            losses.append(loss.item())

        loss.backward()
        
        if i%400==0:
            clip_tanh(optimizer)
            optimizer.step()
            optimizer.zero_grad()
            if args.wandb:
                wandb.log({'global_step': i//400, 'mean':np.mean(losses).item(), 'std':np.std(losses).item()})
            if not args.silent:
                print(i//400, np.mean(losses),np.std(losses),max(losses),min(losses))
            losses=[]

if args.train_conv:
    try:
         approx()
    except KeyboardInterrupt:
        print('training interrupted')
        if os.path.exists(args.model_path):
            w = load_file(args.model_path)
        else:
            w = {}
        w.update({'conv1-depth':conv1.depthwise.weight,'conv2-depth':conv2.depthwise.weight,'conv3-depth':conv3.depthwise.weight,
                          'conv1-point':conv1.pointwise.weight,'conv2-point':conv2.pointwise.weight,'conv3-point':conv3.pointwise.weight})
        
        save_file(w,args.model_path)
else:
    try:
        train()
    except KeyboardInterrupt:
        print('training interrupted')
        model.to('cpu')
        w = loraweights(model[0])
        if not args.train_base:
            w.update({'conv1-depth':conv1.depthwise.weight,'conv2-depth':conv2.depthwise.weight,'conv3-depth':conv3.depthwise.weight,
                              'conv1-point':conv1.pointwise.weight,'conv2-point':conv2.pointwise.weight,'conv3-point':conv3.pointwise.weight})
        w.update({'linear-weight':model[1].weight,'linear-bias':model[1].bias})
        
        save_file(w,args.model_path)

print('model saved')
