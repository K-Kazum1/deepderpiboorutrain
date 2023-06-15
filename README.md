# deepderpiboorutrain
Train an image tagger model using open clip's vision model, with LoRA

# Setup

do `pip install -r requirements.txt` with pip or `conda install --file requirements.txt`
Setup a folder with all the images you want to train on, preferably at `./image`, but you can use the `--image_dir` flag
Setup a `./tags.json` that lists the image name as the key and a list of tags as the values
Setup a `./tags` that lists the tags you want the model to be able to tag, separated by new lines

The first time you train the model, it might take a while to calculate the frequency of the tags in the dataset.

# Training
1. run `python train.py --train_conv` to align a new set of convolution layers to the pretrained convolution, this typically takes about 20 minutes
2. run `python train.py --train_base` to train the LoRA weights in the original image sizes. This might take a few hours or a day.
3. run `python train.py` to train the whole model with the convolution.
