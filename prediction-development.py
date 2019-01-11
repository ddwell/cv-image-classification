
# coding: utf-8

# In[1]:


# Import needed packages
import os#, datetime, random
import numpy as np
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.autograd import Variable


class Unit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class Network(nn.Module):
    def __init__(self,num_classes=3):
        super(Network,self).__init__()
        
        #Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3,out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)
        
        #Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 ,self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=128,out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,128)
        output = self.fc(output)
        return output
    
    
class IPSModel():
    def __init__(self, cuda=True, model_path=None, image_size=32, num_classes=3):
        self.device = torch.device("cuda" if cuda else "cpu")
        self.model = Network(num_classes=3)

        if model_path is None:
            self.model_path = os.path.join(os.getcwd(),'models','ips_model_32.dict')
        else:
            self.model_path = model_path
        
        self.image_size = image_size
        
        self.labels_mapping = {
            0: 'image',
            1: 'photo',
            2: 'screenshot'
        }
        self.model = Network(num_classes=num_classes)
        self.checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(self.checkpoint)
        self.model.eval()
        
    def get_most_frequent(self, array):
        most_common, num_most_common = Counter(array).most_common(1)[0] 
        return most_common

    def predict_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((self.image_size,self.image_size)).convert('RGB')
        transformation = transforms.Compose([
                    transforms.Resize((self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image_tensor = transformation(image).float().unsqueeze_(0)   
        input = Variable(image_tensor)
        output = self.model(input)
        probabilities = [np.argmax(item) for item in output.data.numpy()]
        index = self.get_most_frequent( probabilities )
        return index


# In[10]:


# MODEL_PATH = 'models/ips_model_71_.model'#
MODEL_PATH = 'models/ips_model_32.dict'
IMAGE_SIZE = 32

labels_mapping_full = {
    'image': 0,
    'photo': 1,
    'screenshot': 2,
    0: 'image',
    1: 'photo',
    2: 'screenshot'
}

ips_model = IPSModel(model_path=MODEL_PATH, image_size=IMAGE_SIZE, num_classes=3)
ips_model.model.eval()


# In[54]:


ips_model.predict_image(file_path)

image_file = 'data/example/orig_NNvqMow.gif'
# image_file = 'data/example/7blsbotrkxoz.jpg'
# image_file = 'data/example/screenshot-android.png'

image_path = os.path.join(os.getcwd(), image_file)

index = predict_image(image_path, loaded_model)
print("Predicted Class: %d, %s" % (index, labels_mapping_full[index]))

