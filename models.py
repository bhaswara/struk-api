import torch
import torch.nn as nn
from PIL import Image
import torchvision
from torchvision import transforms
import numpy as np
import io
import time
from skimage.filters import laplace
import base64

class StrukModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,6,3,1,1,bias=True),
            nn.ReLU(), 
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,8,3,1,1,bias=True),
            nn.ReLU(), 
            nn.MaxPool2d(2,2),
            nn.Conv2d(8,12,3,1,1,bias=True),
            nn.ReLU(), 
            nn.MaxPool2d(2,2),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(3072,256, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256,2, bias=True),
            nn.LogSoftmax()
        ) 
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x 


class StrukNet:
    def __init__(self):
        trained_weight_path = 'weights/student_.pth'

        self.classes = ['non_struk', 'struk']

        self.device = "cpu"

        self.model = StrukModel()
        self.model.load_state_dict(torch.load(trained_weight_path, map_location=self.device))
        self.model.eval()

        self.threshold = 0.01
    
    def infer(self, image):
        start_time = time.time()

        try:
            input_image = Image.open(io.BytesIO(image))
        except:
            input_image = Image.open(io.BytesIO(base64.b64decode(image)))
                   
        input_image = input_image.convert('RGB')

        with torch.no_grad():
            preprocess = transforms.Compose([transforms.Resize(135), 
                                    transforms.CenterCrop(128),
                                    transforms.ToTensor()])

            image = preprocess(input_image)

            fm = laplace(image).var()

            if fm < self.threshold:
                end_time = time.time()
                exec_time = end_time-start_time

                output = {"label": "blur", "execution_time": exec_time}
            else:
                image = image.unsqueeze(0)
                preds = self.model(image.to(self.device)).argmax(1)

                end_time = time.time()
                exec_time = end_time-start_time
            
                output = {"label": self.classes[preds.item()], "execution_time": exec_time}
        return output 
    

'''
if __name__ == '__main__':
    #start = time.time()
    model = StrukNet()
    output = model.infer('test_images/struk-png-1.png')
    #end = time.time()
    #elapsed = end-start
    print(output)
    #print(elapsed)
'''