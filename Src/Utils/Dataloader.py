import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image

#tranformations for train images
train_transforms=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor()])
#transformation for val images
val_transforms=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                  transforms.Resize((224, 224)),
                                  transforms.ToTensor()])
#transformation for test imagese
test_transforms=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()])   


#tranforming the dataset
class dataset(Dataset):
    def __init__(self,file_list,transform=None):
        self.file_list=file_list
        self.transform=transform
    def __len__(self):
        self.filelength=len(self.file_list)
        return self.filelength
    def __getitem__(self,idx):
        img_path=self.file_list[idx]
        img=Image.open(img_path)
        img_transformed=self.transform(img)
        return img_transformed


#creating data loader for train, val and test    
class Loader():
    def __init__(self,data,batch_size):
        self.data=data
        self.batch_size=batch_size
    def load(self):
        loader=DataLoader(dataset = self.data,batch_size=self.batch_size,shuffle=True)
        return loader






