import torch
from Src.Utils.NoiseGenerator import add_gaussian,s_p #noise in gaussaian and impulse(salt and pepper)
from Src.Utils.Dataloader import dataset,Loader,train_transforms,test_transforms,val_transforms
import numpy as np

#intializing to gpu or cpu depending on availabilty
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#list od img paths for train,val and test
train_list=np.load("Src/Utils/Data List/train.npy")
val_list=np.load("Src/Utils/Data List/val.npy")
test_list=np.load("Src/Utils/Data List/test.npy")

#transform data
train_data=dataset(train_list,transform=train_transforms)
val_data=dataset(val_list,transform=val_transforms)
test_data=dataset(test_list,transform=test_transforms)

#creating loader 
train=Loader(train_data,batch_size=16)
val=Loader(val_data,batch_size=16)
test=Loader(test_data,batch_size=16)

#train,test and val loader
train_loader=train.load()
val_loader=val.load()
test_loader=test.load()

