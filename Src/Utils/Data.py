import numpy as np
import os
import glob
from torch.utils.data import random_split
import torch
import random

# image path in my computer
data0="/home/gokul/g0kul6/Medical/Medical_Data/all-mias/"
data1="/home/gokul/g0kul6/Medical/Medical_Data/dental/"
data2=["/home/gokul/g0kul6/Medical/Medical_Data/Alzheimers Dataset/test/MildDemented/",
       "/home/gokul/g0kul6/Medical/Medical_Data/Alzheimers Dataset/test/ModerateDemented/",
       "/home/gokul/g0kul6/Medical/Medical_Data/Alzheimers Dataset/test/VeryMildDemented/",
       "/home/gokul/g0kul6/Medical/Medical_Data/Alzheimers Dataset/test/NonDemented/",
        "/home/gokul/g0kul6/Medical/Medical_Data/Alzheimers Dataset/train/MildDemented/",
       "/home/gokul/g0kul6/Medical/Medical_Data/Alzheimers Dataset/train/ModerateDemented/",
       "/home/gokul/g0kul6/Medical/Medical_Data/Alzheimers Dataset/train/VeryMildDemented/",
       "/home/gokul/g0kul6/Medical/Medical_Data/Alzheimers Dataset/train/NonDemented/"]
data3=["/home/gokul/g0kul6/Medical/Medical_Data/chest_xray/train/NORMAL/",
        "/home/gokul/g0kul6/Medical/Medical_Data/chest_xray/train/PNEUMONIA/",
        "/home/gokul/g0kul6/Medical/Medical_Data/chest_xray/test/NORMAL/",
        "/home/gokul/g0kul6/Medical/Medical_Data/chest_xray/test/PNEUMONIA/",
        "/home/gokul/g0kul6/Medical/Medical_Data/chest_xray/val/Normal/",
        "/home/gokul/g0kul6/Medical/Medical_Data/chest_xray/val/PNEUMONIA/"]

#image list
list0=glob.glob(os.path.join(data0,"*.pgm"))
list1=glob.glob(os.path.join(data1,"*.jpg"))
#list2=glob.glob(os.path.join(data2[0],"*.jpg"))
#list3=glob.glob(os.path.join(data2[1],"*.jpg"))
#list4=glob.glob(os.path.join(data2[2],"*.jpg"))
#list5=glob.glob(os.path.join(data2[3],"*.jpg"))
#list6=glob.glob(os.path.join(data2[4],"*.jpg"))
#list7=glob.glob(os.path.join(data2[5],"*.jpg"))
#list8=glob.glob(os.path.join(data2[6],"*.jpg"))
#list9=glob.glob(os.path.join(data2[7],"*.jpg"))
list10=glob.glob(os.path.join(data3[0],"*.jpeg"))
list11=glob.glob(os.path.join(data3[1],"*.jpeg"))
list12=glob.glob(os.path.join(data3[2],"*.jpeg"))
list13=glob.glob(os.path.join(data3[3],"*.jpeg"))
list14=glob.glob(os.path.join(data3[4],"*.jpeg"))
list15=glob.glob(os.path.join(data3[5],"*.jpeg"))

#train_test split for list0 and list1
list16=list0+list1
test=0.2
list16_train,list16_test=random_split(list16,[len(list16)-int(test*len(list16)),int(test*len(list16))],generator=torch.Generator().manual_seed(42))
list16_train=list(list16_train)
list16_test=list(list16_test)


#complete img list
train_list=list10+list11+list14+list15+list16_train
test_list=list12+list13+list16_test

#train_val split
random.shuffle(train_list)
random.shuffle(test_list)

val=0.13
train_list,val_list=random_split(train_list,[len(train_list)-int(val*len(train_list)),int(val*len(train_list))],generator=torch.Generator().manual_seed(42))

np.save("Src/Utils/Data List/train",train_list)
np.save("Src/Utils/Data List/val",val_list)
np.save("Src/Utils/Data List/test",test_list)
