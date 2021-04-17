from lib import *
from make_data_list import *

class Transform_image():
    def __init__(self,resize,mean,std):
        self.transform={
            'train':transforms.Compose([
                transforms.RandomResizedCrop(resize,(0.8,1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ]),
            'val':transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])

        }

    def __call__(self,image,phase='train'):
        return self.transform[phase](image)

