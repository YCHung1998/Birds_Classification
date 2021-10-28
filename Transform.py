from torchvision import transforms
from torchvision.io import read_image
'''
Simulate monai's Dictionary Transforms writing 
https://docs.monai.io/en/latest/transforms.html#dictionary-transforms
'''

class read_imaged(): 
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = read_image(data[key])
            else:
                 raise KeyError(f'{key} is not a key of {data}')

        return data

    
class Resized(): 
    def __init__(self, keys, size=(384,384)):
        self.keys = keys
        self.size = size
    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = transforms.Resize(self.size)(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return data
    
    
class RandHoriFlipd():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = transforms.RandomHorizontalFlip()(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return data
    
    
class RandRotd():
    def __init__(self, keys, degrees=10):
        self.keys = keys
        self.degrees = degrees

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = transforms.RandomRotation(self.degrees)(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return data


class RandPerspectived():  # Not be used
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')        

        return data
        

class RandResizedCropd():  # Not be used
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = transforms.RandomResizedCrop(size=300, scale=(0.08, 1))(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')       

        return data


class RandAffined():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = transforms.RandomAffine(0, shear=10, scale=(0.8,1.2))(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')   

        return data
    
    
class CenterCropd():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = transforms.CenterCrop(200)(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')   

        return data
    

class RandomCropd():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = transforms.RandomCrop((375,375))(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')       

        return data


class Normalized():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')
                
        return data    

    
class Scale01d():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = data[key].float()/255
            else:
                raise KeyError(f'{key} is not a key of {data}') 

        return data  

    
import torch
class RandomGaussianNoise(object):
    def __init__(self, sig=0.01, p=0.5):
        self.sig = sig
        self.p = p
        
    def __call__(self, img):
        if random.random() > self.p:
            img += self.sig * torch.randn(img.shape)

        return img

    
class RandGaussNoised(RandomGaussianNoise):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys
        
    def __call__(self, data):
        for key in self.keys:
            if key in data:
                 data[key] = RandomGaussianNoise()(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')  

        return data

    
    
import random    
class GridMask(object):
    def __init__(self, dmin=90, dmax=160, ratio=0.8, p=0.6):
        """Original Setting : dmin=90, dmax=300, ratio=0.7, p=0.5
        after augmentation, again masking with (90, 160, 0.8, 0.6) 
        [ dmin, dmax ] : range of the d in uniform random 
        random variable probibilaty > p, swith on the function"""
        self.dmin   = dmin 
        self.dmax   = dmax
        self.ratio  = ratio
        self.p = p
        
    def __call__(self, Img):
        if random.random() < self.p:
            return Img
        d = random.randint(self.dmin, self.dmax)
        dx, dy = random.randint(0,d-1), random.randint(0,d-1)
        sl = int((1-self.ratio)*d)
        for i in range(dx, Img.shape[1], d):
            for j in range(dy, Img.shape[2], d):
                row_end, col_end = min(i+sl, Img.shape[1]), min(j+sl, Img.shape[2])
                Img[:, i:row_end, j:col_end] = 0

        return Img
        
        
class GridMaskd(GridMask):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys
        
    def __call__(self, data):
        for key in self.keys:
            if key in data:
                 data[key] = GridMask()(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return data

 
        
class Transforms():
    def __init__(self, size = 500):
        self.size = size
        
    def train_transforms(self):
        train_trans = transforms.Compose([read_imaged(keys=['image']),
                                          Scale01d(keys=['image']),
                                          Resized(keys=['image'], size=(self.size,self.size)),
                                          RandGaussNoised(keys=['image']),
                                          transforms.RandomOrder(
                                              [RandHoriFlipd(keys=['image']),
                                              RandRotd(keys=['image'], degrees=10),
                                              RandAffined(keys=['image']) 
                                              ]),
                                          GridMaskd(keys=['image']),
                                          Normalized(keys=['image'])
                                         ])

        return train_trans

    def valid_transforms(self):
        valid_trans = transforms.Compose([read_imaged(keys=['image']),
                                          Scale01d(keys=['image']),
                                          Resized(keys=['image'], size=(self.size,self.size)),
                                          Normalized(keys=['image'])])

        return valid_trans

    def test_transforms(self): 
        test_trans = transforms.Compose([read_imaged(keys=['image']),
                                        Scale01d(keys=['image']),
                                        Resized(keys=['image'], size=(self.size,self.size)),
                                        Normalized(keys=['image'])])

        return test_trans
