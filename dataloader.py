"""
This is from the PyTorch dataloader tutorial at pytorch.org! Thanks!
"""
from __future__ import print_function, division
import os
import torch 
import pandas as pd
from skimage import io, transform 
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils 

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# plt.ion() # interactive

""" Dataset is in /faces
Comes with a CSV file with annotations that looks like this:

image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y
0805personali01.jpg,27,83,27,98, ... 84,134
1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312
"""

# Read the CSV
landmarks_frame = pd.read_csv('faces/face_landmarks.csv')

# get annotations into an (N,2 array) - N is number of landmarks
n = 65
img_name = landmarks_frame.iloc[n,0]
landmarks = landmarks_frame.iloc[n,1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1,2)


print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 landmarks: {}'.format(landmarks[:4]))

def show_landmarks(image,landmarks):
    # show the image with the landmarks
    plt.imshow(image)
    plt.scatter(landmarks[:,0],landmarks[:,1], s=10, marker='.', c='r')
    plt.pause(0.001) # give time for plot update

    plt.figure()

show_landmarks(io.imread(os.path.join('faces/', img_name)),
                landmarks)
plt.show()

""" dataset class 

torch.utils.data.Dataset is an abstract class representing a dataset

we inherit and override __len__ and __getitem__

"""
class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file (string): path to csv file with annotations
        root_dir (string): directory with all the images
        transform(callable, optional): optional transform to be applied on a sample
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform 

    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx,0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx,1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1,2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

# transforms

class Rescale(object):
    """
    Rescale images in a data set to be a given size
    
    args:
    output_size (tuple or int): If tuple, output is matched to output_size
                                If int, given size will be = smallest imag edge
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample): # this is called when we Rescale()
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w: # this maintains aspect ratio
                new_h, new_w = self.output_size * h / w, self.output_size
            else: 
                new_h, new_w = self.output_size, self.output_size * w / h    

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        landmarks = landmarks * [new_w / w, new_h / h] # scaling landmark coords?

        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """
    CROP RANDOMLY!

    args: output_size (tuple or int): desired output size. if int, output will be square
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, 
                      left: left + new_w]
              

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    """ convert ndarrays in sample to Tensors"""
    
    def __call__(self, sample):
        image, landmarks = sample['sample'], sample['landmarks']

        # we have to swap the colour axis because 
        # numpy image: H x W x C
        # torch image: C x H x W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

    
# instantiate the class and iterate through the data samples

face_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',
                                    root_dir='faces/')

# fig = plt.figure()

# for i in range(len(face_dataset)):
#     sample = face_dataset[i]

#     print(i, sample['image'].shape, sample['landmarks'].shape)

#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     # ax.set_title('Sample #{}').format(i)
#     ax.set_title('Whatevs!')
#     ax.axis('off')
#     show_landmarks(**sample) # basically treat the elements as entities rather than looking at the whole dict
    
#     if i == 3:
#         plt.show()
#         break

scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

# do transforms

fig = plt.figure()
sample = face_dataset[65]

for i, tsfrm in enumerate([scale, crop, composed]):
    transform_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transform_sample)
    
plt.show()