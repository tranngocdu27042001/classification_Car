from lib import *
from make_data_list import Make_data_list


# dung cho data_loader
class My_dataset(data.Dataset):
    def __init__(self, data_list, transform, phase='train'):
        self.data_list = data_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path = self.data_list[index]
        image = Image.open(image_path)
        image_transformed = self.transform(image, self.phase)
        if self.phase == 'train':
            label = image_path[58:65]
        elif self.phase == 'val':
            label = image_path[58:62]
        if label == 'ferrari':
            label = 0
        elif label == 'lamborg':
            label = 1
        elif label == 'vinfast':
            label = 2

        return image_transformed, label
