from lib import *

def Make_data_list(path, phase='train'):
    data_list = []

    for path in glob.glob(root_path + phase + '\\**\\*.jpg'):
        data_list.append(path)
    return data_list

