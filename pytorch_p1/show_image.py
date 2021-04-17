from lib import *

def show_image(img,label):
    image=img.numpy().transpose(1,2,0)
    np.clip(image,0,1)
    plt.imshow(image)
    if label==0:
        label='ferrari'
    elif label==1:
        label='lamborghini'
    elif label==2:
        label='vinfast'
    plt.title(label)
    plt.show()