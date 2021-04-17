from lib import *
from make_data_list import Make_data_list
from transform import Transform_image
from my_dataset import My_dataset
from transfer import *
from show_image import show_image

# tao list path anh
data_list_path = Make_data_list(root_path, phase='train')
# tao transform
transform = Transform_image(resize, mean, std)
# tao data_set_train
train_data_set = My_dataset(data_list=data_list_path, transform=transform, phase='train')

data_loader_train = data.DataLoader(train_data_set, batch_size=batch_size, shuffle=True)

# print(len(data_loader_train.dataset))

loss_func = nn.CrossEntropyLoss()

optimizer = optim.SGD(param_update, lr=0.01, momentum=0.9)


def train_model(models, loss_func, optimizer, epochs, data_loader):
    models.train()
    epoch_corres = 0.0
    num_epoch_correct = 0
    for epoch in range(epochs):
        print('epoch/epochs:{}/{}'.format(epoch,epochs))
        for inputs, label in tqdm(data_loader):
            optimizer.zero_grad()
            output = models(inputs)
            loss = loss_func(output, label)
            _, predict = torch.max(output, 1)

            loss.backward()
            optimizer.step()

            epoch_corres += loss.item() * inputs.size(0)
            num_epoch_correct += torch.sum(predict == label)

        epoch_corres = epoch_corres / len(data_loader.dataset)
        accuracy = num_epoch_correct / len(data_loader.dataset)
        print('epoch:{},loss:{:.4f},accuracy:{:.4f}'.format(epoch, epoch_corres, accuracy))


train_model(model, loss_func, optimizer, epochs, data_loader_train)
