from lib import *

use_prepare_train = True

model = models.vgg16(use_prepare_train)
model.train()
model.classifier[6] = nn.Linear(in_features=4096, out_features=3)

# config model
name_edit=['classifier.6.weight','classifier.6.bias']

param_update=[]
for name,param in model.named_parameters():
    if name in name_edit:
        param.requires_grad=True
        param_update.append(param)
    else:
        param.requires_grad=False

