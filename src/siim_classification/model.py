# once we have engine.py, the next step is to create "model.py"
# it will consist of our model
# its a good idea to keep it separate as allows us to easily experiment with different models and architectures
# A PyTorch libary - "pretrainedmodels" has a lot of different architectures, such as AlexNet, ResNet, DenseNet etc..

import torch.nn as nn
import pretrainedmodels

def get_model(pretrained):
    if pretrained:
        model = pretrainedmodels.__dict__["alexnet"](pretrained="imagenet")
    else:
        model = pretrainedmodels.__dict__["alexnet"](pretrained=None)
    # print the model to see whats going on
    # print(model)
    model.last_linear = nn.Sequential(nn.BatchNorm1d(4096),
                                        nn.Dropout(p=0.25),
                                        nn.Linear(in_features=4096, out_features=2048),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(in_features=2048, out_features=1))

    return model

print(get_model(pretrained=True))