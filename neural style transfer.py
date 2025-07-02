import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

def image_loader(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])     # transform it into a torch tensor
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)

def image_unloader(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    unloader = transforms.ToPILImage()
    return unloader(image)

content_img = image_loader('content.jpg', (128, 128))
style_img = image_loader('style.jpg', (128, 128))

cnn = models.vgg19(pretrained=True).features.eval()

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

model = nn.Sequential()
content_losses = []
style_losses = []

i = 0
for layer in cnn.children():
    if isinstance(layer, nn.Conv2d):
        i += 1
        name = f'conv_{i}'
    elif isinstance(layer, nn.ReLU):
        name = f'relu_{i}'
        layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
        name = f'pool_{i}'
    elif isinstance(layer, nn.BatchNorm2d):
        name = f'bn_{i}'
    else:
        continue
    model.add_module(name, layer)
    if name in content_layers:
        target = model(content_img).detach()
        content_loss = ContentLoss(target)
        model.add_module("content_loss_" + str(i), content_loss)
        content_losses.append(content_loss)
    if name in style_layers:
        target_feature = model(style_img).detach()
        style_loss = StyleLoss(target_feature)
        model.add_module("style_loss_" + str(i), style_loss)
        style_losses.append(style_loss)

for i in range(len(model) - 1, -1, -1):
    if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
        break
model = model[:i+1]

input_img = content_img.clone().requires_grad_(True)

optimizer = optim.LBFGS([input_img])
num_steps = 100

for step in range(num_steps):
    def closure():
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_score * 1e6 + content_score
        loss.backward()
        return loss
    optimizer.step(closure)

input_img.data.clamp_(0, 1)
output_image = image_unloader(input_img)
output_image.save('output.jpg')
print("Style transfer complete! Output saved as output.jpg")
