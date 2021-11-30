# encoding:utf-8
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os
from data import MyDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

test_transform=transforms.Compose([
                                                transforms.Resize((600, 600), Image.BILINEAR),
                                                transforms.CenterCrop(448),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                ])


car_test_root='/home/ubuntu/erick/FGVC/HOI-Net/cub_test_list.txt'
testset =  MyDataset(car_test_root,transforms=test_transform)                                              
testloader = torch.utils.data.DataLoader(testset, batch_size=20,
                                         shuffle=False, num_workers=6)
criterion = nn.NLLLoss()
model = torch.load('HOI-resnet101.pth')      # model = torch.load('HOI-resnet50.pth')                                  

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
       for data, target in testloader:
          data, target = data.cuda(), target.cuda()
          output= model(data)
          test_loss += criterion(output, target).data.item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).cpu().sum()
       test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss * 8., correct, len(testloader.dataset),
        100.0 * float(correct) / len(testloader.dataset)))
    return float(correct) / len(testloader.dataset)

acc=test()


