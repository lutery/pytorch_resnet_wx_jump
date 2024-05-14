#写一个dataset的示例类

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os

class WXJumpDataset(data.Dataset):
    def __init__(self, imgpath, transform=None):
        self.imgpath = imgpath
        self.transform = transform
        self.transform = transform
        self.imglist = []
        self.imglabel = []
        self.min_label = 10000
        self.max_label = 0
        self.loadimgs()
        self.num_samples = len(self.imglist)


    # 将标签设置为one-hot编码
    def loadimgs(self):
        for imgfile in os.listdir(self.imgpath):
            imgfilepath = os.path.join(self.imgpath, imgfile)
            self.imglist.append(imgfilepath)
            # 从文件名分割出标签，文件名：i-index-标签.png
            label_png = imgfile.split('-')[-1]
            label = float(label_png.split('.')[0] + "." + label_png.split('.')[1])
            self.imglabel.append(int(round(label, 2) * 100))
            if  self.imglabel[-1] < self.min_label:
                self.min_label = label
            if self.imglabel[-1] > self.max_label:
                self.max_label = label
            

    def __getitem__(self, index):
        img = Image.open(self.imglist[index]).convert('RGB')
        label = self.imglabel[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return self.num_samples
    

# 写一段测试pytorch dataset有效性
if __name__ == '__main__':
    imgpath = r'F:\Projects\datasets\resnet\wx_jump\train'
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        WXJumpDataset(imgpath= imgpath, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            # normalize,
        ])),
        batch_size=32, shuffle=True, pin_memory=True)
    
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        # 显示索引为0的imgs
        print(batch_idx, imgs.size(), labels.size())
        import matplotlib.pyplot as plt
        plt.imshow(imgs[0].permute(1, 2, 0))
        plt.show()
        print(labels[0])
        break
                            
