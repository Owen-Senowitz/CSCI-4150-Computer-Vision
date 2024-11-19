### Code based on https://lightning.ai/lightning-ai/studios/image-segmentation-with-pytorch-lightning

import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision

transform_source = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256,256)),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

transform_target = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256,256))])

train_dataset = torchvision.datasets.VOCSegmentation("seg_data/","2012","trainval",transform=transform_source,target_transform=transform_target,download=True)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.8), int(len(train_dataset)*0.1), len(train_dataset) - int(len(train_dataset)*0.9)])

train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=8)
test_loader = DataLoader(test_dataset,batch_size=8)


class LitNetwork(pl.LightningModule):
    def __init__(self):
        super(LitNetwork, self).__init__()

        self.model = torchvision.models.segmentation.fcn_resnet50(num_classes=21)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy("multiclass",num_classes=21,average='micro')
        self.test_acc = torchmetrics.Accuracy("multiclass",num_classes=21,average='micro')

    def forward(self, x):
        return self.model(x)["out"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, data, batch_idx):
        im, targets = data[0], data[1]
        outs = self.forward(im)
        loss = self.loss_func(outs, targets.long().squeeze(1))
        self.log("train_loss",loss,prog_bar=True,on_step=True,on_epoch=True,batch_size=8,sync_dist=True)
        return loss
    
    def validation_step(self, val_data, batch_idx):
        im, targets = val_data[0], val_data[1]
        outs = self.forward(im)
        self.val_acc(outs,targets.long().squeeze(1))
        self.log("val_acc",self.val_acc,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return None

    def test_step(self, test_data, batch_idx):
        im, targets = test_data[0], test_data[1]
        outs = self.forward(im)
        self.test_acc(outs,targets.long().squeeze(1))
        self.log("test_acc",self.test_acc,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return None


model = LitNetwork()
checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max')
logger = pl_loggers.TensorBoardLogger(save_dir="my_logs")
#logger = pl_loggers.CSVLogger(save_dir="my_logs",name="my_csv_logs")

device = "gpu" # Use 'mps' for Mac M1 or M2 Core, 'gpu' for Windows with Nvidia GPU, or 'cpu' for Windows without Nvidia GPU

trainer = pl.Trainer(max_epochs=10, accelerator=device, callbacks=[checkpoint], logger=logger)
trainer.fit(model,train_loader,val_loader)
    
trainer.test(ckpt_path="best", dataloaders=test_loader)

#Visualize Example Output
import matplotlib.pyplot as plt
import numpy as np
im = plt.imread("my_example.jpg")
im_tensor = transform_source(im).unsqueeze(0)
out = model(im_tensor)
out = torch.argmax(out,dim=1).squeeze(0)
out = out.detach().cpu().numpy()
colors = np.random.randint(0,255,(21,3))
out_rgb = np.zeros((256,256,3),dtype=np.uint8)
for i in range(1,21):
    out_rgb[out==i] = colors[i]
plt.subplot(1,2,1)
plt.imshow(im)
plt.subplot(1,2,2)
plt.imshow(out_rgb)
plt.show()
