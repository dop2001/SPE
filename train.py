import torch
from torch.utils.data import DataLoader
from models.UNet import UNet
from utils.data_loader import MedicalDataset
from torchvision import transforms
from tqdm import tqdm
import os
from utils.metrics import *
from utils.loger import Loger, TensorboardWriter


def train(model, train_dataloader, val_dataloader, optimizer, criterion, network_config, loger):
    device = network_config["device"]
    img_w, img_h = network_config["image_size"]
    for i in range(network_config["epoch"]):
        model.train()
        train_loss, train_acc, train_iou = [], [], []
        for data in tqdm(train_dataloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            batch_size = images.shape[0]
            outputs = model(images)

            outputs = outputs.reshape(batch_size, 2, -1)
            labels = labels.reshape(batch_size, -1)

            # compute loss and update network's parameters
            loss = criterion(outputs, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            # calculate accuracy and IOU
            prediction = outputs.reshape(batch_size, 2, img_w, img_h).argmax(1)
            prediction = prediction.cpu().numpy().astype('uint8')
            labels = labels.reshape(batch_size, img_w, img_h)
            labels = labels.cpu().numpy().astype('uint8')

            train_acc.extend(accuracy_score(prediction, labels))
            train_iou.extend(iou(prediction, labels))

        # validation
        model.eval()
        val_loss, val_acc, val_iou = [], [], []
        with torch.no_grad():
            for data in tqdm(val_dataloader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                batch_size = images.shape[0]
                outputs = model(images)

                outputs = outputs.reshape(batch_size, 2, -1)
                labels = labels.reshape(batch_size, -1)

                loss = criterion(outputs, labels.long())
                val_loss.append(loss.item())

                prediction = outputs.reshape(batch_size, 2, img_w, img_h).argmax(1)
                prediction = prediction.cpu().numpy().astype('uint8')
                labels = labels.reshape(batch_size, img_w, img_h)
                labels = labels.cpu().numpy().astype('uint8')

                val_acc.extend(accuracy_score(prediction, labels))
                val_iou.extend(iou(prediction, labels))

        loger.info("epoch.{} | train_loss: {} - train_acc: {} - train_iou: {} | "
                   "val_loss: {} - val_acc: {} - val_iou: {}".format(i,
                                                                     sum(train_loss) / len(train_loss),
                                                                     sum(train_acc) / len(train_acc),
                                                                     sum(train_iou) / len(train_iou),
                                                                     sum(val_loss) / len(val_loss),
                                                                     sum(val_acc) / len(val_acc),
                                                                     sum(val_iou) / len(val_iou),
                                                                     ))

        IOU = sum(val_iou) / len(val_iou)
        torch.save(model.state_dict(), os.path.join(network_config["pth_save_path"], '{}_{}.pt'.format(i, IOU)))


def train_segmentation_model(train_config):
    train_prepare(train_config)
    # get loger and SummaryWriter
    loger = Loger(train_config.loger_config).getLoger()
    writer = TensorboardWriter(train_config.summaryWriter_config).getSummaryWriter()

    # get dataset_config and network_config
    dataset_config = train_config.dataset_config
    network_config = train_config.network_config

    # only training dataset use data augmentation
    train_transformers = transforms.Compose([
        transforms.autoaugment.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Resize(network_config["image_size"])
    ])

    others_transformers = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(network_config["image_size"])
    ])

    # DataLoader
    train_dataset = MedicalDataset(img_path=dataset_config['img_root_path'],
                                   lab_path=dataset_config['lab_root_path'], mode='train')
    validation_dataset = MedicalDataset(img_path=dataset_config['img_root_path'],
                                        lab_path=dataset_config['lab_root_path'], mode='valid')

    train_dataloader = DataLoader(train_dataset, batch_size=network_config["batch_size"],
                                  shuffle=True, num_workers=network_config["num_workers"])

    validation_dataloader = DataLoader(validation_dataset, batch_size=network_config["batch_size"],
                                       shuffle=False, num_workers=network_config["num_workers"])

    loger.info("training dataset size: {}".format(len(train_dataset)))
    loger.info("validation dataset size: {}".format(len(validation_dataset)))

    # UNet model
    model = UNet().to(network_config["device"])

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=network_config["lr"])

    # loss function
    weight = torch.tensor([1.0, 20.0]).to(network_config['device'])
    criterion = torch.nn.CrossEntropyLoss(weight=weight).to(network_config["device"])

    # start training
    train(model, train_dataloader, validation_dataloader, optimizer, criterion, network_config, loger)


def train_prepare(train_config):
    os.makedirs(os.path.dirname(train_config.loger_config['log_path']), exist_ok=True)
    os.makedirs(os.path.dirname(train_config.summaryWriter_config['summary_path']), exist_ok=True)
    os.makedirs(train_config.network_config['pth_save_path'], exist_ok=True)


if __name__ == '__main__':
    from configs.unet_config import UnetConfig
    train_config = UnetConfig()
    train_segmentation_model(train_config)

