import time

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from data import create_dataset, load_data

step = 0
loss_list = {"Adam": [], "SGD": [], "AdamW": []}
acc_list = {"Adam": [], "SGD": [], "AdamW": []}
cuda_memory_list = {"Adam": [], "SGD": [], "AdamW": []}
time_list = {"Adam": None, "SGD": None, "AdamW": None}
test_acc = {"Adam": None, "SGD": None, "AdamW": None}


class Net(pl.LightningModule):
    def __init__(self, lr=0.001, opt_index=0, batch_norm=True, dropout=0.0, cnn=True):
        super(Net, self).__init__()

        if batch_norm:
            if cnn:
                # CNN model with batch normalization
                # 结构：卷积层->激活->池化层->卷积层->激活->池化层->全连接层 实验结果是和全连接差不多 一层全连接层解码能力不够，信息损失比较大
                # 结构：卷积层->激活->池化层->全连接层->全连接层 参数量大了八倍，解码能力更强 实验效果adam和adamw上升一个点 sgd的效果提高了20+点
                # 结构：卷积层->激活->池化层->全连接层->全连接层->全连接层 参数量大了约26倍，解码能力更强 实验效果：对adam和adamw的效果几乎没有提示，但是对sgd的效果提高了20+点
                self.model = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Flatten(),
                    nn.Linear(16 * 14 * 14, 768),  # 全连接层 7 = 28/2/2 14 = 28/2
                    nn.BatchNorm1d(768),
                    nn.ReLU(),
                    nn.Linear(768, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, 10),
                    nn.Softmax(dim=1)
                )
            else:
                # MLP model with batch normalization
                self.model = nn.Sequential(
                    nn.Linear(28 * 28, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Linear(64, 10),
                    nn.LogSoftmax(dim=1)
                )

        else:
            self.model = nn.Sequential(
                nn.Linear(28 * 28, 120),
                nn.ReLU(),
                nn.Linear(120, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
                nn.LogSoftmax(dim=1)
            )
        self.cnn = cnn
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.automatic_optimization = False
        # 本次实验显示，adam和adamw的效果差不多，但是adamw比adam更快
        # sgd最快但是效果比adam和adamw差，在使用全连接时，的准确率只有0.09

        self.opt_list = [

            optim.Adam(self.parameters(), lr=self.lr),
            optim.SGD(self.parameters(), lr=self.lr),
            optim.AdamW(self.parameters(), lr=self.lr)
        ]
        self.opt = self.opt_list[opt_index]

    def forward(self, x):
        if self.cnn:
            x = x.view(-1, 1, 28, 28)
        else:
            x = x.view(-1, 28 * 28)
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.opt.zero_grad()
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        cuda_memory_list[self.opt.__class__.__name__].append(torch.cuda.max_memory_allocated() / 1024 / 1024)
        self.manual_backward(loss)
        self.opt.step()

        log_dict = {'train_loss': loss, 'train_acc': torch.mean((torch.argmax(y_pred, dim=1) == y).float())}
        self.log_dict(log_dict, prog_bar=True, on_epoch=True, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        global step
        step += 1
        x, y = batch

        y_pred = self(x)
        loss = self.loss(y_pred, y)

        acc = torch.mean((torch.argmax(y_pred, dim=1) == y).float())
        loss_list[self.opt.__class__.__name__].append(loss.item())
        acc_list[self.opt.__class__.__name__].append(acc.item())
        log_dict = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(log_dict, on_step=True, prog_bar=True, on_epoch=True, rank_zero_only=True)
        # self.log('val_loss', loss, on_step=True, prog_bar=True, on_epoch=True)
        return log_dict

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        acc = torch.mean((torch.argmax(y_pred, dim=1) == y).float())
        log_dict = {'test_loss': loss, 'test_acc': acc}
        self.log_dict(log_dict, on_step=True, prog_bar=True, on_epoch=True, rank_zero_only=True)
        return log_dict

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     self.log('val_loss', avg_loss, on_step=True, on_epoch=True)
    #     return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return self.opt


def plot_message(data, title):
    plt.figure(figsize=(16, 9))
    for key, val in data.items():
        plt.plot(val, label=key)
    plt.legend()
    plt.title(title)
    plt.show()


def plot_test_acc_or_time(data, title):
    key = list(data.keys())
    val = list(data.values())
    plt.figure(figsize=(16, 9))

    # plt.xticks(range(len(key)),key)

    plt.xlabel("Optimizer")
    if "time" in title:
        bar = plt.bar(x=range(len(key)), height=val, color="red")
        plt.ylabel("Time(s)")
        plt.bar_label(bar, val)
    else:
        bar = plt.bar(x=range(len(key)), height=[k[0]['test_acc_epoch'] for k in val], color="red")
        plt.ylabel("Test Accuracy")
        plt.bar_label(bar, [k[0]['test_acc_epoch'] for k in val])

    plt.xticks(range(len(key)), key)
    plt.title(title)
    plt.show()


def main(args):
    # 加载数据
    print('==> Preparing data..')
    train_img, train_label, test_img, test_label = load_data(r"G:\DataSet\MNIST")
    train_loader, val_loader, test_loader = create_dataset(train_img, train_label, test_img, test_label,
                                                           args.batch_size, args.shuffle)

    # 创建模型
    print('==> Building model..')
    print("==> Training with adam optimizer")
    t1 = time.time()
    torch.cuda.empty_cache()
    model = Net(batch_norm=args.batch_norm, cnn=args.cnn, lr=args.lr, opt_index=0)
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() and args.device else 0,
                         max_epochs=args.epochs, fast_dev_run=False, precision=16,
                         auto_lr_find=True, auto_scale_batch_size=True)
    trainer.fit(model, train_loader, val_loader)
    t2 = time.time()
    time_list["Adam"] = t2 - t1
    print("==> Adam training time:", t2 - t1)
    print("==> test Adam")
    # trainer.test(test_loader)
    test_acc["Adam"] = trainer.test(model, test_loader)
    print("==> Training with sgd optimizer")
    # model.zero_grad()
    t3 = time.time()
    torch.cuda.empty_cache()
    trainer_sgd = pl.Trainer(gpus=1 if torch.cuda.is_available() and args.device else 0,
                             max_epochs=args.epochs, fast_dev_run=False, precision=16,
                             auto_lr_find=True, auto_scale_batch_size=True, callbacks={})
    model_sgd = Net(opt_index=1, batch_norm=args.batch_norm, cnn=args.cnn, lr=args.lr)
    trainer_sgd.fit(model_sgd, train_loader, val_loader)
    t4 = time.time()
    time_list["SGD"] = t4 - t3
    print("==> SGD training time:", t4 - t3)
    print("==> test SGD")
    test_acc["SGD"] = trainer_sgd.test(model_sgd, test_loader)
    print("==> Training with adamw optimizer")
    t5 = time.time()
    torch.cuda.empty_cache()
    trainer_adamw = pl.Trainer(gpus=1 if torch.cuda.is_available() and args.device else 0,
                               max_epochs=args.epochs, fast_dev_run=False, precision=16,
                               auto_lr_find=True, auto_scale_batch_size=True, callbacks={})
    model_adamw = Net(opt_index=2, batch_norm=args.batch_norm, cnn=args.cnn, lr=args.lr)
    trainer_adamw.fit(model_adamw, train_loader, val_loader)
    t6 = time.time()
    time_list["AdamW"] = t6 - t5
    print("==> AdamW training time:", t6 - t5)
    print("==> test AdamW")
    test_acc["AdamW"] = trainer_adamw.test(model_adamw, test_loader)
    if args.batch_norm:
        plot_message(loss_list, "loss in using batch normalization")
        plot_message(acc_list, "acc in using batch normalization")
        plot_test_acc_or_time(test_acc, "test acc in using batch normalization")
        plot_test_acc_or_time(time_list, "time in using batch normalization")
        plot_message(cuda_memory_list, "cuda memory in using batch normalization")
    else:
        plot_message(loss_list, "loss in not using batch normalization")
        plot_message(acc_list, "acc in not using batch normalization")
        plot_test_acc_or_time(test_acc, "test acc in not using batch normalization")
        plot_test_acc_or_time(time_list, "time in not using batch normalization")
        plot_message(cuda_memory_list, "cuda memory in not using batch normalization")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='使用的设备')
    parser.add_argument('--batch_size', type=int, default=1024, help='批量大小')
    parser.add_argument('--shuffle', type=bool, default=True, help='是否打乱数据')
    parser.add_argument('--epochs', type=int, default=5, help='训练的epoch数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch_norm', type=bool, default=True, help='是否使用BN')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--cnn', type=bool, default=True, help='是否使用CNN')
    args = parser.parse_args()
    main(args)
