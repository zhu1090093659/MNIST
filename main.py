import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import create_dataset, load_data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 120),
            nn.ReLU(),
            nn.Linear(120, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.model(x)
        # x = torch.argmax(x, dim=1)
        return x


def train(model, train_loader, val_loader, optimizer, criterion, device, epochs):
    model.train()
    pbar = tqdm(train_loader)
    processed = 0
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0.0
        for batch_idx, (data, target) in enumerate(pbar):
            # 将数据放到GPU上
            if torch.cuda.is_available() and device == 'cuda':
                data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            y_pred = model(data)
            # y_pred = torch.argmax(y_pred, dim=1).view(-1,1)
            loss = criterion(y_pred, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(
                desc=f'train Batch_id={batch_idx} Loss={total_loss / (batch_idx + 1):0.5f}  Accuracy={100 * correct / processed:0.2f}%')
            # print(f'Epoch: {epoch + 1} Loss: {total_loss / processed:0.5f} Accuracy: {100 * correct / processed:0.2f}')
        # 每个epoch验证一次
        validate(model, val_loader, criterion, device)


def validate(model, val_loader, criterion, device):
    # print('==> Validation..')
    model.eval()
    pbar = tqdm(val_loader)
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            if torch.cuda.is_available() and device == 'cuda':
                data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            loss = criterion(output, target)
            total_loss += loss.item()
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
            pbar.set_description(
                desc=f'validate Accuracy={100 * correct / total:0.2f}% Loss: {total_loss / len(val_loader):0.5f}')
    # print(f'Accuracy: {100 * correct / total:0.2f}')
    # print(f'Loss: {total_loss / len(val_loader):0.5f}')


def test(model, test_loader, criterion, device):
    model.eval()
    pbar = tqdm(test_loader)
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, target in pbar:
            if torch.cuda.is_available() and device == 'cuda':
                data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            loss = criterion(output, target)
            total_loss += loss.item()
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
            pbar.set_description(
                desc=f'test Accuracy={100 * correct / total:0.2f} Loss: {total_loss / len(test_loader):0.5f}')
    # print(f'Accuracy: {100 * correct / total:0.2f}')
    # print(f'Loss: {total_loss / len(test_loader):0.5f}')


def main(args):
    # 加载数据
    print('==> Preparing data..')
    train_img, train_label, test_img, test_label = load_data(r"G:\DataSet\MNIST")
    train_loader, val_loader, test_loader = create_dataset(train_img, train_label, test_img, test_label,
                                                           args.batch_size, args.shuffle)

    # 创建模型
    print('==> Building model..')
    model = Net()
    # 将模型放到GPU上
    if torch.cuda.is_available() and args.device == 'cuda':
        model.to('cuda')
    # 创建损失函数
    print('==> Building loss and optimizer..')
    criterion = nn.CrossEntropyLoss()
    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # 训练模型
    print('==> Training model..')
    train(model, train_loader, val_loader, optimizer, criterion, 'cuda', args.epochs)
    # 测试模型
    print('==> Testing model..')
    test(model, test_loader, criterion, 'cuda')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='使用的设备')
    parser.add_argument('--batch_size', type=int, default=256, help='批量大小')
    parser.add_argument('--shuffle', type=bool, default=True, help='是否打乱数据')
    parser.add_argument('--epochs', type=int, default=300, help='训练的epoch数')
    args = parser.parse_args()
    main(args)
