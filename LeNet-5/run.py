from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import h5py
import numpy as np
import struct



def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
	with h5py.File(fname, 'r') as h5f:
		for k, v in net.state_dict().items():
			print(k)
			try:
				param = torch.from_numpy(np.asarray(h5f[k]))
			except KeyError:
				print("no value")
			else:
				v.copy_(param)




data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)
torch.multiprocessing.set_sharing_strategy('file_system')

net = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)

cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}


def train(epoch):
    global cur_batch_win

    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        loss.backward()
        optimizer.step()


def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))
    save_net('./original.weights', net)


# def train_and_test(epoch):
#     train(epoch)
#     test()


# def main():
#     for e in range(1, 16):
#         train_and_test(e)

# if __name__ == '__main__':
#     main()


load_net('./original.weights', net)
test()


for k, v in net.state_dict().items():
	print(k)
	flat_weight = v.contiguous().view(v.numel())
	param_list = []
	param_list.extend(flat_weight.tolist())
	for i in param_list:
		i = float(i)
	print(len(param_list))


	fp = open('./' + k + '.bin', 'wb')
	s = struct.pack('f'*len(param_list), *param_list)
	fp.write(s)


path = '/home/cong/Code/FCCM2020/LeNet-5/data/testSet/testSet/'
from PIL import Image

for idx in range(1, 10001):
	f = path + 'img_' + str(idx) + '.jpg'
	print(f)
	image = Image.open(f).convert('L')
	blank = Image.new('L', (32, 32), (0))
	blank.paste(image, (1, 1))

	image = np.asarray(blank)
	image = image.flatten()

	bin_file = path + 'image_bins/' + str(idx) + ".bin"
	fo = open(bin_file, "wb")
	s = struct.pack('B'*len(image), *image)
	fo.write(s)
	fo.close()

