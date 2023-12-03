import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import numpy as np
import time
import copy
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import rotate as scipyrotate
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import Dataset
import os





def save_images_per_class(N, dataloader, class_names, name):

    classes = list(class_names.values())
    class_indices = {cls: [] for cls in classes}

    # Iterate over data to get indices for each class
    for idx, (image, label) in enumerate(dataloader.dataset):
        # check if label is a tensor or not
        if torch.is_tensor(label):
            label = label.item()
        class_label = class_names[label]
        class_indices[class_label].append((image, label))

        # Only keep the first N indices for each class
        if len(class_indices[class_label]) > N:
            class_indices[class_label] = class_indices[class_label][:N]

    # Plotting
    fig, axes = plt.subplots(N, len(classes), figsize=(len(classes) * 2, N * 2))

    for i, cls in enumerate(classes):
        for j, (image, label) in enumerate(class_indices[cls]):
            image = image * 0.5 + 0.5

          
            if image.shape[0] in [1, 3]:  # assuming channel-first format
                image = np.transpose(image.numpy(), (1, 2, 0))

            axes[j, i].imshow(image)
            axes[j, i].set_title(cls)
            axes[j, i].axis('off')

    cwd = os.getcwd()
    save_path = os.path.join(cwd, name + '.png')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig) 



def main():
    # Define your network architecture (MLP)
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size*4)
            self.fc2 = nn.Linear(hidden_size*4, hidden_size*2)
            self.fc3 = nn.Linear(hidden_size*2, hidden_size)
            self.fc4 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten input
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            y = F.relu(self.fc3(x))
            z = self.fc4(y)
            return z
        
        def feature(self, x):
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            y = F.relu(self.fc3(x))
            return y


    batch_real = 256
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    condense_iterations= 20000             #Authors consider default 20000
    batch_real = 128
    lr_img = 1e-1  # Authors consider default 1.0
    train_data_dir = 'Training'

    transform = transforms.Compose([
                                    transforms.Resize((224, 224)),  
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    #  transforms.Normalize(mean=mean, std=std)
                                    ])

    dst_train = datasets.ImageFolder(root=train_data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(dst_train, batch_size=128, shuffle=True)

    class_names= {train_loader.dataset.class_to_idx[key]: key for key in train_loader.dataset.class_to_idx.keys()}

    save_images_per_class(50, train_loader,class_names, 'real_data')


    test_data_dir = 'Testing'
    dst_test = datasets.ImageFolder(root=test_data_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(dst_test, batch_size=128, shuffle=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sample_img=next(iter(train_loader))[0][0]
    im_size = (sample_img.shape[1], sample_img.shape[2])
    channel = 3
    num_classes =max(dst_train.targets)+1
    epochs = 40
        #------------------Train the Net--------------------------------


    net= MLP(input_size=channel * im_size[0] * im_size[1], hidden_size=128, output_size=num_classes).to(device)


    # check if 'trained_net.pth' exists, if not, train the net
    if os.path.isfile('trained_net.pth'):
        print('\nFound trained feature extractor')
        net.load_state_dict(torch.load('trained_net.pth'))

    else:
        print('\nTraining a feature extractor..')
        criterion = nn.CrossEntropyLoss()
        optimizer=torch.optim.Adam(net.parameters(), lr=1e-3)

        # Teacher training before starting the dataset distillation process
        for epochy in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                output = net(data) 
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        del loss
        del output
        del optimizer

        #savenet
        torch.save(net.state_dict(), 'trained_net.pth')



    # make all parameters non-trainable, so as to make image_syn the only trainable parameter
    for param in list(net.parameters()):
        param.requires_grad = False
    #---------------------------------------------------------------------------
    net.eval()

    net.to(device)
    #----------------------------------------------------------------------

    
    def test(model, data_loader, device):
        model.to(device)
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy
        

    class TensorDataset(Dataset):
        def __init__(self, images, labels): # images: n x c x h x w tensor
            self.images = images.detach().float()
            self.labels = labels.detach()

        def __getitem__(self, index):
            return self.images[index], self.labels[index]

        def __len__(self):
            return self.images.shape[0]
        


    def epoch(mode, dataloader, net, optimizer, criterion):
        net = net.to(device)
        criterion = criterion.to(device)

        if mode == 'train':
            net.train()
        else:
            net.eval()

        for i_batch, datum in enumerate(dataloader):
            img = datum[0].float().to(device)
            lab = datum[1].long().to(device)

            output = net(img)
            loss = criterion(output, lab)
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



    running_time=[]
    test_accs=[]
    test_accs2=[]
    ipc_record=[]

    ipc_trys=[10]

    for exp in range(len(ipc_trys)):
        print('\n================== Condensed Images Per Class: %d ==================\n '%ipc_trys[exp])

        ipc=ipc_trys[exp]
        starting_time = time.time()

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)


        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

       
        ''' initialize the synthetic data from random noise '''
        image_syn = torch.randn(size=(num_classes*ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device)
        label_syn = torch.tensor([np.ones(ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        
        # ''' copy the real data to synthetic data for initialization'''
        for c in range(num_classes):
            image_syn.data[c*ipc:(c+1)*ipc] = get_images(c, ipc).detach().data


        ''' training '''
        optimizer_img = torch.optim.Adam([image_syn, ], lr=lr_img) # optimizer_img for synthetic data
        optimizer_img.zero_grad()



        #---Starting the condensation process
        for it in range(condense_iterations):
            # Train Synthetic Data

            loss_syn = torch.tensor(0.0).to(device)
            for c in range(num_classes):
                img_real = get_images(c, batch_real)
                img_real=img_real.to(device)
                img_syn = image_syn[c * ipc: (c + 1) * ipc].reshape((ipc, channel, im_size[0], im_size[1]))



                output_real = net.feature(img_real).detach()
                output_syn = net.feature(img_syn)

                loss_syn+= torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)



            optimizer_img.zero_grad()
            loss_syn.backward()
            optimizer_img.step()

        ending_time = time.time()
        running_time.append(ending_time - starting_time)
        ipc_record.append(ipc)

        image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())  # avoid any unaware modification
        dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
        trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=32, shuffle=True, num_workers=0)

        save_images_per_class(ipc, trainloader, class_names, 'syn_data')

        testing_net=MLP(input_size=channel * im_size[0] * im_size[1], hidden_size=128, output_size=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optim_testing_net=torch.optim.Adam(testing_net.parameters(), lr=1e-3)
        for _ in range(30):
            for batch in trainloader:
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                optim_testing_net.zero_grad()
                output = testing_net(imgs)
                loss_test = criterion(output, labels)
                optim_testing_net.zero_grad()
                loss_test.backward()
                optim_testing_net.step()


        test_acc=test(testing_net, train_loader, device)
        test_acc2=test(testing_net, test_loader, device)
        test_accs.append(test_acc)
        test_accs2.append(test_acc2)


    stat_data = {
        'Running Time': running_time,
        'Accuracy over Real Training': test_accs,
        'Accuracy over Real Testing': test_accs2,
        'IPC': ipc_record
    }

    df = pd.DataFrame(stat_data)


    df.to_csv('DM_without_DSA.csv', index=False)




if __name__ == '__main__':
    main()