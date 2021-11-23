import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self,
                 model,
                 model_config,
                 device='auto'):

        self.device = device
        if device == 'auto':
            device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(device_name)
        self.model = model(model_config)
        self.n_epochs = model_config.get('n_epochs', 1)

        # TODO: init things
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.loss = None

    def fit(self, data):
        self.model.init_optimizers()
        # loop over the dataset multiple times
        for epoch in tqdm(range(self.n_epochs)):
            running_loss = 0.0
            for i in range(len(data)):
                data_batch=data[i]
                inputs = data_batch.to(self.device)

                m = self.model.forward(inputs)
                self.model.backward()

                # running_loss += m

            print(f'Loss: {running_loss}')

        print('Finished Training')
    
    def pretrain_on_prior(self, data):
        prior = self.model.prior.to(self.device)
        self.model.init_optimizers()
        optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.depth_net.parameters()),
            lr=0.0001, betas=(0.9, 0.999), weight_decay=5e-4)

        train_loss = []
        print("Pretraining depth net on prior shape")
        iterator = tqdm(range(1000))
        for j in iterator:
            for i in range(len(data)):
                data_batch=data[i]
                inputs = data_batch.to(self.device)
                depth_raw = self.model.depth_net(inputs)
                depth_raw = depth_raw.squeeze(0)
                depth_centered = depth_raw - depth_raw.view(1,1,-1).mean(2).view(1,1,1,1)
                depth = torch.tanh(depth_centered)
                loss = F.mse_loss(depth,prior)
                optim.zero_grad()
                loss.backward()
                optim.step()
                if j % 10 == 0:
                    with torch.no_grad():
                        iterator.set_description("Loss = " + str(loss.cpu()))
                        train_loss.append(loss.cpu())
        plt.plot(train_loss)
        plt.show()
        return train_loss