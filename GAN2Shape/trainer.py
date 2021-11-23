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
                data_batch = data_batch.to(self.device)
                m = self.model.forward(data_batch)


                self.model.backward()

                # running_loss += m

            print(f'Loss: {running_loss}')

        
        self.fit_step1()
        self.fit_step2()
        self.fit_step3()
        print('Finished Training')

    def fit_step1(self,data):
        pass
    def fit_step2(self,data):
        pass
    def fit_step3(self,data):
        pass
    
    def pretrain_on_prior(self, data, plot_example=None):
        prior = self.model.prior.to(self.device)
        self.model.init_optimizers()
        optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.depth_net.parameters()),
            lr=0.0001, betas=(0.9, 0.999), weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim,T_0=10,eta_min=0.0001)
        train_loss = []
        print("Pretraining depth net on prior shape")
        iterator = tqdm(range(10))
        for j in iterator:
            for i in range(len(data)):
                data_batch=data[i]
                inputs = data_batch.to(self.device)
                depth_raw = self.model.depth_net(inputs)
                depth_centered = depth_raw - depth_raw.view(1,1,-1).mean(2).view(1,1,1,1)
                depth = torch.tanh(depth_centered).squeeze(0)
                depth = self.model.depth_rescaler(depth)
                loss = F.mse_loss(depth,prior.detach())
                optim.zero_grad()
                loss.backward()
                optim.step()
                if j % 10 == 0:
                    with torch.no_grad():
                        iterator.set_description("Loss = " + str(loss.cpu()))
                        train_loss.append(loss.cpu())
            # scheduler.step()
        # plt.plot(train_loss)
        # plt.title("Pretrain prior - loss / 10 steps")
        # plt.show()
        if plot_example is not None:
            with torch.no_grad():
                self.model.plot_predicted_depth_map(data, self.device, img_idx=plot_example)
        return train_loss