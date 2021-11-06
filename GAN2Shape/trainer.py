import torch
import tqdm
from GAN2Shape.stylegan2 import StyleGAN2


class Trainer():
    def __init__(self,
                 n_epochs,
                 device='auto'):

        self.n_epochs = n_epochs
        self.device = device
        if device == 'auto':
            device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(device_name)
        self.model = StyleGAN2(10, 20)

        # TODO: init things
        self.trainloader = None
        self.vallaoder = None
        self.testloader = None
        self.loss = None

    def fit(self, optimizer):
        # loop over the dataset multiple times
        for epoch in tqdm(range(self.n_epochs)):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print('Loss: {}'.format(running_loss))

        print('Finished Training')
