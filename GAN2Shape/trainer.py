import torch
import tqdm


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
