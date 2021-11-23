import torch
from tqdm import tqdm


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
