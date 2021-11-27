import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Trainer():
    def __init__(self,
                 model,
                 model_config):
        self.model = model(model_config)
        self.n_epochs = model_config.get('n_epochs', 1)

        # TODO: init things
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.loss = None

    def fit(self, data):
        # loop over the dataset multiple times
        for epoch in tqdm(range(self.n_epochs)):
            self.model.pretrain_depth_net(data)
            running_loss = 0.0
            for i in range(len(data)):
                data_batch = data[i]
                data_batch = data_batch.cuda()
                m = self.model.forward(data_batch)

                self.model.backward()

                # running_loss += m

            print(f'Loss: {running_loss}')

        # FIXME: not sure what these are for
        # self.fit_step1()
        # self.fit_step2()
        # self.fit_step3()
        print('Finished Training')

    # FIXME: not sure what these are for
    # def fit_step1(self, data):
    #     pass

    # def fit_step2(self, data):
    #     pass

    # def fit_step3(self, data):
        # pass
