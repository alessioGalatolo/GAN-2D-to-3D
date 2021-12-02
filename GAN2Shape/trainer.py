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
        self.learning_rate = model_config.get('learning_rate', 1e-4)
        self.refinement_iterations = model_config.get('refinement_iterations', 4)

    def fit(self, images, latents, plot_depth_map=False):
        optim = Trainer.default_optimizer(self.model, lr=self.learning_rate)
        self.pretrain_on_prior(images, plot_depth_map)
        collected = None
        # loop over the dataset multiple times
        for epoch in tqdm(range(self.n_epochs)):  # FIXME: not sure if what they call epochs are actually epochs
            running_loss = 0.0
            for image_batch, latent_batch in zip(images, latents):
                for _ in range(self.refinement_iterations):
                    image_batch = image_batch.cuda()
                    latent_batch = latent_batch.cuda()

                    for num in range(1, 3):  # Perform each training step
                        optim.zero_grad()
                        loss, collected = getattr(self.model, f'forward_step{num}')(image_batch, latent_batch, collected)
                        loss.backward()
                        # running_loss += m #FIXME

            print(f'Loss: {running_loss}')

        print('Finished Training')

    def pretrain_on_prior(self, data, plot_depth_map):
        optim = Trainer.default_optimizer(self.model.depth_net)
        train_loss = []
        print("Pretraining depth net on prior shape")
        iterator = tqdm(range(len(data)))
        for i in iterator:
            data_batch = data[i]
            inputs = data_batch.cuda()
            loss = self.model.depth_net_forward(inputs)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if i % 10 == 0:
                with torch.no_grad():
                    iterator.set_description("Loss = " + str(loss.cpu()))
                    train_loss.append(loss.cpu())
            # scheduler.step()
        # plt.plot(train_loss)
        # plt.title("Pretrain prior - loss / 10 steps")
        # plt.show()
        if plot_depth_map:
            self.model.plot_predicted_depth_map(data)
        return train_loss

    @staticmethod
    def default_optimizer(model, lr=1e-4, betas=(0.9, 0.999), weight_decay=5e-4):
        depth_net_params = filter(lambda param: param.requires_grad,
                                  model.parameters())
        return torch.optim.Adam(depth_net_params, lr=lr,
                                betas=betas, weight_decay=weight_decay)

    # FIXME: not sure what these are for
    # def fit_step1(self, data):
    #     pass

    # def fit_step2(self, data):
    #     pass

    # def fit_step3(self, data):
        # pass
