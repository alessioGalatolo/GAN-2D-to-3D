import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_reconstructions

class Trainer():
    def __init__(self,
                 model,
                 model_config,
                 debug=False,
                 plot_intermediate=False):
        self.model = model(model_config, debug)
        # self.n_epochs = model_config.get('n_epochs', 1)
        self.n_epochs_prior = model_config.get('n_epochs_prior', 1)
        self.learning_rate = model_config.get('learning_rate', 1e-4)
        self.refinement_iterations = model_config.get('refinement_iterations', 4)
        self.n_proj_samples = model_config.get('n_proj_samples', 1)
        self.plot_intermediate = plot_intermediate

    def fit(self, images, latents, plot_depth_map=False):
        optim = Trainer.default_optimizer(self.model, lr=self.learning_rate)

        ## Pretrain depth net on the prior shape
        # self.pretrain_on_prior(images, plot_depth_map)

        
        self.reconstructions = {'images':[None] * len(images), 'depths':[None] * len(images)}
        total_it = 0
        collected = None
        # stages = [  {'step1': 700, 'step2': 700, 'step3': 600},
        #             {'step1': 200, 'step2': 500, 'step3': 400},
        #             {'step1': 200, 'step2': 500, 'step3': 400},
        #             {'step1': 200, 'step2': 500, 'step3': 400}]
        # stages = [  {'step1': 10, 'step2': 10, 'step3': 8}]
        stages = [  {'step1': 1, 'step2': 1, 'step3': 1}]

        ## Sequential training of the D,A,L,V nets
        for stage in tqdm(range(len(stages))):
            running_loss = 0.0

            iterator = tqdm(range(len(images)))
            for i_batch in iterator:
                iterator.set_description("Stage: " + str(stage) + "/" +
                                    str(len(stages)) + ". Image: " + str(i_batch+1) + "/" +
                                    str(len(images)) + ".")
                for _ in range(self.refinement_iterations):
                    image_batch = images[i_batch].cuda()
                    latent_batch = latents[i_batch].cuda()

                    step1_iterator = tqdm(range(stages[stage]['step1']))
                    for _ in step1_iterator:
                        optim.zero_grad()
                        loss, collected_step1 = self.model.forward_step1(image_batch, latent_batch, collected)
                        loss.backward()
                        step1_iterator.set_description("Loss = " + str(loss.detach().cpu()))
                        total_it += 1
                    
                    step2_iterator = tqdm(range(stages[stage]['step2']))
                    for _ in step2_iterator:
                        optim.zero_grad()
                        loss, collected_step2 = self.model.forward_step2(image_batch, latent_batch, collected_step1)
                        loss.backward()
                        step2_iterator.set_description("Loss = " + str(loss.detach().cpu()))
                        total_it += 1

                    projected_samples, masks = collected_step2
                    step3_iterator = tqdm(range(stages[stage]['step3']))
                    for _ in step3_iterator:
                        for i_proj in range(len(projected_samples)):
                            optim.zero_grad()
                            collected = projected_samples[i_proj].unsqueeze(0), masks[i_proj].unsqueeze(0)
                            loss, _ = self.model.forward_step3(image_batch, latent_batch, collected)
                            loss.backward()
                            step3_iterator.set_description("Loss = " + str(loss.detach().cpu()))
                            total_it += 1
            
            if self.plot_intermediate:
                recon_im, recon_depth = self.model.evaluate_results(image_batch)
                plot_reconstructions(recon_im.cpu(), recon_depth.cpu(), total_it)

            # print(f'Loss: {running_loss}') # FIXME

        print('Finished Training')

    def pretrain_on_prior(self, data, plot_depth_map):
        optim = Trainer.default_optimizer(self.model.depth_net)
        train_loss = []
        print("Pretraining depth net on prior shape")
        iterator = tqdm(range(self.n_epochs_prior))
        for epoch in iterator:
            for i in range(len(data)):
                data_batch = data[i]
                inputs = data_batch.cuda()
                loss = self.model.depth_net_forward(inputs)
                optim.zero_grad()
                loss.backward()
                optim.step()
            
            if epoch % 1 == 0:
                with torch.no_grad():
                    iterator.set_description("Epoch (prior): " + str(epoch+1) + "/" + \
                        str(self.n_epochs_prior) + ". Loss = " + str(loss.cpu()))
                    train_loss.append(loss.cpu())

        if plot_depth_map:
            self.model.plot_predicted_depth_map(data)
        return train_loss

    @staticmethod
    def default_optimizer(model, lr=1e-4, betas=(0.9, 0.999), weight_decay=5e-4):
        depth_net_params = filter(lambda param: param.requires_grad,
                                  model.parameters())
        return torch.optim.Adam(depth_net_params, lr=lr,
                                betas=betas, weight_decay=weight_decay)
