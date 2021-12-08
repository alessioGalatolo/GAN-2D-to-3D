import torch
from tqdm import tqdm
from random import shuffle
from plotting import plot_reconstructions
import wandb
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self,
                 model,
                 model_config,
                 debug=False,
                 plot_intermediate=False,
                 log_wandb=False):
        self.model = model(model_config, debug)
        # self.n_epochs = model_config.get('n_epochs', 1)
        self.n_epochs_prior = model_config.get('n_epochs_prior', 1)
        self.learning_rate = model_config.get('learning_rate', 1e-4)
        self.refinement_iterations = model_config.get('refinement_iterations', 4)
        self.plot_intermediate = plot_intermediate
        self.log_wandb = log_wandb
        self.debug=debug

    def fit(self, images, latents, plot_depth_map=False):
        self.model.reinitialize_model()
        optim = Trainer.default_optimizer(self.model, lr=self.learning_rate)

        # # Pretrain depth net on the prior shape
        # self.pretrain_on_prior(images, plot_depth_map)

        self.reconstructions = {'images': [None] * len(images), 'depths': [None] * len(images)}
        total_it = 0
        stages = [{'step1': 700, 'step2': 700, 'step3': 600},
                  {'step1': 200, 'step2': 500, 'step3': 400},
                  {'step1': 200, 'step2': 500, 'step3': 400},
                  {'step1': 200, 'step2': 500, 'step3': 400}]
        # stages = [{'step1': 70, 'step2': 70, 'step3': 60},
        #           {'step1': 20, 'step2': 50, 'step3': 40},
        #           {'step1': 20, 'step2': 50, 'step3': 40},
        #           {'step1': 20, 'step2': 50, 'step3': 40}]
        # stages = [  {'step1': 7, 'step2': 7, 'step3': 6},
        #             {'step1': 2, 'step2': 5, 'step3': 4}]
        # # stages = [{'step1': 1, 'step2': 1, 'step3': 1}]
        # stages = [  {'step1': 1, 'step2': 1, 'step3': 1},
        #             {'step1': 1, 'step2': 1, 'step3': 1}]

        # array to keep the same shuffling among images, latents, etc.
        shuffle_ids = [i for i in range(len(images))]
        # Sequential training of the D,A,L,V nets
        for stage in tqdm(range(len(stages))):
            running_loss = 0.0

            old_collected = [None]*len(images)
            for step in [1, 2]:  # step 1, 2
                step_iterator = tqdm(range(stages[stage][f'step{step}']))
                current_collected = [None]*len(images)
                for _ in step_iterator:
                    shuffle(shuffle_ids)
                    iterator = tqdm(shuffle_ids)
                    for i_batch in iterator:
                        iterator.set_description("Stage: " + str(stage) + "/"
                                                 + str(len(stages)) + ". Image: "
                                                 + str(i_batch+1) + "/"
                                                 + str(len(images)) + ".")
                        image_batch = images[i_batch].cuda()
                        latent_batch = latents[i_batch].cuda()
                        optim.zero_grad()
                        collected = old_collected[i_batch]
                        loss, collected = getattr(self.model, f'forward_step{step}') \
                            (image_batch, latent_batch, collected)
                        # We want to make sure we keep track of the index corresponding to the original image
                        # hence this change
                        current_collected[i_batch]=collected
                        loss.backward()
                        optim.step()
                        step_iterator.set_description("Loss = " + str(loss.detach().cpu()))
                        total_it += 1

                        # #the albedo net parameters are not changing!!?? 
                        # edit: we are actual idiots <3
                        # if self.debug:
                        #     if step==1:
                        #         paramsum=0
                        #         for param in self.model.albedo_net.named_parameters():
                        #             param = param[1]
                        #             s = torch.sum(param)
                        #             paramsum+=torch.sum(param)
                        #     print(f"Albedo param sum = {paramsum:.100}\n")

                        if self.log_wandb:
                            wandb.log({"stage": stage,
                                       "total_it": total_it,
                                       f"loss_step{step}": loss})
                old_collected = current_collected

            # step 3
            step_iterator = tqdm(range(stages[stage]['step3']))
            current_collected = []
            for _ in step_iterator:
                current_collected = [None]*len(images)
                shuffle(shuffle_ids)
                iterator = tqdm(shuffle_ids)
                for i_batch in iterator:
                    iterator.set_description("Stage: " + str(stage) + "/"
                                             + str(len(stages)) + ". Image: "
                                             + str(i_batch+1) + "/"
                                             + str(len(images)) + ".")
                    image_batch = images[i_batch].cuda()
                    latent_batch = latents[i_batch].cuda()
                    projected_samples, masks = old_collected[i_batch] 
                    shuffle_projected = [i for i in range(len(projected_samples))]
                    shuffle(shuffle_projected)
                    for i_proj in shuffle_projected:
                        optim.zero_grad()
                        collected = projected_samples[i_proj].unsqueeze(0).cuda(), masks[i_proj].unsqueeze(0).cuda()
                        
                        # #we can delete this later (of course)
                        # if self.debug:
                        #     im = image_batch[0].cpu().transpose(0,2).transpose(0,1)
                        #     proj_im = projected_samples[i_proj].cpu().transpose(0,2).transpose(0,1)
                        #     plt.imshow(im)
                        #     plt.show()
                        #     breakpoint = True
                        #     plt.imshow(proj_im)
                        #     plt.show()
                        #     breakpoint = True

                        loss, _ = self.model.forward_step3(image_batch, latent_batch, collected)
                        loss.backward()
                        optim.step()
                        step_iterator.set_description("Loss = " + str(loss.detach().cpu()))
                        total_it += 1

                    if self.log_wandb:
                        wandb.log({"stage": stage,
                                   "total_it": total_it,
                                   "loss_step3": loss})

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
                    iterator.set_description("Epoch (prior): " + str(epoch+1) + "/"
                                             + str(self.n_epochs_prior)
                                             + ". Loss = " + str(loss.cpu()))
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
