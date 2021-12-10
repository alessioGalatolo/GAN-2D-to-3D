from logging import debug
import math
import torch
from tqdm import tqdm
from random import shuffle
from plotting import plot_reconstructions
import wandb
import matplotlib.pyplot as plt
from gan2shape import utils


class Trainer():
    def __init__(self,
                 model,
                 model_config,
                 debug=False,
                 plot_intermediate=False,
                 log_wandb=False,
                 save_ckpts=False):
        self.model = model(model_config, debug)
        self.image_size = model_config.get('image_size')
        self.n_epochs_prior = model_config.get('n_epochs_prior', 1000)
        self.learning_rate = model_config.get('learning_rate', 1e-4)
        self.plot_intermediate = plot_intermediate
        self.log_wandb = log_wandb
        self.save_ckpts = save_ckpts
        self.debug = debug
        self.prior_name =  model_config.get('prior_name', "box")

    def fit(self, images, latents, plot_depth_map=False):
        # self.model.reinitialize_model()
        self.optim_step1 = Trainer.default_optimizer([self.model.albedo_net], lr=self.learning_rate)
        self.optim_step2 = Trainer.default_optimizer([self.model.offset_encoder_net], lr=self.learning_rate)
        self.optim_step3 = Trainer.default_optimizer([  self.model.lighting_net,
                                                        self.model.viewpoint_net,
                                                        self.model.depth_net,
                                                        self.model.albedo_net], lr=self.learning_rate)

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
        # stages = [  {'step1': 1, 'step2': 1, 'step3': 1},
        #             {'step1': 1, 'step2': 1, 'step3': 1}]
        self.n_stages = len(stages)
        # array to keep the same shuffling among images, latents, etc.
        shuffle_ids = [i for i in range(len(images))]
        shuffle_ids = [0]
        # Sequential training of the D,A,L,V nets

        #-----------------Main loop through all images--------------------------
        iterator = tqdm(shuffle_ids)
        for i_batch in shuffle_ids:
            print(f'Training on image {i_batch}/{len(shuffle_ids)}')
            image_batch = images[i_batch].cuda()
            latent_batch = latents[i_batch].cuda()

            # Pretrain depth net on the prior shape
            self.pretrain_on_prior(image_batch, i_batch, plot_depth_map)

            #-----------------Loop through all stages---------------------------
            for stage in range(self.n_stages):
                iterator.set_description("Stage: " + str(stage) + "/"
                                         + str(self.n_stages) + ". Image: "
                                         + str(i_batch+1) + "/"
                                         + str(len(images)) + ".")
                running_loss = 0.0
                old_collected = [None]*len(images)

                #-----------------------Step 1 and 2----------------------------
                for step in [1, 2]:
                    print(f"Doing step {step}, stage {stage + 1}/{self.n_stages}")
                    step_iterator = tqdm(range(stages[stage][f'step{step}']))
                    current_collected = [None]*len(images)
                    optim = getattr(self,f'optim_step{step}')
                    for _ in step_iterator:
                        optim.zero_grad()
                        collected = old_collected[i_batch]

                        loss, collected = getattr(self.model, f'forward_step{step}')\
                            (image_batch, latent_batch, collected)
                        
                        current_collected[i_batch] = collected
                        loss.backward()
                        optim.step()
                        step_iterator.set_description("Loss = " + str(loss.detach().cpu()))
                        total_it += 1

                        if self.log_wandb:
                            wandb.log({"stage": stage,
                                       "total_it": total_it,
                                       f"loss_step{step}": loss,
                                       "image_num": i_batch})
                    old_collected = current_collected

                #-----------------------------Step 3----------------------------
                print(f"Doing step 3, stage {stage + 1}/{self.n_stages}")
                step_iterator = tqdm(range(stages[stage]['step3']))
                current_collected = []
                optim = self.optim_step3
                for _ in step_iterator:
                    current_collected = [None]*len(images)
                    projected_samples, masks = old_collected[i_batch]
                    shuffle_projected = [i for i in range(len(projected_samples))]
                    shuffle(shuffle_projected)
                    for i_proj in shuffle_projected:
                        optim.zero_grad()
                        collected = projected_samples[i_proj].unsqueeze(0).cuda(), masks[i_proj].unsqueeze(0).cuda()

                        loss, _ = self.model.forward_step3(image_batch, latent_batch, collected)
                        loss.backward()
                        optim.step()
                        step_iterator.set_description("Loss = " + str(loss.detach().cpu()))
                        total_it += 1

                    if self.log_wandb:
                        wandb.log({"stage": stage,
                                   "total_it": total_it,
                                   "loss_step3": loss,
                                   "image_num": i_batch})

                if self.plot_intermediate:
                    if i_batch % 3 == 0:
                        recon_im, recon_depth = self.model.evaluate_results(image_batch)
                        recon_im, recon_depth = recon_im.cpu(), recon_depth.cpu()
                        plot_reconstructions(recon_im, recon_depth,
                                total_it=str(total_it), im_idx=str(i_batch), stage=str(stage))
                
                if self.save_ckpts:
                    self.model.save_checkpoint(stage, total_it)


            # print(f'Loss: {running_loss}') # FIXME
            # maybe WandB logging is enough?

        print('Finished Training')

    def pretrain_on_prior(self, image, i_batch, plot_depth_map):
        optim = Trainer.default_optimizer([self.model.depth_net])
        train_loss = []
        print("Pretraining depth net on prior shape")
        prior = self.prior_shape(image, shape=self.prior_name)

        iterator = tqdm(range(self.n_epochs_prior))
        for epoch in iterator:
            inputs = image.cuda()
            loss = self.model.depth_net_forward(inputs, prior)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if epoch % self.n_epochs_prior / 10 == 0:
                with torch.no_grad():
                    iterator.set_description("Epoch (prior): " + str(epoch+1) + "/"
                                            + str(self.n_epochs_prior)
                                            + ". Loss = " + str(loss.cpu()))
                    
            if self.log_wandb:
                wandb.log({"loss_prior":loss.cpu(),
                            "image_num": i_batch})

        if plot_depth_map:
            self.model.plot_predicted_depth_map(image)
        return train_loss

    def prior_shape(self, image, shape="box"):
        with torch.no_grad():
            height, width = self.image_size, self.image_size
            center_x, center_y = int(width / 2), int(height / 2)
            if shape == "box":
                box_height, box_width = int(height*0.5*0.5), int(width*0.8*0.5)
                prior = torch.zeros([1, height, width])
                prior[0,
                      center_y-box_height: center_y+box_height,
                      center_x-box_width: center_x+box_width] = 1
                return prior.cuda()
            elif shape == "ellipsoid":
                h, w = self.image_size, self.image_size
                c_x, c_y = w / 2, h / 2

                ## FIXME: utils.get_mask_range(mask) throws error
                # see Trainer.image_mask() for more details

                # mask = self.image_mask(image)[0, 0] >= 0.7
                # max_y, min_y, max_x, min_x = utils.get_mask_range(mask)

                #Temporary "fix"
                box_height, box_width = int(h*0.5*0.5), int(w*0.8*0.5)
                max_x, min_x = int(c_x+box_width), int(c_x-box_width)
                max_y, min_y = int(c_y+box_height), int(c_y-box_height)

                # if self.category in ['car', 'church']:
                #     max_y = max_y + (max_y - min_y) / 6

                r_pixel = (max_x - min_x) / 2
                ratio = (max_y - min_y) / (max_x - min_x)
                c_x = (max_x + min_x) / 2
                c_y = (max_y + min_y) / 2
                radius = 0.4
                near = 0.91
                far = 1.02

                ellipsoid = torch.Tensor(1, h, w).fill_(far)
                i, j = torch.meshgrid(torch.linspace(0, w-1, w),
                                      torch.linspace(0, h-1, h))
                i = (i - h/2) / ratio + h/2
                temp = math.sqrt(radius**2 - (radius - (far - near))**2)
                dist = torch.sqrt((i - c_y)**2 + (j - c_x)**2)
                area = dist <= r_pixel
                dist_rescale = dist / r_pixel * temp
                depth = radius - torch.sqrt(torch.abs(radius ** 2 - dist_rescale ** 2)) + near
                ellipsoid[0, area] = depth[area]
                return ellipsoid.cuda()
            else:
                return torch.ones([1, height, width])

    def image_mask(self, image):
        with torch.no_grad():
            size = 473
            image = utils.resize(image, [size, size])
            image = image / 2 + 0.5
            image[:, 0].sub_(0.485).div_(0.229)
            image[:, 1].sub_(0.456).div_(0.224)
            image[:, 2].sub_(0.406).div_(0.225)
            out = self.model.mask_net(image)
            out = out.argmax(dim=1, keepdim=True)
            # FIXME: out has no elems == 7, I think something may
            # be wrong with the mask_net 
            # This results in an error in Trainer.prior_shape() for the ellipsoid
            # because the mask is only False
            # => which gives an error in utils.get_mask_range(mask)
            mask = (out == 7).float()  # FIXME: only cars
        return utils.resize(mask, [self.image_size, self.image_size])

    @staticmethod
    def default_optimizer(model_list, lr=1e-4, betas=(0.9, 0.999), weight_decay=5e-4):
        param_list = []
        for model in model_list:
            params = filter(lambda param: param.requires_grad,
                                  model.parameters())
            param_list+=list(params)
        return torch.optim.Adam(param_list, lr=lr,
                                betas=betas, weight_decay=weight_decay)
