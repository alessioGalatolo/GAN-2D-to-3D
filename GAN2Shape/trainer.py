import math
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from plotting import plot_predicted_depth_map, plot_reconstructions
from gan2shape import utils
try:
    import wandb
except ImportError:
    wandb = None


class Trainer():
    CATEGORIES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                  'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                  'horse', 'motorbike', 'person', 'pottedplant',
                  'sheep', 'sofa', 'train', 'tvmonitor']
    CATEGORY2NUMBER = {category: i+1 for i, category in enumerate(CATEGORIES)}

    def __init__(self,
                 model,
                 model_config,
                 debug=False,
                 plot_intermediate=False,
                 log_wandb=False,
                 save_ckpts=False,
                 load_dict=None):
        self.model = model(model_config, debug)
        self.image_size = model_config.get('image_size')
        self.category = model_config.get('category')
        self.n_proj_samples = model_config.get('n_proj_samples', 8)
        self.n_epochs_prior = model_config.get('n_epochs_prior', 1000)
        self.n_workers = model_config.get('n_workers', 0)
        self.learning_rate = model_config.get('learning_rate', 1e-4)
        self.prior_name = model_config.get('prior_name', "box")
        self.plot_intermediate = plot_intermediate
        self.log_wandb = log_wandb
        self.save_ckpts = save_ckpts
        self.debug = debug
        if load_dict is not None:
            self.load_model_checkpoint(load_dict)
        self.optim_step1 = Trainer.default_optimizer([self.model.albedo_net],
                                                     lr=self.learning_rate)
        self.optim_step2 = Trainer.default_optimizer([self.model.offset_encoder_net],
                                                     lr=self.learning_rate)
        self.optim_step3 = Trainer.default_optimizer([self.model.lighting_net,
                                                      self.model.viewpoint_net,
                                                      self.model.depth_net,
                                                      self.model.albedo_net],
                                                     lr=self.learning_rate)

    def fit(self, images_latents, plot_depth_map=False, load_dict=None,
            stages=[{'step1': 1, 'step2': 1, 'step3': 1}]*2,
            shuffle=False):

        # continue previously started training
        if load_dict is not None:
            self.load_model_checkpoint(load_dict)

        total_it = 0
        n_stages = len(stages)

        # the original training is instance-based => batch size = 1
        dataloader = DataLoader(images_latents,
                                batch_size=1,
                                shuffle=shuffle,
                                num_workers=self.n_workers)
        # Sequential training of the D,A,L,V nets

        # -----------------Main loop through all images------------------------
        data_iterator = tqdm(dataloader)
        for batch in data_iterator:
            image, latent, data_index = batch
            image, latent, data_index = image.cuda(), latent.cuda(), data_index[0]
            logging.info(f'Training on image {data_index}/{len(data_iterator)}')

            if not self.debug:
                # Pretrain depth net on the prior shape
                self.pretrain_on_prior(image, data_index, plot_depth_map)

            # -----------------Loop through all stages-------------------------
            for stage in range(n_stages):
                # store the results of previous step (i.e. pseudo imgs, etc.)
                old_collected = [None]*len(images_latents)

                # -----------------------Step 1, 2 and 3-----------------------
                for step in [1, 2, 3]:
                    if self.debug:
                        logging.info(f"Doing step {step}, stage {stage + 1}/{n_stages}")
                    data_iterator.set_description(f"Stage: {stage}/{n_stages}. "
                                                  + f"Image: {data_index+1}/{len(images_latents)}."
                                                  + f"Step: {step}.")
                    current_collected = [None]*len(images_latents)
                    optim = getattr(self, f'optim_step{step}')
                    for _ in tqdm(range(stages[stage][f'step{step}'])):
                        optim.zero_grad()
                        collected = old_collected[data_index]

                        loss, collected = getattr(self.model, f'forward_step{step}')\
                            (image, latent, collected, n_proj_samples=self.n_proj_samples)

                        current_collected[data_index] = collected
                        loss.backward()
                        optim.step()
                        total_it += 1

                        if self.log_wandb:
                            wandb.log({"stage": stage,
                                       "total_it": total_it,
                                       f"loss_step{step}": loss,
                                       "image_num": data_index})
                    old_collected = current_collected

            if self.plot_intermediate:
                recon_im, recon_depth = self.model.evaluate_results(image)
                recon_im, recon_depth = recon_im.cpu(), recon_depth.cpu()
                plot_reconstructions(recon_im, recon_depth,
                                     total_it=str(total_it),
                                     im_idx=str(data_index.item()),
                                     stage=str(stage))

            if self.save_ckpts:
                self.model.save_checkpoint(data_index, stage, total_it, self.category)
        logging.info('Finished Training')

    def pretrain_on_prior(self, image, i_batch, plot_depth_map):
        optim = Trainer.default_optimizer([self.model.depth_net])
        train_loss = []
        logging.info("Pretraining depth net on prior shape")
        prior = self.prior_shape(image, shape=self.prior_name)

        if plot_depth_map:
            plt_prior = prior.unsqueeze(0).detach().cpu().numpy()
            plot_predicted_depth_map(plt_prior, self.image_size,
                                     block=False, save=True,
                                     img_idx=i_batch.item(),
                                     filename="prior")

        iterator = tqdm(range(self.n_epochs_prior))
        for _ in iterator:
            inputs = image.cuda()
            optim.zero_grad()
            loss, depth = self.model.depth_net_forward(inputs, prior)
            loss.backward()
            optim.step()

            with torch.no_grad():
                iterator.set_description(f"Depth net prior loss = {loss.cpu()}")

            if self.log_wandb:
                wandb.log({"loss_prior": loss.cpu(),
                           "image_num": i_batch})

        if plot_depth_map:
            depth = depth.detach().cpu().numpy()
            plot_predicted_depth_map(depth, self.image_size, block=True)
        return train_loss

    def prior_shape(self, image, shape="box"):
        #FIXME: should probably move to its own file since its getting long
        with torch.no_grad():
            height, width = self.image_size, self.image_size
            center_x, center_y = int(width / 2), int(height / 2)
            near = 0.91
            far = 1.02
            noise_treshold = 0.7
            prior = torch.Tensor(1, height, width).fill_(far)
            if shape == "box":
                box_height, box_width = int(height*0.5*0.5), int(width*0.8*0.5)
                prior = torch.zeros([1, height, width])
                prior[0,
                      center_x-box_width: center_x+box_width,
                      center_y-box_height: center_y+box_height] = 1
                prior = prior
                return prior.cuda()
            elif shape == "masked_box":
                # same as box but only project object
                box_height, box_width = int(height*0.5*0.5), int(width*0.8*0.5)
                mask = self.image_mask(image)[0].cpu()

                # cut noise in mask
                noise = mask < noise_treshold
                mask[noise] = 0
                mask = (mask - noise_treshold) / (1 - noise_treshold)

                prior = far - prior * mask
                return prior.cuda()

            elif shape == "smoothed_box":
                # Smoothed masked_box
                box_height, box_width = int(height*0.5*0.5), int(width*0.8*0.5)
                mask = self.image_mask(image)[0].cpu()

                # cut noise in mask
                noise = mask < noise_treshold
                mask[noise] = 0
                mask = (mask - noise_treshold) / (1 - noise_treshold)

                prior = far - prior * mask
                
                #Smoothing through repeated convolution
                kernel_size = 11
                pad = 5
                n_convs = 3
                conv = torch.nn.Conv2d(in_channels=1,out_channels=1, kernel_size=kernel_size, stride=1, padding=0)
                filt = torch.ones(1,1, kernel_size, kernel_size)
                filt = filt / torch.norm(filt)
                conv.weight = torch.nn.Parameter(filt)
                prior = prior.unsqueeze(0)
                for i in range(n_convs):
                    prior = conv(prior)
                    # Rescale depth values to appropriate range
                    prior = near + ((prior - torch.min(prior))*(far - near)) / (torch.max(prior) - torch.min(prior))
                    # Pad result with 'far' to keep the image size
                    prior = torch.nn.functional.pad(prior,tuple([pad]*4), value=far)
                        
                return prior.squeeze(0).cuda()

            elif shape == "ellipsoid":
                radius = 0.4
                mask = self.image_mask(image)[0, 0] >= noise_treshold
                max_y, min_y, max_x, min_x = utils.get_mask_range(mask)

                # if self.category in ['car', 'church']:
                #     max_y = max_y + (max_y - min_y) / 6

                r_pixel = (max_x - min_x) / 2
                ratio = (max_y - min_y) / (max_x - min_x)
                c_x = (max_x + min_x) / 2
                c_y = (max_y + min_y) / 2

                i, j = torch.meshgrid(torch.linspace(0, width-1, width),
                                      torch.linspace(0, height-1, height))
                i = (i - height/2) / ratio + height/2
                temp = math.sqrt(radius**2 - (radius - (far - near))**2)
                dist = torch.sqrt((i - c_y)**2 + (j - c_x)**2)
                area = dist <= r_pixel
                dist_rescale = dist / r_pixel * temp
                depth = radius - torch.sqrt(torch.abs(radius ** 2 - dist_rescale ** 2)) + near
                prior[0, area] = depth[area]
                return prior.cuda()
            else:
                return torch.ones([1, height, width])

    def image_mask(self, image):
        with torch.no_grad():
            size = 473
            image = utils.resize(image, [size, size])
            # FIXME: only if car, cat
            image = image / 2 + 0.5
            image[:, 0].sub_(0.485).div_(0.229)
            image[:, 1].sub_(0.456).div_(0.224)
            image[:, 2].sub_(0.406).div_(0.225)
            out = self.model.mask_net(image)
            out = out.argmax(dim=1, keepdim=True)
            if self.category in Trainer.CATEGORIES:
                mask = (out == Trainer.CATEGORY2NUMBER[self.category])
            else:
                mask = torch.ones(out.size(), dtype=torch.bool)

            if not torch.any(mask):
                logging.warning(f'Did not find any {self.category} in image {image}')
                mask = torch.ones(out.size(), dtype=torch.bool)
            mask = mask.float()
        return utils.resize(mask, [self.image_size, self.image_size])

    @staticmethod
    def default_optimizer(model_list, lr=1e-4, betas=(0.9, 0.999), weight_decay=5e-4):
        param_list = []
        for model in model_list:
            params = filter(lambda param: param.requires_grad,
                            model.parameters())
            param_list += list(params)
        return torch.optim.Adam(param_list, lr=lr,
                                betas=betas, weight_decay=weight_decay)


class GeneralizingTrainer(Trainer):
    # exactly as the training class but the training loop
    # is designed to favor generalization
    def fit(self, images_latents, plot_depth_map=False, load_dict=None,
            stages=[{'step1': 1, 'step2': 1, 'step3': 1}]*2,
            batch_size=2, shuffle=False):
        if load_dict is not None:
            self.load_model_checkpoint(load_dict)

        total_it = 0
        n_stages = len(stages)
        dataloader = DataLoader(images_latents,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=self.n_workers)

        # -----------------Pretrain on all images------------------------
        data_iterator = tqdm(dataloader)
        data_iterator.set_description("Pretraining depth net")
        for batch in data_iterator:
            images, latents, data_indices = batch
            images, latents = images.cuda(), latents.cuda()
            if not self.debug:
                # Pretrain depth net on the prior shape
                self.pretrain_on_prior(images, data_indices, plot_depth_map)

        # -----------------Loop through all stages-------------------------
        for stage in range(n_stages):
            # -----------------------------Step 1--------------------------
            if self.debug:
                logging.info(f"Doing step 1, stage {stage + 1}/{n_stages}")
            data_iterator.set_description(f"Stage: {stage}/{n_stages}. "
                                          + f"Image: {data_indices+1}/{len(images_latents)}."
                                          + "Step: 1.")
            step1_collected = [None]*len(images_latents)
            optim = self.optim_step1
            for _ in range(stages[stage]['step1']):
                # -----------------Loop through all images-----------------
                for batch in data_iterator:
                    images, latents, data_indices = batch
                    images, latents = images.cuda(), latents.cuda()

                    optim.zero_grad()

                    loss, collected = self.model.forward_step1(images, latents, None)

                    normals, lights_a, lights_b, albedos, depths, canon_masks = collected
                    for collected_index, data_index in enumerate(data_indices):
                        step1_collected[data_index] = (normals[collected_index:collected_index+1],
                                                       lights_a[collected_index:collected_index+1],
                                                       lights_b[collected_index:collected_index+1],
                                                       albedos[collected_index:collected_index+1],
                                                       depths[collected_index:collected_index+1],
                                                       canon_masks[collected_index])
                    loss.backward()
                    optim.step()
                    total_it += 1

                    if self.log_wandb:
                        wandb.log({"stage": stage,
                                   "total_it": total_it,
                                   "loss_step1": loss,
                                   "image_num": data_indices})
            # -----------------------------Step 2 and 3------------------------
            if self.debug:
                logging.info(f"Doing step 3, stage {stage + 1}/{n_stages}")
            data_iterator.set_description(f"Stage: {stage}/{n_stages}. "
                                          + f"Image: {data_indices+1}/{len(images_latents)}."
                                          + "Step: 3.")
            for _ in range(stages[stage]['step2']):
                for batch in data_iterator:
                    images, latents, data_indices = batch
                    images, latents = images.cuda(), latents.cuda()

                    for batch_index in range(len(images)):
                        image = images[batch_index:batch_index+1]
                        latent = latents[batch_index:batch_index+1]
                        index = data_indices[batch_index]

                        self.optim_step2.zero_grad()
                        self.optim_step3.zero_grad()
                        collected = step1_collected[index]

                        # step 2
                        loss_step2, collected = self.model.forward_step2(image, latent, collected, self.n_proj_samples)

                        # step 3
                        loss_step3, _ = self.model.forward_step3(image, latent, collected)
                        step1_collected[index] = collected
                        loss_step2.backward()
                        loss_step3.backward()
                        self.optim_step2.step()
                        self.optim_step3.step()
                        total_it += 1

                        if self.log_wandb:
                            wandb.log({"stage": stage,
                                       "total_it": total_it,
                                       "loss_step2": loss_step2,
                                       "loss_step3": loss_step3,
                                       "image_num": data_indices})

            if self.plot_intermediate:
                if index % 3 == 0:
                    recon_im, recon_depth = self.model.evaluate_results(images)
                    recon_im, recon_depth = recon_im.cpu(), recon_depth.cpu()
                    plot_reconstructions(recon_im, recon_depth,
                                         total_it=str(total_it),
                                         im_idx=str(index),
                                         stage=str(stage))

        if self.save_ckpts:
            self.model.save_checkpoint(data_indices, stage, total_it, self.category)
        logging.info('Finished Training')
