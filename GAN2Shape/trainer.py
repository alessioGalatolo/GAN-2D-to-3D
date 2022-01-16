import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from GAN2Shape.priors import PriorGenerator
from plotting import plot_predicted_depth_map, plot_reconstructions
try:
    import wandb
except ImportError:
    wandb = None


class Trainer():

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
        self.plot_intermediate = plot_intermediate
        self.log_wandb = log_wandb
        self.save_ckpts = save_ckpts
        self.debug = debug

        self.prior_generator = PriorGenerator(self.image_size,
                                              self.category,
                                              model_config.get('prior_name',
                                                               'ellipsoid'))

        self.optim_step1 = Trainer.default_optimizer([self.model.albedo_net],
                                                     lr=self.learning_rate)
        self.optim_step2 = Trainer.default_optimizer([self.model.offset_encoder_net],
                                                     lr=self.learning_rate)
        self.optim_step3 = Trainer.default_optimizer([self.model.lighting_net,
                                                      self.model.viewpoint_net,
                                                      self.model.depth_net,
                                                      self.model.albedo_net],
                                                     lr=self.learning_rate)

        self.load_dict = load_dict
        if load_dict is not None:
            paths, _ = self.model.build_checkpoint_path(load_dict['base_path'],
                                                        load_dict['category'],
                                                        general=True)
            self.model.load_from_checkpoint(paths[-1])

    def fit(self, images_latents, plot_depth_map=False,
            stages=[{'step1': 1, 'step2': 1, 'step3': 1}]*2,
            shuffle=False, **_):

        # continue previously started training

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
                if self.load_dict is None:
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
        prior = self.prior_generator(image)

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
    def __init__(self, model, model_config, debug=False, plot_intermediate=False,
                 log_wandb=False, save_ckpts=False, load_dict=None):
        super().__init__(model, model_config, debug=debug,
                         plot_intermediate=plot_intermediate,
                         log_wandb=log_wandb, save_ckpts=save_ckpts,
                         load_dict=load_dict)
        self.n_epochs = model_config.get('n_epochs_generalized', 1)

    def fit(self, images_latents, plot_depth_map=False, load_dict=None,
            stages=[{'step1': 1, 'step2': 1, 'step3': 1}]*2,
            batch_size=2, shuffle=False):

        total_it = 0
        data_iterator_priors = tqdm(DataLoader(images_latents,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=self.n_workers))

        dataloader = DataLoader(images_latents,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=self.n_workers)

        # -----------------Pretrain on all images------------------------
        data_iterator = tqdm(dataloader)
        # Pretrain depth net on the prior shape
        if self.load_dict is None:
            self.pretrain_on_prior(data_iterator_priors, data_iterator, plot_depth_map=plot_depth_map)

        # -----------------Loop through all epochs-------------------------
        data_iterator = tqdm(range(self.n_epochs))
        for epoch in data_iterator:
            # -----------------------------Step 1--------------------------
            if self.debug:
                logging.info(f"Doing step 1, epoch {epoch + 1}/{self.n_epochs}")
            data_iterator.set_description(f"epoch: {epoch}/{self.n_epochs}. "
                                          + f"Image: {data_indices+1}/{len(images_latents)}."
                                          + "Step: 1.")
            step1_collected = [None]*len(images_latents)
            optim = self.optim_step1
            for _ in range(stages[0]['step1']):
                # -----------------Loop through all images-----------------
                for batch in tqdm(dataloader):
                    images, latents, data_indices = batch
                    images, latents = images.cuda(), latents.cuda()

                    optim.zero_grad()

                    loss, collected = self.model.forward_step1(images, latents, None)

                    normals, lights_a, lights_b, albedos, depths, canon_masks = collected
                    if type(canon_masks) is not list:
                        canon_masks = [canon_masks]
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
                        wandb.log({"epoch": epoch,
                                   "total_it": total_it,
                                   "loss_step1": loss,
                                   "image_num": data_indices})
            # -----------------------------Step 2 and 3------------------------
            if self.debug:
                logging.info(f"Doing step 3, epoch {epoch + 1}/{self.n_epochs}")
            data_iterator.set_description(f"epoch: {epoch}/{self.n_epochs}. "
                                          + f"Image: {data_indices+1}/{len(images_latents)}."
                                          + "Step: 3.")
            for _ in range(stages[0]['step2']):
                for batch in tqdm(dataloader):
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
                        loss_step2, collected = self.model.forward_step2(image,
                                                                         latent,
                                                                         collected,
                                                                         self.n_proj_samples)

                        # step 3
                        loss_step3, _ = self.model.forward_step3(image, latent, collected)
                        step1_collected[index] = collected
                        loss_step2.backward()
                        loss_step3.backward()
                        self.optim_step2.step()
                        self.optim_step3.step()
                        total_it += 1

                        if self.log_wandb:
                            wandb.log({"epoch": epoch,
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
                                         epoch=str(epoch))

        if self.save_ckpts:
            self.model.save_checkpoint(data_indices, epoch, total_it, self.category)
        logging.info('Finished Training')

    def pretrain_on_prior(self, data_iterator_priors, data_iterator, plot_depth_map=False):
        optim = Trainer.default_optimizer([self.model.depth_net])
        train_loss = []
        logging.info("Pretraining depth net on prior shape")
        data_iterator_priors.set_description("Generating priors for the dataset")
        priors = torch.zeros((len(data_iterator_priors), self.image_size, self.image_size))
        for img in data_iterator_priors:
            image, _, img_idx = img
            image = image.cuda()
            prior = self.prior_generator(image, device='cpu')
            priors[img_idx] = prior

        data_iterator.set_description("Pretraining depth net")
        iterator = tqdm(range(self.n_epochs_prior))
        for epoch in iterator:
            for batch in data_iterator:
                images, latents, data_indices = batch
                images, latents = images.cuda(), latents.cuda()
                prior = priors[data_indices].cuda()
                optim.zero_grad()
                loss, depth = self.model.depth_net_forward(images, prior)
                loss.backward()
                optim.step()

            with torch.no_grad():
                iterator.set_description(f"Depth net prior loss = {loss.cpu()}")

            if self.log_wandb:
                wandb.log({"loss_prior": loss.cpu(),
                           "epoch_prior": epoch})

        if plot_depth_map:
            depth = depth.detach().cpu().numpy()
            plot_predicted_depth_map(depth, self.image_size, block=True)
        return train_loss


class GeneralizingTrainer2(GeneralizingTrainer):
    # exactly as the training class but the training loop
    # is designed to favor generalization
    def __init__(self, model, model_config, debug=False, plot_intermediate=False,
                 log_wandb=False, save_ckpts=False, load_dict=None):
        super().__init__(model, model_config, debug=debug,
                         plot_intermediate=plot_intermediate,
                         log_wandb=log_wandb, save_ckpts=save_ckpts,
                         load_dict=load_dict)
        self.n_epochs = model_config.get('n_epochs_generalized', 1)

    def fit(self, images_latents, plot_depth_map=False,
            stages=[{'step1': 1, 'step2': 1, 'step3': 1}]*2,
            batch_size=2, shuffle=False):

        total_it = 0
        data_iterator_priors = tqdm(DataLoader(images_latents,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=self.n_workers))

        dataloader = DataLoader(images_latents,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=self.n_workers)

        # -----------------Pretrain on all images------------------------
        data_iterator = tqdm(dataloader)
        # Pretrain depth net on the prior shape
        if self.load_dict is None:
            self.pretrain_on_prior(data_iterator_priors, data_iterator, plot_depth_map=plot_depth_map)

        # -----------------Loop through all epochs-------------------------
        data_iterator = tqdm(range(self.n_epochs))
        for epoch in data_iterator:
            # -----------------------------Step 1--------------------------
            if self.debug:
                logging.info(f"Doing step 1, epoch {epoch + 1}/{self.n_epochs}")

            optim = self.optim_step1
            # -----------------Loop through all images-----------------
            for i_b, batch in enumerate(dataloader):
                data_iterator.set_description(f"epoch: {epoch}/{self.n_epochs}. "
                                              + f"Batch: {i_b+1}/{len(dataloader)}."
                                              + "Step: 1.")
                images, latents, data_indices = batch

                step1_collected = [None] * images.shape[0]

                images, latents = images.cuda(), latents.cuda()
                for _ in range(stages[0]['step1']):
                    optim.zero_grad()

                    loss, collected = self.model.forward_step1(images, latents, None)

                    normals, lights_a, lights_b, albedos, depths, canon_masks = collected
                    if type(canon_masks) is not list:
                        canon_masks = [canon_masks]
                    loss.backward()
                    optim.step()
                    total_it += 1
                for batch_index in range(images.shape[0]):
                    step1_collected[batch_index] = (normals[batch_index].unsqueeze(0).cpu(),
                                                    lights_a[batch_index].unsqueeze(0).cpu(),
                                                    lights_b[batch_index].unsqueeze(0).cpu(),
                                                    albedos[batch_index].unsqueeze(0).cpu(),
                                                    depths[batch_index].unsqueeze(0).cpu(),
                                                    canon_masks[batch_index])

                if self.log_wandb:
                    wandb.log({"epoch": epoch,
                               "total_it": total_it,
                               "loss_step1": loss})

                # -----------------------------Step 2 and 3------------------------
                if self.debug:
                    logging.info(f"Doing step 3, epoch {epoch + 1}/{self.n_epochs}")

                # Free up GPU memory
                images, latents = images.cpu(), latents.cpu()
                torch.cuda.empty_cache()
                for batch_index in range(len(images)):
                    image = images[batch_index].unsqueeze(0).cuda()
                    latent = latents[batch_index].unsqueeze(0).cuda()
                    step1_collected_batch = step1_collected[batch_index]
                    # Ugly af, probably some cleaner way of doing this
                    normal, light_a, light_b, albedo, depth, canon_mask = step1_collected_batch
                    step1_collected_batch = (normal.cuda(),
                                             light_a.cuda(),
                                             light_b.cuda(),
                                             albedo.cuda(),
                                             depth.cuda(),
                                             canon_mask)

                    data_iterator.set_description(f"epoch: {epoch}/{self.n_epochs}. "
                                                  + f"Batch: {i_b+1}/{len(dataloader)}. "
                                                  + f"Sub-batch: {batch_index+1}/{len(images)}. "
                                                  + "Step: 2.")
                    for _ in range(stages[0]['step2']):
                        self.optim_step2.zero_grad()
                        # step 2
                        loss_step2, collected = self.model.forward_step2(image,
                                                                         latent,
                                                                         step1_collected_batch,
                                                                         self.n_proj_samples)
                        loss_step2.backward()
                        self.optim_step2.step()
                        total_it += 1

                    data_iterator.set_description(f"epoch: {epoch}/{self.n_epochs}. "
                                                  + f"Batch: {i_b+1}/{len(dataloader)}. "
                                                  + f"Sub-batch: {batch_index+1}/{len(images)}. "
                                                  + "Step: 3.")
                    for _ in range(stages[0]['step3']):
                        self.optim_step3.zero_grad()
                        # step 3
                        loss_step3, _ = self.model.forward_step3(image, latent, collected)
                        loss_step3.backward()
                        self.optim_step3.step()
                        total_it += 1

                    torch.cuda.empty_cache()
                    if self.log_wandb:
                        wandb.log({"epoch": epoch,
                                   "total_it": total_it,
                                   "loss_step2": loss_step2,
                                   "loss_step3": loss_step3,
                                   "image_num": data_indices})

            # if self.plot_intermediate:
            #     if epoch % 20 == 0:
            #         sample
            #         recon_im, recon_depth = self.model.evaluate_results(images)
            #         recon_im, recon_depth = recon_im.cpu(), recon_depth.cpu()
            #         plot_reconstructions(recon_im, recon_depth,
            #                              total_it=str(total_it),
            #                              im_idx=str(index),
            #                              epoch=str(epoch))

            if epoch % 20 == 0 and self.save_ckpts:
                self.model.save_checkpoint("", epoch, total_it, self.category)
        logging.info('Finished Training')
