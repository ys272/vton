import random
import sys
import argparse
from torch.utils.tensorboard import SummaryWriter # TensorBoard support
from torch.nn import init
from torchvision.utils import save_image
from torch.optim import Adam
from pathlib import Path
from model import *
from algo.nn_utils import *
from diffusion_ddim import *
from datasets_fmnist import train_loader as fmnist_train_dataloader
import time
from datetime import datetime
import torchvision
import copy
from utils import denormalize_img
from diffusion_karras import *
from algo.base_vton.datasets import train_dataloader, valid_dataloader, test_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training experiment description')
    parser.add_argument('-d', '--desc', type=str, help='training experiment description')
    args = parser.parse_args()
    description = ''
    if args.desc is not None:
        description = '_' + args.desc
        description = description.replace(' ', '_')

    current_timestamp = time.time()
    current_date = datetime.fromtimestamp(current_timestamp).strftime('%d-%B-%H:%M') # e.g '20-July-17:38'
    human_readable_timestamp = current_date + description
    tb = SummaryWriter(log_dir=os.path.join(c.MODEL_OUTPUT_TBOARD_DIR, human_readable_timestamp))
    
    img_size = c.IMAGE_SIZE
    img_height = c.VTON_RESOLUTION[img_size][0]
    img_width = c.VTON_RESOLUTION[img_size][1]

    # model = Unet_Person_Masked(channels=6, level_dims=(360, 360, 360),level_attentions=(False, True),level_repetitions = (4,5,6),) # 3 for masked image, 3 for noise
    # model = Unet_Person_Masked(channels=6, level_dims=(460, 460, 460),level_attentions=(False, True),level_repetitions = (4,5,6),) # 3 for masked image, 3 for noise
    model = Unet_Person_Masked(channels=6, level_dims=(16, 16, 16),level_attentions=(False, True),level_repetitions = (4,5,6),) # 3 for masked image, 3 for noise
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters in the model: {total_params}')
    model.to(c.DEVICE)

    # initial_learning_rate = 1e-8 # Use this when applying 1cycle policy.
    # final_learning_rate = 1e-4
    # learning_rates = np.linspace(1e-8, 1e-4, num=10000)
    initial_learning_rate = 1e-4
    optimizer = Adam(model.parameters(), lr=initial_learning_rate, eps=1e-5)
    batch_num = 0

    # Load model from checkpoint.
    if False:
        model_state = torch.load(os.path.join(c.MODEL_OUTPUT_PARAMS_DIR, '24-July_karras.pth'))
        model.load_state_dict(model_state['model_state_dict'])
        optimizer.load_state_dict(model_state['optimizer_state_dict'])
        batch_num = model_state['batch_num']
        initial_learning_rate = model_state['learning_rate']
        for g in optimizer.param_groups:
            g['lr'] = initial_learning_rate
        used_checkpoint_msg = f'LOADED CHECKPOINT!!! LR: {initial_learning_rate}'
        print(used_checkpoint_msg)
        with open(os.path.join(c.MODEL_OUTPUT_LOG_DIR, f'{human_readable_timestamp}_train_log.txt'), 'w') as log_file:
            log_file.write(used_checkpoint_msg)
    
    # x = torch.randn(c.BATCH_SIZE, 3, image_size, image_size, device=c.DEVICE) * c.NOISE_SCALING_FACTOR
    # t = torch.randint(0, c.NUM_TIMESTEPS, (c.BATCH_SIZE,), device=c.DEVICE).long()
    # output = model(x,t)
    # print(output.size())
    # make_dot(model(x,t), params=dict(model.named_parameters())).render("/home/yoni/Desktop/fash_model", format="png")
    
    # call_sampler_simple(model, (10, 3, image_size, image_size), sampler='ddim', clip_model_output=True, show_all=True, eta=1)
    # call_sampler_simple(model, (50, 3, image_size, image_size), sampler='ddim', clip_model_output=True, show_all=False, eta=1)
    # call_sampler_simple_karras(model, 50, sampler='euler_ancestral', steps=250, sigma_max=80.0, clip_model_output=True, show_all=False)
    
    clothing_aug, masked_aug, person, pose, sample_original_string_id, sample_unique_ordinal_id, noise_amount_clothing, noise_amount_masked = next(iter(train_dataloader))
    person = person.to(c.DEVICE)
    grid = torchvision.utils.make_grid(person)
    tb.add_image('images', grid)
    t = torch.randint(0, c.NUM_TIMESTEPS, (c.BATCH_SIZE,), device=c.DEVICE)
    noise = torch.randn_like(masked_aug) * c.NOISE_SCALING_FACTOR
    noise_and_masked_aug = torch.cat((noise, masked_aug), dim=1).to(torch.float32)
    input_tuple = (noise_and_masked_aug.to(c.DEVICE), pose.to(torch.float32).to(c.DEVICE), noise_amount_masked.to(torch.float32).to(c.DEVICE), t.to(c.DEVICE))
    tb.add_graph(model, input_tuple)

    epochs = 1000
    scaler = torch.cuda.amp.GradScaler()
    ema_batch_num_start = 50000
    ema = EMA(0.995, ema_batch_num_start)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    trainer_helper = TrainerHelper(human_readable_timestamp)
    # Enable cuDNN auto-tuner: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner
    torch.backends.cudnn.benchmark = True
    if c.OPTIMIZE:
        torch.autograd.set_detect_anomaly(False) 
        torch.autograd.profiler.profile(enabled=False)
        torch.autograd.gradcheck_enabled = False
        

    with open(os.path.join(c.MODEL_OUTPUT_LOG_DIR, f'{human_readable_timestamp}_train_log.txt'), 'w') as log_file:
        for epoch in range(epochs):
            for step, batch in enumerate(train_dataloader):
                # Code for finding maximum learning rate.
                # if mini_batch_counter%100==0 and mini_batch_counter != 0 and initial_learning_rate < 0.01:
                #     initial_learning_rate *= math.sqrt(10) 
                #     for g in optimizer.param_groups:
                #         g['lr'] = initial_learning_rate
                #     print(f'mini_batch_counter {mini_batch_counter} lr: {initial_learning_rate}')
                
                # Code for applying 1cycle policy.
                # if mini_batch_counter < 10000:
                #     for g in optimizer.param_groups:
                #         g['lr'] = learning_rates[mini_batch_counter]
                        
                optimizer.zero_grad(set_to_none=True)
                batch_num += 1
                clothing_aug, masked_aug, person, pose, sample_original_string_id, sample_unique_ordinal_id, noise_amount_clothing, noise_amount_masked = batch
                clothing_aug, masked_aug, person, pose, noise_amount_clothing, noise_amount_masked = clothing_aug.cuda(), masked_aug.cuda(), person.cuda(), pose.cuda(), noise_amount_clothing.cuda(), noise_amount_masked.cuda()

                # show_example_noise_sequence(batch[:5].squeeze(1))
                # show_example_noise_sequence_karras(batch[:5].squeeze(1), 100)
                # Sample t uniformally for every example in the batch
                batch_size = masked_aug.shape[0]
                if c.REVERSE_DIFFUSION_SAMPLER == 'karras':
                    t = (torch.randn([batch_size])*1.2-1.2).exp().to(c.DEVICE)
                else:
                    t = torch.randint(0, c.NUM_TIMESTEPS, (batch_size,), device=c.DEVICE)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = p_losses(model, clothing_aug, masked_aug, person, pose, noise_amount_clothing, noise_amount_masked, t, loss_type="l1")

                if batch_num % 100 == 0:
                    print(f'Loss for epoch {epoch}, batch {batch_num}:', loss.item())

                scaler.scale(loss).backward() # loss.backward()
                scaler.step(optimizer) # optimizer.step()
                scaler.update()
                ema.step_ema(ema_model, model)
                
                num_batches_since_min_loss = trainer_helper.update_loss_possibly_save_model(loss, model, optimizer, batch_num)
                if num_batches_since_min_loss > 10000:
                    if num_batches_since_min_loss > 25000:
                        sys.exit('Loss has not improved for 25,000 batches. Terminating the flow.')
                    if trainer_helper.num_batches_since_last_learning_rate_reduction(batch_num) > 10000:
                        for g in optimizer.param_groups:
                            g['lr'] /= math.sqrt(10) # divide learning rate by sqrt(10)
                        trainer_helper.update_last_learning_rate_reduction(batch_num)
                        lr_reduction_msg = f'LR REDUCTION: {g["lr"]}, at batch num: {batch_num}\n'
                        print(lr_reduction_msg)
                        log_file.write(lr_reduction_msg)
                elif num_batches_since_min_loss == 0:
                    loss_decrease_msg = f'LOSS DECREASED & MODEL SAVED, at batch num: {batch_num}\n'
                    print(loss_decrease_msg)
                    log_file.write(loss_decrease_msg)

                # Save generated images.
                if batch_num != 0 and batch_num % 100 == 0:
                    # sample one random batch
                    random_samples = random.sample(list(iter(valid_dataloader)), 1)
                    clothing_aug, masked_aug, person, pose, sample_original_string_id, sample_unique_ordinal_id, noise_amount_clothing, noise_amount_masked = random_samples[0]
                    clothing_aug, masked_aug, person, pose, noise_amount_clothing, noise_amount_masked = clothing_aug.cuda(), masked_aug.cuda(), person.cuda(), pose.cuda(), noise_amount_clothing.cuda(), noise_amount_masked.cuda()

                    for m,suffix in [(model, ''), (ema_model, '_ema')]:
                        if suffix == '_ema' and batch_num < ema_batch_num_start:
                                continue
                        if c.REVERSE_DIFFUSION_SAMPLER == 'karras':
                            img_sequences = p_sample_loop_karras(sample_euler_karras, model, steps=250)
                        else:
                            img_sequences = p_sample_loop(m, shape=(4, 3, img_height, img_width))
                        for i,img in enumerate(img_sequences[-1]):
                            if c.REVERSE_DIFFUSION_SAMPLER == 'karras':
                                img = img.clamp(-1,1)
                            img = denormalize_img(img)
                            if c.REVERSE_DIFFUSION_SAMPLER != 'karras':
                                img = torch.from_numpy(img)
                            save_image(img, os.path.join(c.MODEL_OUTPUT_IMAGES_DIR, f'sample-{batch_num}_{i}{suffix}.png'), nrow = 4//2)
                
                tb.add_scalar('train loss', loss, batch_num)
                
            # for name, param in model.named_parameters():
            #     tb.add_histogram(name, param, epoch)
            #     tb.add_histogram(f'{name}.grad', param.grad, epoch)
            
    tb.close()
    
    