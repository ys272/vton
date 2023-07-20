import sys
import argparse
from torch.utils.tensorboard import SummaryWriter # TensorBoard support
from torch.nn import init
from torchvision.utils import save_image
from torch.optim import Adam
from pathlib import Path
from model import *
from algo.nn_utils import *
from algo.diffusion_utils import *
from datasets import train_loader
import time
from datetime import datetime
import torchvision
import copy
from utils import call_sampler_simple


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

    save_and_sample_every = 1000
    image_size = 28
    num_channels = 1
    num_dims_first_layer = 16

    model = Unet(
        dim=16,
        channels=num_channels,
        dim_mults=(1, 2, 4,)
    )
    model.to(c.DEVICE)

    # initial_learning_rate = 1e-8 # Use this when applying 1cycle policy.
    # final_learning_rate = 1e-4
    # learning_rates = np.linspace(1e-8, 1e-4, num=10000)
    initial_learning_rate = 1e-3
    optimizer = Adam(model.parameters(), lr=initial_learning_rate, eps=1e-5)
    batch_num = 0

    # Load model from checkpoint.
    if False:
        model_state = torch.load(os.path.join(c.MODEL_OUTPUT_PARAMS_DIR, '20-07-23-1689863578.pth'))
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
    
    # x = torch.randn(c.BATCH_SIZE, num_channels, image_size, image_size, device=c.DEVICE)
    # t = torch.randint(0, c.NUM_TIMESTEPS, (c.BATCH_SIZE,), device=c.DEVICE).long()
    # output = model(x,t)
    # print(output.size())
    # make_dot(model(x,t), params=dict(model.named_parameters())).render("/home/yoni/Desktop/fash_model", format="png")
    
    # call_sampler_simple(model, (10, num_channels, image_size, image_size), sampler='ddim', clip_model_output=True)
    
    images, labels = next(iter(train_loader))
    images = images.to(c.DEVICE)
    grid = torchvision.utils.make_grid(images)
    tb.add_image('images', grid)
    t = torch.randint(0, c.NUM_TIMESTEPS, (c.BATCH_SIZE,), device=c.DEVICE)
    input_tuple = (images.to(torch.float32), t)
    tb.add_graph(model, input_tuple)

    epochs = 1000
    scaler = torch.cuda.amp.GradScaler()
    ema_batch_num_start = 50000
    ema = EMA(0.995, ema_batch_num_start)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    trainer_helper = TrainerHelper(human_readable_timestamp)

    with open(os.path.join(c.MODEL_OUTPUT_LOG_DIR, f'{human_readable_timestamp}_train_log.txt'), 'w') as log_file:
        for epoch in range(epochs):
            for step, batch in enumerate(train_loader):
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
                        
                optimizer.zero_grad()
                batch_num += 1
                batch = batch[0].to(c.DEVICE)
                # Sample t uniformally for every example in the batch
                t = torch.randint(0, c.NUM_TIMESTEPS, (c.BATCH_SIZE,), device=c.DEVICE)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = p_losses(model, batch, t, loss_type="huber")

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
                if batch_num != 0 and batch_num % save_and_sample_every == 0:
                    for m,suffix in [(model, ''), (ema_model, '_ema')]:
                        img_sequences = p_sample_loop(m, shape=(4, num_channels, image_size, image_size))
                        for i,img in enumerate(img_sequences[-1]):
                            if suffix == '_ema' and batch_num < ema_batch_num_start:
                                continue
                            if c.MIN_NORMALIZED_VALUE == -0.5:
                                img = img + 0.5 # [-0.5,0.5] normalization 
                            elif c.MIN_NORMALIZED_VALUE == -1:
                                img =  (img + 1) * 0.5 # [-1,1] normalization
                            else:
                                sys.exit('unsupported normalization')
                            save_image(torch.from_numpy(img), os.path.join(c.MODEL_OUTPUT_IMAGES_DIR, f'sample-{batch_num}_{i}{suffix}.png'), nrow = 4//2)
                
                tb.add_scalar('train loss', loss, batch_num)
                
            for name, param in model.named_parameters():
                tb.add_histogram(name, param, epoch)
                tb.add_histogram(f'{name}.grad', param.grad, epoch)
            
    tb.close()