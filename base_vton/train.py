import sys
import argparse
from torch.utils.tensorboard import SummaryWriter # TensorBoard support
from torchvision.utils import save_image
from torch.optim import Adam, AdamW
from model import *
from algo.nn_utils import *
from diffusion_ddim import *
import time
from datetime import datetime
import copy
from utils import denormalize_img, save_or_return_img_w_overlaid_keypoints
from diffusion_karras import *
from algo.base_vton.datasets import create_datasets


def hook_fn(name, batch_num=None):
    def hook(module, inp, output):
        mean = torch.mean(output)
        # mean_i = torch.mean(inp[0].float())
        if torch.isnan(mean) or torch.isinf(mean):
            print(f'!!!!!!!!!!!!!!!!!!NAN!!! {name}, {mean}')
        # else:
        #     print(f'{name}: {mean_i.item():.3f} --- {mean.item():.3f}')
        tb.add_histogram(name + '.activation-in', inp[0], global_step=batch_num)
        tb.add_histogram(name + '.activation-out', output, global_step=batch_num)
    return hook
                
def add_hooks(module, name='', base_name='main_', batch_num=None):
    for child_name, child_module in module.named_children():
        if list(child_module.children()):
            # If the child_module has children, recursively call add_hooks.
            if name == '':
                add_hooks(child_module, child_name, base_name=base_name)
            else:
                add_hooks(child_module, name + '.' + child_name, base_name=base_name)
        else:
            # If the child_module is a leaf node, register the hook
            final_name = name + '.' + child_name if name != '' else child_name
            # hooks[base_name + name + '.' + child_name] = child_module.register_forward_hook(hook_fn(base_name + name + '.' + child_name, batch_num=batch_num))
            hooks[base_name + name + '.' + child_name] = child_module.register_forward_hook(hook_fn(base_name + final_name, batch_num=batch_num))

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
    
    img_height = c.VTON_RESOLUTION[c.IMAGE_SIZE][0]
    img_width = c.VTON_RESOLUTION[c.IMAGE_SIZE][1]
    init_dim = c.MODELS_INIT_DIM
    level_dims_main = c.MODELS_PARAMS[c.IMAGE_SIZE][0]
    level_dims_aux = c.MODELS_PARAMS[c.IMAGE_SIZE][1]
    level_attentions = c.MODELS_PARAMS[c.IMAGE_SIZE][2]
    level_repetitions_main = c.MODELS_PARAMS[c.IMAGE_SIZE][3]
    level_repetitions_aux = c.MODELS_PARAMS[c.IMAGE_SIZE][4]
    
    model_main = Unet_Person_Masked(channels=19, init_dim=init_dim, level_dims=level_dims_main, level_dims_cross_attn=level_dims_aux, level_attentions=level_attentions,level_repetitions = level_repetitions_main,).to(c.DEVICE)
    model_aux = Unet_Clothing(channels=3, init_dim=init_dim, level_dims=level_dims_aux,level_repetitions=level_repetitions_aux,).to(c.DEVICE)
    print(f'Total parameters in the main model: {sum(p.numel() for p in model_main.parameters()):,}')
    print(f'Total parameters in the aux model:  {sum(p.numel() for p in model_aux.parameters()):,}')

    # initial_learning_rate = 1e-4
        
    initial_learning_rate = 1e-6 # Use this when applying 1cycle policy.
    final_learning_rate = 1e-4
    num_LR_decay_cycles = 40000
    learning_rates = np.linspace(initial_learning_rate, final_learning_rate, num=num_LR_decay_cycles)
    
    optimizer = Adam(list(model_main.parameters()) + list(model_aux.parameters()), lr=initial_learning_rate, eps=c.ADAM_EPS)
    scaler = torch.cuda.amp.GradScaler()
    accumulation_rate = c.BATCH_ACCUMULATION
    batch_num = 0
    batch_num_last_accumulate_rate_update = batch_num
    # Represents the number of batches we've done backprop on (will differ from batch num if accumulation_rate != 1)
    backprop_batch_num = 0
    min_loss = float('inf')
    
    last_save_batch_num = 0
    ema_batch_num_start = 1000000
    ema = EMA(0.999, ema_batch_num_start)
    if c.RUN_EMA:
        ema_model_main = copy.deepcopy(model_main).eval().requires_grad_(False)
        ema_model_aux = copy.deepcopy(model_aux).eval().requires_grad_(False)
    else:
        ema_model_main = None
        ema_model_aux = None
    
    # Load model from checkpoint.
    if 1:
        model_state = torch.load(os.path.join(c.MODEL_OUTPUT_PARAMS_DIR, '19-September-21:57_2740012_normal_loss_0.037.pth'))
        model_main.load_state_dict(model_state['model_main_state_dict'])
        model_aux.load_state_dict(model_state['model_aux_state_dict'])
        optimizer.load_state_dict(model_state['optimizer_state_dict'])
        scaler.load_state_dict(model_state['scaler_state_dict'])
        
        min_loss = model_state['loss']
        batch_num = model_state['batch_num']
        backprop_batch_num = model_state['backprop_batch_num']
        batch_num_last_accumulate_rate_update = batch_num
        accumulation_rate = model_state['accumulation_rate']
        initial_learning_rate = model_state['learning_rate']
        last_save_batch_num = model_state['last_save_batch_num']
        
        was_ema_initialized = model_state['was_ema_initialized']
        if c.RUN_EMA and was_ema_initialized:
            ema_model_main.load_state_dict(model_state['model_ema_main_state_dict'])
            ema_model_aux.load_state_dict(model_state['model_ema_aux_state_dict'])
            ema = EMA(0.999, ema_batch_num_start, was_i_initialized=was_ema_initialized)
        
        del model_state
        torch.cuda.empty_cache()
        
        for g in optimizer.param_groups:
            g['lr'] = initial_learning_rate

        used_checkpoint_msg = f'LOADED CHECKPOINT!!! LR: {initial_learning_rate}, accumulation rate: {accumulation_rate}, batch num {batch_num}'
        print(used_checkpoint_msg)
        with open(os.path.join(c.MODEL_OUTPUT_LOG_DIR, f'{human_readable_timestamp}_train_log.txt'), 'w') as log_file:
            log_file.write(used_checkpoint_msg)

    train_dataloader, valid_dataloader, test_dataloader = create_datasets()
    
    # clothing_aug, mask_coords, masked_aug, person, pose, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = next(iter(test_dataloader))
    # num_eval_samples = min(8, clothing_aug.shape[0])
    # inputs = [clothing_aug[:num_eval_samples].cuda().float(), mask_coords[:num_eval_samples].cuda(), masked_aug[:num_eval_samples].cuda().float(), person[:num_eval_samples].cuda().float(), pose[:num_eval_samples].cuda().float(), sample_original_string_id, sample_unique_string_id, noise_amount_clothing[:num_eval_samples].cuda().float(), noise_amount_masked[:num_eval_samples].cuda().float()]
    # call_sampler_simple(model_main, model_aux, inputs, shape=(num_eval_samples, 3, img_height, img_width), sampler='ddim', clip_model_output=True, show_all=True, eta=1)
    # call_sampler_simple_karras(model_main, model_aux, inputs, sampler='euler',steps=250, sigma_max=c.KARRAS_SIGMA_MAX, clip_model_output=True, show_all=True)
    
    epochs = 1000000
    trainer_helper = TrainerHelper(human_readable_timestamp, min_loss = min_loss, min_loss_batch_num=batch_num, backprop_batch_num=backprop_batch_num, last_save_batch_num=last_save_batch_num)
    # Enable cuDNN auto-tuner: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner
    torch.backends.cudnn.benchmark = True
    if c.OPTIMIZE:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(enabled=False)
        torch.autograd.gradcheck_enabled = False
    
    hooks = {}
    running_loss = 0
    valid_dataloader_iterator = iter(valid_dataloader)
    last_loss_print = 0
    with open(os.path.join(c.MODEL_OUTPUT_LOG_DIR, f'{human_readable_timestamp}_train_log.txt'), 'w') as log_file:
        training_start_time = time.time()
        batch_training_end_time = training_start_time
        for epoch in range(epochs):
            for batch in train_dataloader:
                batch_num += 1 
                
                if c.DEBUG_FIND_MIN_MEDIAN_GRAD_PER_BATCH:
                    min_grad = float('inf')
                    max_grad = 0
                    very_low_gradients = set()

                # Code for finding maximum learning rate.
                # if batch_num%400==0 and batch_num != 0 and initial_learning_rate < 0.01:
                #     initial_learning_rate *= math.sqrt(10) 
                #     for g in optimizer.param_groups:
                #         g['lr'] = initial_learning_rate
                #     print(f'mini_batch_counter {batch_num} lr: {initial_learning_rate}')
                
                # Code for applying 1cycle policy.
                if batch_num < num_LR_decay_cycles:
                    for g in optimizer.param_groups:
                        g['lr'] = learning_rates[batch_num]
                    
                clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = batch
                clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, noise_amount_clothing, noise_amount_masked = clothing_aug.cuda(), mask_coords.cuda(), masked_aug.cuda(), person.cuda(), pose_vector.cuda(), pose_matrix.cuda(), noise_amount_clothing.cuda(), noise_amount_masked.cuda()
                if not c.USE_AMP:
                    clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, noise_amount_clothing, noise_amount_masked = clothing_aug.float(), mask_coords, masked_aug.float(), person.float(), pose_vector.float(), pose_matrix.float(), noise_amount_clothing.float(), noise_amount_masked.float()
                else:
                    if not c.USE_BFLOAT16:
                        clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, noise_amount_clothing, noise_amount_masked = clothing_aug.to(torch.float16), mask_coords, masked_aug.to(torch.float16), person.to(torch.float16), pose_vector.to(torch.float16), pose_matrix.to(torch.float16), noise_amount_clothing.to(torch.float16), noise_amount_masked.to(torch.float16)

                # show_example_noise_sequence(person[:5].squeeze(1))
                # show_example_noise_sequence_karras(person[:5].squeeze(1), steps=100, sigma_max=c.KARRAS_SIGMA_MAX, rho=7)
                # Sample t uniformally for every example in the batch
                batch_size = masked_aug.shape[0]
                if c.REVERSE_DIFFUSION_SAMPLER == 'karras':
                    t = (torch.randn([batch_size], device=c.DEVICE)*1.2-1.2).exp()
                else:
                    t = torch.randint(0, c.NUM_DIFFUSION_TIMESTEPS, (batch_size,), device=c.DEVICE)
                
                batch_training_start_time = time.time()
                    
                # if batch_num==1 or batch_num==50 or batch_num % 1005 == 0:
                #     hooks = {}
                #     add_hooks(model_main, base_name='main_', batch_num=batch_num)
                #     add_hooks(model_aux, base_name='aux_', batch_num=batch_num)
                
                apply_cfg = random.random() < 0.1 and batch_num % 1005 != 0
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=c.USE_AMP):
                    loss = p_losses(model_main, model_aux, clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, noise_amount_clothing, noise_amount_masked, t, loss_type="l1", apply_cfg=apply_cfg)
                    
                running_loss += loss.item()
                
                if loss == 0 or loss > 1e10:
                    loss_oob_msg = f'----------------------------Loss is OOB: {loss}, for {sample_original_string_id}, {sample_unique_string_id}, {t}'
                    print(loss_oob_msg)
                    log_file.write(loss_oob_msg+'\n')
                
                loss /= accumulation_rate
                if c.USE_AMP:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if (batch_num - batch_num_last_accumulate_rate_update) % accumulation_rate == 0:
                    if c.USE_AMP:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if c.RUN_EMA:
                        ema.step_ema(ema_model_main, model_main, ema_model_aux, model_aux, batch_num)
                            
                if batch_num % 10005 == 0:
                    log_file.flush()
                    for name, param in model_main.named_parameters():
                        tb.add_histogram('main_'+name, param, batch_num)
                        mean = torch.mean(param.grad)
                        if not torch.isnan(mean):
                            tb.add_histogram(f'main_{name}.grad', param.grad, batch_num)
                            if c.DEBUG_FIND_MIN_MEDIAN_GRAD_PER_BATCH:
                                min_grad = min(min_grad, torch.median(torch.abs(param.grad)))
                                max_grad = max(max_grad, mean)
                                if torch.abs(mean) < 1e-8:
                                    very_low_gradients.add('main_'+name)
                        else:
                            print(f'NAN!!!------------------- {name},{batch_num}')
                        
                    for name, param in model_aux.named_parameters():
                        tb.add_histogram('aux_'+name, param, batch_num)
                        mean = torch.mean(param.grad)
                        if not torch.isnan(mean):
                            tb.add_histogram(f'aux_{name}.grad', param.grad, batch_num)
                            if c.DEBUG_FIND_MIN_MEDIAN_GRAD_PER_BATCH:
                                min_grad = min(min_grad, torch.median(torch.abs(param.grad)))
                                max_grad = max(max_grad, mean)
                                if torch.abs(mean) < 1e-8:
                                    very_low_gradients.add('aux_'+name)
                        else:
                            print(f'NAN!!!------------------- {name},{batch_num}')
                        
                #     for name, hook in hooks.items():
                #         hook.remove()
                
                # for name,param in model_main.named_parameters():
                #     if param.grad is None or not torch.isfinite(torch.mean(param.grad)):
                #         print(f'!!!!!!!!!!!!!!!!!!NAN!!!main {name}')
                
                # for name,param in model_aux.named_parameters():
                #     if param.grad is None or not torch.isfinite(torch.mean(param.grad)):
                #         print(f'!!!!!!!!!!!!!!!!!!NAN!!!aux {name}')
                        
                if (batch_num - batch_num_last_accumulate_rate_update) % accumulation_rate == 0:
                    optimizer.zero_grad(set_to_none=True)
                
                batch_training_end_time_prev = batch_training_end_time
                batch_training_end_time = time.time()
                training_batch_time = batch_training_end_time - batch_training_start_time
                entire_batch_loop_time = batch_training_end_time - batch_training_end_time_prev
                
                # print(f'epoch {epoch}, batch {batch_num}, loss: {train_loss_pre_accumulation:.4f};   training time: {training_batch_time:.3f}, entire loop time: {entire_batch_loop_time:.3f}, ratio: {(training_batch_time/entire_batch_loop_time):.3f}')                
                if (batch_num-1) % c.EVAL_FREQUENCY == 0:
                    print(f'epoch {epoch}, batch {batch_num}, training time: {training_batch_time:.3f}, entire loop time: {entire_batch_loop_time:.3f}, ratio: {(training_batch_time/entire_batch_loop_time):.3f}')
                
                if (batch_num - batch_num_last_accumulate_rate_update) % accumulation_rate == 0:
                    running_loss /= accumulation_rate
                    num_batches_since_min_loss = trainer_helper.update_loss_possibly_save_model(running_loss, model_main, model_aux, ema_model_main, ema_model_aux, ema.was_i_initialized, optimizer, scaler, batch_num, accumulation_rate, save_from_this_batch_num=1000)
                    if num_batches_since_min_loss > 5000:
                        # if num_batches_since_min_loss > 100000:
                        #     termination_msg = 'Loss has not improved for 100,000 batches. Terminating the flow.'
                        #     log_file.write(termination_msg+'\n')
                        #     log_file.flush()
                        #     sys.exit(termination_msg)
                        # If the loss hasn't been reduced for this long, increase the accumulation rate.
                        if accumulation_rate < c.MAX_ACCUMULATION_RATE and trainer_helper.num_batches_since_last_accumulation_rate_increase(batch_num) > 5000:
                            accumulation_rate *= 2
                            trainer_helper.update_last_accumulation_rate_increase(batch_num)
                            scaler = torch.cuda.amp.GradScaler()
                            batch_num_last_accumulate_rate_update = batch_num
                            accumulation_msg = f'-----Accumulation rate increased: {accumulation_rate}, effective batch size: {accumulation_rate * c.BATCH_SIZE}\n'
                            log_file.write(accumulation_msg)
                            print(accumulation_msg)
                            # Fake a learning rate reduction so that one isn't made for another 5000 batches.
                            # trainer_helper.update_last_learning_rate_reduction(batch_num)
                            
                        # If the accumulation rate was already increased, reduce the learning rate.
                        # Commenting this out since it never seems to help.
                        # if trainer_helper.num_batches_since_last_learning_rate_reduction(batch_num) > 10000:
                        #     reduction_rate = math.sqrt(10) # divide learning rate by sqrt(10)
                        #     for g in optimizer.param_groups:
                        #         g['lr'] /= reduction_rate
                        #     trainer_helper.update_last_learning_rate_reduction(batch_num)
                        #     lr_reduction_msg = f'-----------------------LR REDUCTION: {g["lr"]}, at batch num: {batch_num}\n'
                        #     print(lr_reduction_msg)
                        #     log_file.write(lr_reduction_msg+'\n')
                    elif num_batches_since_min_loss == 0:
                        loss_decrease_msg = f'---LOSS DECREASED for epoch {epoch}, batch {batch_num}: {running_loss:.3f}, training time: {training_batch_time:.3f}, entire loop time: {entire_batch_loop_time:.3f}, ratio: {(training_batch_time/entire_batch_loop_time):.3f}'
                        print(loss_decrease_msg)
                        log_file.write(loss_decrease_msg+'\n')

                # Save generated images.
                if batch_num % c.EVAL_FREQUENCY == 0:
                    try:
                        clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = next(valid_dataloader_iterator)
                    except StopIteration:
                        valid_dataloader_iterator = iter(valid_dataloader)
                        clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = next(valid_dataloader_iterator)
                    if clothing_aug.shape[0] > 2:
                        clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = clothing_aug[[0,2]], mask_coords[[0,2]], masked_aug[[0,2]], person[[0,2]], pose_vector[[0,2]], pose_matrix[[0,2]], sample_original_string_id, sample_unique_string_id, noise_amount_clothing[[0,2]], noise_amount_masked[[0,2]]
                    num_eval_samples = clothing_aug.shape[0]
                    if not c.USE_AMP:
                        inputs = [clothing_aug.cuda().float(), mask_coords.cuda().float(), masked_aug.cuda().float(), person.cuda().float(), pose_vector.cuda().float(), pose_matrix.cuda().float(), sample_original_string_id, sample_unique_string_id, noise_amount_clothing.cuda().float(), noise_amount_masked.cuda().float()]
                    else:
                        if not c.USE_BFLOAT16:
                            inputs = [clothing_aug.cuda().to(torch.float16), mask_coords.cuda(), masked_aug.cuda().to(torch.float16), person.cuda().to(torch.float16), pose_vector.cuda().to(torch.float16), pose_matrix.cuda().to(torch.float16), sample_original_string_id, sample_unique_string_id, noise_amount_clothing.cuda().to(torch.float16), noise_amount_masked.cuda().to(torch.float16)]
                        else:
                            inputs = [clothing_aug.cuda(), mask_coords.cuda(), masked_aug.cuda(), person.cuda(), pose_vector.cuda(), pose_matrix.cuda(), sample_original_string_id, sample_unique_string_id, noise_amount_clothing.cuda(), noise_amount_masked.cuda()]
                        
                    val_loss = 0
                    for eval_mode,eval_mode_id in [(True, 'with_cfg'), (False, 'no_cfg')]:
                        for model_main_,model_aux_,suffix in [(model_main, model_aux, '_no_ema'), (ema_model_main, ema_model_aux, '_with_ema')]:
                            if suffix == '_with_ema' and (not c.RUN_EMA or batch_num < ema_batch_num_start):
                                continue
                            if eval_mode == True and not c.USE_CLASSIFIER_FREE_GUIDANCE:
                                continue
                            if c.REVERSE_DIFFUSION_SAMPLER == 'karras':
                                img_sequences = p_sample_loop_karras(sample_euler_ancestral_karras, model_main_, model_aux_, inputs, steps=c.NUM_DIFFUSION_TIMESTEPS)
                            else:
                                img_sequences = p_sample_loop(model_main_, model_aux_, inputs, shape=(num_eval_samples, 3, img_height, img_width), eval_mode=eval_mode)
                            for i,img in enumerate(img_sequences[-1]):
                                if c.REVERSE_DIFFUSION_SAMPLER == 'karras':
                                    img = img.clamp(-1,1)
                                if suffix == '_no_ema':
                                    val_loss += F.l1_loss(img.cpu(), person[i]).item()
                                img = denormalize_img(img)
                                full_string_identifier = sample_original_string_id[i] + '_' + str(sample_unique_string_id[i])
                                save_image(img, os.path.join(c.MODEL_OUTPUT_IMAGES_DIR, f'sample-{batch_num}_{i}{suffix}-{eval_mode_id}_PRED.png'), nrow = 4//2)
                                if suffix == '_no_ema' and eval_mode_id == 'no_cfg':
                                    masked_img = (((masked_aug[i].to(dtype=torch.float16).numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
                                    person_img = (((person[i].to(dtype=torch.float16).numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
                                    clothing_img = (((clothing_aug[i].to(dtype=torch.float16).numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
                                    pose_img = save_or_return_img_w_overlaid_keypoints(person_img.copy(), pose_vector[i], return_value=True)
                                    cv2.imwrite(os.path.join(c.MODEL_OUTPUT_IMAGES_DIR, f'sample-{batch_num}_{i}{suffix}-{eval_mode_id}-{full_string_identifier}_masked.png'), masked_img)
                                    cv2.imwrite(os.path.join(c.MODEL_OUTPUT_IMAGES_DIR, f'sample-{batch_num}_{i}{suffix}-{eval_mode_id}-{full_string_identifier}_person.png'), person_img)
                                    cv2.imwrite(os.path.join(c.MODEL_OUTPUT_IMAGES_DIR, f'sample-{batch_num}_{i}{suffix}-{eval_mode_id}-{full_string_identifier}_clothing.png'), clothing_img)
                                    cv2.imwrite(os.path.join(c.MODEL_OUTPUT_IMAGES_DIR, f'sample-{batch_num}_{i}{suffix}-{eval_mode_id}-{full_string_identifier}_pose.png'), pose_img)
                    val_loss /= num_eval_samples
                    tb.add_scalar('val loss', val_loss, batch_num)
                if (batch_num - batch_num_last_accumulate_rate_update) % accumulation_rate == 0:
                    tb.add_scalar('train loss', running_loss, batch_num)
                    if batch_num - last_loss_print > 500:
                        print(f'epoch {epoch}, batch {batch_num}, loss: {running_loss:.4f};   training time: {training_batch_time:.3f}, entire loop time: {entire_batch_loop_time:.3f}, ratio: {(training_batch_time/entire_batch_loop_time):.3f}')
                        last_loss_print = batch_num
                    running_loss = 0
                if c.DEBUG_FIND_MIN_MEDIAN_GRAD_PER_BATCH and max_grad != 0:
                    print(f'batch {batch_num}, min max grad: {min_grad}, {max_grad}')
                    print(very_low_gradients)        
    tb.close()
