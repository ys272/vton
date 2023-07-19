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

current_timestamp = time.time()
current_date = datetime.fromtimestamp(current_timestamp).strftime("%d-%m-%y")
tb = SummaryWriter(log_dir=f'/home/yoni/Desktop/f/model_output/tboard/{current_date + "-" + str(current_timestamp).split(".")[0]}')

results_folder = Path("/home/yoni/Desktop/f/model_output/images/")
results_folder.mkdir(exist_ok = True)
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
optimizer = Adam(model.parameters(), lr=1e-3, eps=1e-5) # epsilon increased a la fast.ai optimization

# x = torch.randn(c.BATCH_SIZE, num_channels, image_size, image_size, device=c.DEVICE)
# t = torch.randint(0, c.NUM_TIMESTEPS, (c.BATCH_SIZE,), device=c.DEVICE).long()
# output = model(x,t)
# print(output.size())
# make_dot(model(x,t), params=dict(model.named_parameters())).render("/home/yoni/Desktop/fash_model", format="png")


images, labels = next(iter(train_loader))
images = images.to(c.DEVICE)
grid = torchvision.utils.make_grid(images)
tb.add_image('images', grid)
t = torch.randint(0, c.NUM_TIMESTEPS, (c.BATCH_SIZE,), device=c.DEVICE).long()
input_tuple = (images, t)
tb.add_graph(model, input_tuple)


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


epochs = 60
for epoch in range(epochs):
    for step, batch in enumerate(train_loader):
        if step>3:
            break
        
        optimizer.zero_grad()

        batch = batch[0].to(c.DEVICE)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, c.NUM_TIMESTEPS, (c.BATCH_SIZE,), device=c.DEVICE).long()

        # Everything that isn't an identity (residual) convolution should be zeroed in upsampling and downsampling processing path.
        # Downsampling convolutions should be initialized orthogonally.
        # Final layer should also be zerod out.
        for d in model.downs:
            # first two components are resnet blocks
            # for resnet_block in d[:2]: 
            #     resnet_block.block1.proj.weight.data.zero_()
            #     resnet_block.block2.proj.weight.data.zero_()
            # # third component is attention
            # attention = d[2]
            # attention.fn.fn.to_qkv.weight.data.zero_()
            # attention.fn.fn.to_out[0].weight.data.zero_()
            # # fourth component is downsampling, which includes residual identity 
            downsample = d[3]
            if not isinstance(downsample,torch.nn.modules.conv.Conv2d):
                init.orthogonal_(downsample[1].weight)
                      
        model.final_conv.weight.data.zero_()
        
        loss = p_losses(model, batch, t, loss_type="huber")

        if step % 100 == 0:
            print(f'Loss for epoch {epoch}, step {step}:', loss.item())

        loss.backward()
        optimizer.step()

        # save generated images
        if step != 0 and step % save_and_sample_every == 0:
            img_sequences = p_sample_loop(model, shape=(4, num_channels, image_size, image_size))
            for i,img in enumerate(img_sequences[-1]):
                img = (img + 1) * 0.5
                save_image(torch.from_numpy(img), str(results_folder / f'sample-{epoch}_{step}_final.png'), nrow = 4//2)
    
    tb.add_scalar('Train loss', loss, epoch)
    # tb.add_scalar('Val loss', val_loss, epoch)
    for name, param in model.named_parameters():
      tb.add_histogram(name, param, epoch)
      tb.add_histogram(f'{name}.grad', param.grad, epoch)
    
    
tb.close()