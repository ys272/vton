def init_model(model):
  # Everything that isn't an identity (residual) convolution should be zeroed in upsampling and downsampling processing path.
  # Downsampling convolutions should be initialized orthogonally.
  # Final layer should also be zerod out.
  for d in model.downs:
      # first two components are resnet blocks
      for resnet_block in d[:2]: 
          resnet_block.block1.proj.weight.data.zero_()
          resnet_block.block2.proj.weight.data.zero_()
      # third component is attention
      attention = d[2]
      attention.fn.fn.to_qkv.weight.data.zero_()
      attention.fn.fn.to_out[0].weight.data.zero_()
      # fourth component is downsampling, which includes residual identity 
      downsample = d[3]
      if not isinstance(downsample,torch.nn.modules.conv.Conv2d):
          init.orthogonal_(downsample[1].weight)
  model.final_conv.weight.data.zero_()
  return model