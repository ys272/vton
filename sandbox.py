import torch
with open('/home/yoni/Desktop/f/test/ready_data/schp_raw_output/densepose/densepose.pkl', 'rb') as f:
    data = torch.load(f)
    for densefile in data:
      filename = densefile['file_name'].split('/')[-1].split('.')[0]
      torch.save(densefile, f'/home/yoni/Desktop/f/test/ready_data/schp_raw_output/densepose/{filename}.pkl')