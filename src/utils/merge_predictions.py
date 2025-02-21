import torch
import os

def merge(save_dir):
    result_dirs = [result_dir for result_dir in  os.listdir(save_dir)  if result_dir != 'merged' and result_dir != 'mean' and not '_' in result_dir and not result_dir.endswith('.pt')]
    result_files = [os.listdir(os.path.join(save_dir, result_dir)) for result_dir in result_dirs]

    list(map(lambda x: x.sort(), result_files))

    if not os.path.exists(os.path.join(save_dir, 'mean')):
        os.makedirs(os.path.join(save_dir, 'mean'))

    for file in range(len(result_files[0])):
        if result_files[0][file].startswith('cell'):
            file_contents = []
            for result_dir in range(len(result_dirs)):
                file_contents.append(torch.load(os.path.join(save_dir, result_dirs[result_dir], result_files[result_dir][file]), weights_only=True, map_location='cpu'))
            merged = torch.zeros((len(result_files), file_contents[0].shape[0], file_contents[0].shape[1]))
            for i in range(len(result_files)):
                merged[i] = file_contents[i]
            merged = torch.mean(merged, dim=0, keepdim=True)[0].squeeze()
            torch.save(merged, os.path.join(save_dir, 'mean', result_files[0][file]))
            torch.save(torch.sum(merged, dim=0), os.path.join(save_dir, 'mean', result_files[0][file]).replace('cell_', 'roi_'))

