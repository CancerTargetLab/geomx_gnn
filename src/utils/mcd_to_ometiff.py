from imctools.io.mcd.mcdparser import McdParser
from tqdm import tqdm
import os
import tifffile

path = '/home/mjheid/Downloads/mcd_data/'
path_out = '/home/mjheid/geomx_gnn/data/raw/mcd/'

if not (os.path.exists(path_out) and os.path.isdir(path_out)):
    os.makedirs(path_out)

mcd_files = [os.path.join(path, p) for p in os.listdir(path) if p.endswith('.mcd')]

names = None
labels = None
for mcd in mcd_files:
    parser = McdParser(mcd)
    ids = parser.session.acquisition_ids
    with tqdm(ids, desc=f'Converting {mcd} to .ome.tiff', total=len(ids)) as ids:
        for id in ids:
            ac_data = parser.get_acquisition_data(id)
            if names is None:
                names = ac_data.channel_names
                labels = ac_data.channel_labels
            if ac_data.is_valid:
                order = ac_data.acquisition.get_name_indices(names)
                data = ac_data._get_image_stack_cyx(order)
                tifffile.imwrite(os.path.join(path_out,
                                 f'{id:03d}_'+ac_data.acquisition.metaname+'_'+ac_data.acquisition.description+'.ome.tiff'),
                                 data)

with open(os.path.join(path_out, 'channel_labels.csv'), mode='w') as file:
    file.writelines('channel_labels,\n')
    for l in labels:
        file.writelines(l+',\n')
    file.close()
