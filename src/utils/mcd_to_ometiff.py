from imctools.io.mcd.mcdparser import McdParser
from tqdm import tqdm
import os
import numpy as np

path = '/home/mjheid/Downloads/7618945/data'
path_out = '/home/mjheid/Downloads/7618945/data/tiff/'

if not (os.path.exists(path_out) and os.path.isdir(path_out)):
    os.makedirs(path_out)

mcd_files = [os.path.join(path, p) for p in os.listdir(path) if p.endswith('.mcd')]

names = None
labels = None
for mcd in mcd_files:
    parser = McdParser(mcd)
    xml = parser.get_mcd_xml()
    ids = parser.session.acquisition_ids
    with tqdm(ids, desc=f'Converting {mcd} to .ome.tiff', total=len(ids)) as ids:
        for id in ids:
            ac_data = parser.get_acquisition_data(id)
            if names is None:
                names = ac_data.channel_names
                labels = ac_data.channel_labels
            if ac_data.is_valid:
                ac_data.save_ome_tiff(os.path.join(path_out,
                                    f'{id:03d}_'+ac_data.acquisition.metaname+'_'+ac_data.acquisition.description+'.ome.tiff'),
                                    names=names,
                                    xml_metadata=xml)

with open(os.path.join(path_out, 'channel_labels.csv'), mode='x') as file:
    file.writelines('channel_labels,\n')
    for l in labels:
        file.writelines(l+',\n')
    file.close()
