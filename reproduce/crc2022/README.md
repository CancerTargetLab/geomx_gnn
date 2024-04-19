# CRC Atlas 2022

Follow the instructions to intall required packages and become familiar Image2Count.  

Lets start by creating a directory to download images into and then download the data we used(~1.4TB):
```sh
mkdir data/raw/CRC
cd data/raw/CRC
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC02.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC03.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC04.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC05.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC06.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC07.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC08.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC09.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC10.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC11.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC12.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC13.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC14.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC15.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC16.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC17.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC02-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC03-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC04-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC05-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC06-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC07-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC08-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC09-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC10-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC11-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC12-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC13-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC14-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC15-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC16-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC17-features.zip
wget "http://lin-2021-crc-atlas.s3.amazonaws.com/metadata/CRC202105 HTAN channel metadata.csv"
mkdir metadata
mkdir spatial
mv *metadata.csv metadata
cd ../../../
```

As a next step we create the `label.csv` and `measurements.csv` files and clean up the directory:
```sh
python -m src/utils/crc_label_and_pos_data.py
mv data/raw/CRC/*features* data/raw/CRC/spatial
mv data/raw/CRC/CRC_label.csv data/raw/
```
If your setup is different to the one described here you might need to modif where to load data from and where to save it to in the utils script.  


If we want we can now also remove outlier cells with total counts in the top or bottom 1% and setup a subdir for this data:
```sh
mkdir data/raw/CRC_1p
ln -s /path/to/data/raw/CRC/*.tif data/raw/CRC_1p
python -m src/utils/crc_rm_outliers.py
mv data/raw/CRC_1p_measurements.csv data/raw/CRC_1p/
```
If paths differ from how described here, or another percentage threshold should be used, adjust the utils script.  

We can now progress and preprocess the images by ZScore normalisation and cutting out cells:
```sh
python -m main --image_preprocess --preprocess_dir data/raw/CRC/ --cell_cutout 34 --preprocess_workers 26 --preprocess_channels 0,10,14,19
```
The channels that we investigate are the first DNA channel, CD45, Keratin and CD8a. To see what channel is used for what marker, look at the metadata in `data/raw/CRC/metadata/`, markers that failed QC are not included in the Image, as well as 3 controll markers.  

If we have also removed outliers:  

```sh
python -m main --image_preprocess --preprocess_dir data/raw/CRC_1p/ --cell_cutout 34 --preprocess_workers 26 --preprocess_channels 0,10,14,19 --preprocess_mean_std_dir 'data/raw/CRC/'
```

