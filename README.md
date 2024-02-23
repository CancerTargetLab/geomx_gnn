installation:
conda squidpy doesnt work? takers ages
install via pip
important: some qt5 packes are not linked, packaged, debug as follows:
export QT_DEBUG_PLUGINS=1
python -m your_script_with_squidpy_loading
find lib that is missing 
ldd path/to/lib | grep -i "Not found"
install missing lib, google how to :D


Next problem: libGL error: 
MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory
Answer: https://askubuntu.com/questions/1451610/mesa-loader-failed-to-open-swrast-ubuntu-20-04  
Install mesa-utils and libgl1-mesa-dri  
link driver: sudo ln -s /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so /usr/lib/dri/  

https://stackoverflow.com/questions/58424974/anaconda-importerror-usr-lib64-libstdc-so-6-version-glibcxx-3-4-21-not-fo  


https://stackoverflow.com/questions/69952475/how-to-solve-the-pytorch-geometric-install-error-undefined-symbol-zn5torch3ji/73876857#73876857  
https://github.com/pyg-team/pytorch_geometric/issues/999 

```
install:
conda install -c anaconda python=3.11.5  
pip install torch -f https://data.pyg.org/whl/torch-2.1.0+cu121.html  
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html  
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html  
pip install torchvision -f https://data.pyg.org/whl/torch-2.1.0+cu121.html  
```