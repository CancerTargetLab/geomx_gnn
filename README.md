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


