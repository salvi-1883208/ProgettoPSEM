# 3D Cuda version

Compile with `nvcc main.cu -o main`.
<br>
Execute with `./main size, particles, iterations, seed_x, seed_y, seed_z, block_size, random_seed`
- `size` is the size of the space. The space is always a cube. To render the structure better the size is made always odd.
- `particles` is the number of particles to be fired in the simulation, times the `block_size`.
- `iterations` is the maximum number of iterations a single particle is permitted to do, after wich it is skipped. If set to `0` the limit is removed and every particle stops ony when it becomes stuck.
- `seed_x`,  `seed_y` and `seed_z` are the coordinates for the initial stuck particle from wich the structure will generate. If given outside of the image coordinates, the center of the space is set.
- `block_size` is the number of threads in each block. If not given, `block_size` is set to `1024`, the maximum value for the GTX 1080.
- `random_seed` is the seed for the `curand()` function. If not given the seed is set to `3521`.

<br> The execution of the program gives as output a `matrix.txt` file wich contains the 3D matrix used to render the structure.
<br> To render the structure in the `matrix.txt` file see the [3D renderer](/3DRender/).

<br> Example of the structure generated with `./main 300 64 100000 -1 0 0`.
<br> <br> 
<img src="https://user-images.githubusercontent.com/62235561/221424232-a91a6f6c-cb64-4e7f-883e-9d159494b21c.png" title="Structure example">