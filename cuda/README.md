# Cuda version

Compile with `nvcc main.cu -o main`.
<br>
Execute with `./main size, particles, iterations, seed_x, seed_y, block_size, random_seed`
- `size` is the size of the image. The image is always a square.
- `particles` is the number of particles to be fired in the simulation times the `block_size`.
- `iterations` is the maximum number of iterations a single particle is permitted to do, after wich it is skipped. If set to `0` the limit is removed and every particle stops ony when it becomes stuck.
- `seed_x` and `seed_y` are the coordinates for the initial stuck particle from wich the structure will generate. If given outside of the image coordinates, the center of the image is set.
- `block_size` is the number of threads in each block. If not given, `block_size` is set to `1024`, the maximum value for the GTX 1080.
- `random_seed` is the seed for the `curand()` function. If not given the seed is set to `3521`.

<br> Example of the image generated with `./main 1000 64 0 -1 0`.
<br> <br> 
<img src="https://user-images.githubusercontent.com/62235561/220726014-4e7b788e-f8aa-4f72-831c-9ee65a23feef.png" title="Example image">