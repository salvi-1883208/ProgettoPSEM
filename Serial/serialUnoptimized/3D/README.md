# 3D Serial version (not optimized)

Compile with `gcc main.c -Wall -lm -o main`.
<br>
Execute with `./main size, particles, iterations, seed_x, seed_y, seed_z, random_seed`
- `size` is the size of the image. The image is always a square. To render the structure better the size is made always odd.
- `particles` is the number of particles to be fired in the simulation.
- `iterations` is the maximum number of iterations a single particle is permitted to do, after wich it is skipped. If set to `0` the limit is removed and every particle stops ony when it becomes stuck.
- `seed_x`, `seed_y` and `seed_z` are the coordinates for the initial stuck particle from wich the structure will generate. If given outside of the space coordinates, the center of the space is set.
- `random_seed` is the seed for the `rand()` function. If not given the seed is set to `3521`.

<br> The execution of the program gives as output a `matrix.txt` file wich contains the 3D matrix used to render the structure.
<br> To render the structure in the `matrix.txt` file see the [3D renderer](/3DRender/).

<br> Example of the structure generated with `200 10000 100000 -1 0 0`.
<br> <br> 
<img src="https://user-images.githubusercontent.com/62235561/221370550-ae6b4f67-aac8-400f-b418-3101667db5f1.png" title="Optional title">
