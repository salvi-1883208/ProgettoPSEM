# Serial version (not optimized)

Compile with `gcc main.c -Wall -lm -o main`.
<br>
Execute with `./main size, particles, iterations, seed_x, seed_y, random_seed`
- `size` is the size of the image. The image is always a square.
- `particles` is the number of particles to be fired in the simulation.
- `iterations` is the maximum number of iterations a single particle is permitted to do, after wich it is skipped. If set to `0` the limit is removed and every particle stops ony when it becomes stuck.
- `seed_x` and `seed_y` are the coordinates for the initial stuck particle from wich the structure will generate. If given outside of the image coordinates, the center of the image is set.
- `random_seed` is the seed for the `rand()` function. If not given the seed is set to `3521`.

<br> Example of the image generated with `./main 700 30000 0 -1 0`.
<br> <br> 
<img src="https://user-images.githubusercontent.com/62235561/220996882-1d3b86b7-cba0-4119-a20c-ef0e6e8e88b7.png" title="Optional title">
