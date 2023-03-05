# MPI version

Compile with `mpicc -g -Wall -o main main.c`.
<br>
Execute with `mpirun -np processes ./main size, particles, iterations, seed_x, seed_y, random_seed`
- `processes` is the number of processes to be used for the computation.
- `size` is the size of the image. The image is always a square.
- `particles` is the number of particles to be fired in the simulation.
- `iterations` is the maximum number of iterations a single particle is permitted to do, after wich it is skipped.
- `seed_x` and `seed_y` are the coordinates for the initial stuck particle from wich the structure will generate. If given outside of the image coordinates, the center of the image is set.
- `random_seed` is the seed for the `rand()` function. If not given the seed is set to `3521`.

<br> Example of the image generated with `.mpirun -np 6 ./main 700 30000 1000000 0 0`.
<br> <br> 
<img src="https://user-images.githubusercontent.com/62235561/222978154-f71172f3-a771-4efe-b0f9-6cb8b8c309c7.png" title="Example image">

<br> <br>
In this particular implementation, if two or more particles get stuck in the same position, only one is considered stuck and the others are considered skipped.
