# 3D MPI version

Compile with `mpicc -g -Wall -o main main.c`.
<br>
Execute with `mpirun -np processes ./main size, particles, iterations, seed_x, seed_y, seed_z, random_seed`
- `processes` is the number of processes to be used for the computation.
- `size` is the size of the space. The space is always a cube. To render the structure better the size is made always odd.
- `particles` is the number of particles to be fired in the simulation.
- `iterations` is the maximum number of iterations a single particle is permitted to do, after wich it is skipped.
- `seed_x`,  `seed_y` and `seed_z` are the coordinates for the initial stuck particle from wich the structure will generate. If given outside of the space coordinates, the center of the space is set.
- `random_seed` is the seed for the `rand()` function. If not given the seed is set to `3521`.

<br> The execution of the program gives as output a `matrix.txt` file wich contains the 3D matrix used to render the structure.
<br> To render the structure in the `matrix.txt` file see the [3D renderer](/3DRender/).

<br> Example of the structure generated with `mpirun -np 6 ./main 100 10000 1000000 0 0 0`.
<br> <br> 
<img src="https://user-images.githubusercontent.com/62235561/222979675-0867b7c5-ecc7-48ea-95ac-2b7851469c1a.png" title="Structure example">