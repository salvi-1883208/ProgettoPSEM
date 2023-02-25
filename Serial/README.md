# Serial version (optimized)

Compile with `gcc main.c -Wall -lm -o main`.
<br>
Execute with `./main size, particles, iterations, seed_x, seed_y, close_radius, random_seed`
- `size` is the size of the image. The image is always a square.
- `particles` is the number of particles to be fired in the simulation.
- `iterations` is the maximum number of iterations a single particle is permitted to do, after wich it is skipped. If set to `0` the limit is removed and every particle stops ony when it becomes stuck.
- `seed_x` and `seed_y` are the coordinates for the initial stuck particle from wich the structure will generate. If given outside of the image coordinates, the center of the image is set.
- `close_radius` is the radius of the circle around the structure from wich the particle are generated and can not go out. If not given it is set to `size / 5`.
- `random_seed` is the seed for the `rand()` function. If not given the seed is set to `3521`.

<br> Example of the image generated with `./main 1000 60000 0 -1 0 20 1234`.
<br> <br> 
<img src="https://user-images.githubusercontent.com/62235561/208951309-fe9ac857-e6bc-48fa-8221-046bfc7efc88.jpg" title="Example image">

<br> <br>
The optimization used for this version consists in remembering the maximum distance from a stuck particle and the seed, and using it to create a bounding circle around the structure. All the new particles are generated on this bounding circle and can't go outside it. This optimization speeds up the simulation significantly, making the time of execution not in function of the size of the image (`size`), but only of the number of particles to simulate (`particles`) and the maximum number of steps a particle can do (`iterations`). 
<br>
<br> The optimization changes the structrure a little, making it less dense. This is because the particles are generate always at `close_radius` distance from the structure, whereas in the unoptimized version, the particles are generated in a random position in the image, making it more dense. The density becomes more similar to the unoptimized version the smaller is the `close_radius` parameter is. <br>
Another difference from the unoptimized version is that when the maximum number of steps (`iterations`) is not high enough, the structure becomes biased in a certain direction, developing only one branch. This happens because the particles that are not generate close to that branch are unlikely to hit the structure, enforcing the bias.
<br> <br>
Example of a biased structure: <br> <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Image generated with `./main 1000 60000 1000 -1 0 20` <br> <br>
<img src="https://user-images.githubusercontent.com/62235561/221373938-d2a0f326-99cf-4128-b3f5-ac5131ef4ee7.png" title="Biased structure">
