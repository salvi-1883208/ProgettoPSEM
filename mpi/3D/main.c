// MPI implementation of a DLA simulation (3D)
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

void move_particle(int *x, int *y, int *z, int m);
int write_matrix_to_file(bool *matrix, int dim);
int is_close_to_stuck(bool *grid, int size, int x, int y, int z, int process_count, int my_rank, MPI_Win *win);
int main(int argc, char **argv) {
    // variables for the timing of the execution
    double start, end;

    // variables for the MPI environment
    int process_count, my_rank;

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // simulation parameters
    int grid_size, particles, max_steps, si, sj, sk, random_seed;

    // the grid for each process
    bool *my_grid;

    // get the grid size from the command line
    if (my_rank == 0) {
        // command line input: grid size, number of particles, number of steps, seed coordinates, random seed
        if ((argc - 1) < 6) {
            printf("Arguments are: cube matrix size, number of particles, number of maximum steps, seed coordinates, seed for the rand() function.\n\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        // get grid size from args
        grid_size = atoi(argv[1]) | 1;

        // get number of particles sent from args
        particles = atoi(argv[2]);

        // get number of iterations for each particle from args
        max_steps = atoi(argv[3]);

        // get seed coordinates from args
        si = atoi(argv[4]) - 1;
        sj = atoi(argv[5]) - 1;
        sk = atoi(argv[6]) - 1;

        // if the random seed is given from the command line arguments
        if (argc == 8)
            // get seed for the rand() function from args
            random_seed = atoi(argv[7]);
        else
            // default seed
            random_seed = 3521;

        // if given out image coordinates place seed in the middle
        if (si < 0 || sj < 0 || sk < 0 || si > grid_size || sj > grid_size || sk > grid_size) {
            printf("Given outside of image seed coordinates.\n");
            printf("Setting seed coordinates to %d, %d, %d.\n", grid_size / 2, grid_size / 2, grid_size / 2);
            si = (grid_size - 1) / 2;
            sj = (grid_size - 1) / 2;
            sk = (grid_size - 1) / 2;
        }

        // send the parameters to the other processes
        for (int p = 1; p < process_count; p++) {
            MPI_Send(&grid_size, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
            MPI_Send(&particles, 1, MPI_INT, p, 1, MPI_COMM_WORLD);
            MPI_Send(&max_steps, 1, MPI_INT, p, 2, MPI_COMM_WORLD);
            MPI_Send(&si, 1, MPI_INT, p, 3, MPI_COMM_WORLD);
            MPI_Send(&sj, 1, MPI_INT, p, 4, MPI_COMM_WORLD);
            MPI_Send(&sk, 1, MPI_INT, p, 5, MPI_COMM_WORLD);
            MPI_Send(&random_seed, 1, MPI_INT, p, 6, MPI_COMM_WORLD);
        }

        printf("\nSimulating growth...\n");
    } else {
        // receive the parameters from the master process
        MPI_Recv(&grid_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&particles, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&max_steps, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&si, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&sj, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&sk, 1, MPI_INT, 0, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&random_seed, 1, MPI_INT, 0, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // create a window for the grid and allocate it
    MPI_Win win;
    MPI_Win_allocate(grid_size * grid_size * grid_size * sizeof(bool), sizeof(bool), MPI_INFO_NULL, MPI_COMM_WORLD, &my_grid, &win);

    // place the seed particle
    my_grid[si * grid_size * grid_size + sj * grid_size + sk] = 1;

    // Initialize the random number generator
    srand(my_rank * random_seed);

    // divide the particles between the processes
    int my_particles = particles / process_count;
    if (my_rank == process_count - 1)
        my_particles += particles % process_count;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    for (int g = 0; g < my_particles; g++) {
        // count my number of iterations
        int my_i = 0;

        int my_x, my_y, my_z;

        // generate a random particle
        // at this point the grid should be empty
        do {
            my_x = rand() % grid_size;
            my_y = rand() % grid_size;
            my_z = rand() % grid_size;
            my_i++;
        } while (my_grid[my_x * grid_size * grid_size + my_y * grid_size + my_z] && my_i < max_steps);

        while (my_i < max_steps) {
            // if the particle is outside the grid move it back in
            if (my_x < 1)
                my_x = 1;
            else if (my_x > grid_size - 2)
                my_x = grid_size - 2;
            if (my_y < 1)
                my_y = 1;
            else if (my_y > grid_size - 2)
                my_y = grid_size - 2;
            if (my_z < 1)
                my_z = 1;
            else if (my_z > grid_size - 2)
                my_z = grid_size - 2;

            // if the particle is close to an already stuck particle
            if (is_close_to_stuck(my_grid, grid_size, my_x, my_y, my_z, process_count, my_rank, &win))
                break;

            // move the particle in a random direction
            move_particle(&my_x, &my_y, &my_z, rand() % 26);

            // increment the number of iterations for each time the move is made
            my_i++;
        }
    }
    // variable to store the time of the last process to finish
    double last_process = 0;

    // time process end
    end = MPI_Wtime();

    // see who is the last process to finish
    MPI_Allreduce(&end, &last_process, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // if the last process, save the image
    if (end == last_process) {
        // it has to be the last process because of how i implemented the RMA communication during the diffusion
        // (the last process to finish is the one that has the most up to date grid)
        int stuck = write_matrix_to_file(my_grid, grid_size);
        MPI_Send(&stuck, 1, MPI_INT, 0, 8, MPI_COMM_WORLD);
    }

    if (my_rank == 0) {
        // recieve and print the time of execution for each process
        printf("\nCPU time for each processor:\n");
        printf(" - Rank %d: %f seconds\n", my_rank, (end - start));
        for (int p = 1; p < process_count; p++) {
            MPI_Recv(&end, 1, MPI_DOUBLE, p, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf(" - Rank %d: %f seconds\n", p, (end - start));
        }
        // print the total time of execution (the time of the last process to finish)
        printf("Total time of execution in seconds: %f\n\n", last_process - start);
        // recieve and print the number of stuck particles from the last process
        int stuck;
        MPI_Recv(&stuck, 1, MPI_INT, MPI_ANY_SOURCE, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Of %d particles:\n - drawn %d,\n - skipped %d.\n\n", particles, stuck, particles - stuck);
    } else
        // send the time of execution to the master process
        MPI_Send(&end, 1, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);

    // free the memory allocated for the grid
    MPI_Win_free(&win);

    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}

int is_close_to_stuck(bool *grid, int size, int x, int y, int z, int process_count, int my_rank, MPI_Win *win) {
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            for (int k = -1; k <= 1; k++)
                if (grid[(x + i) * size * size + (y + j) * size + (z + k)]) {
                    // if the particle is close to an already stuck particle, attach it to the grid
                    grid[x * size * size + y * size + z] = 1;

                    // increment the number of stuck particles in this position for all processes
                    for (int p = 0; p < process_count; p++)
                        if (p != my_rank) {
                            // lock the window for the process p
                            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, p, 0, *win);
                            // make this particle stuck in this position for the process (replace)
                            MPI_Accumulate(&grid[x * size * size + y * size + z], 1, MPI_C_BOOL, p, x * size * size + y * size + z, 1, MPI_C_BOOL, MPI_REPLACE, *win);
                            // flush and unlock the window for the process p
                            MPI_Win_flush(p, *win);
                            MPI_Win_unlock(p, *win);
                        }
                    // finish the diffusion for this particle
                    return 1;
                }
    return 0;
}

// define the offsets for the 26 directions
const int dx[26] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0};
const int dy[26] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0, 1, 1, 1, 1, 1, 1};
const int dz[26] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1};

// move the particle in the random direction (out of 26)
void move_particle(int *x, int *y, int *z, int m) {
    (*x) += dx[m];
    (*y) += dy[m];
    (*z) += dz[m];
}

// save the grid to a file
int write_matrix_to_file(bool *matrix, int dim) {
    int count = 0;
    FILE *fp = fopen("matrix.txt", "w");
    if (fp == NULL) {
        printf("Error opening file %s\n", "matrix.txt");
        exit(1);
    }
    fprintf(fp, "%d\n", dim);

    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            for (int k = 0; k < dim; k++)
                if (matrix[(i * dim * dim) + (j * dim) + k]) {
                    fprintf(fp, "%d %d %d\n", i, j, k);
                    count++;
                }

    fclose(fp);
    return count;
}