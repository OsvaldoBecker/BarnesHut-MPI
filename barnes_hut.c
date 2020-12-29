/*
 * File: barnes_hut.c: Implements the Barnes Hut algorithm for n-body
 * simulation with galaxy-like initial conditions.
 */
#include "barnes_hut.h"

// Some MPI and global variables
int delta;
int point;
int procs, rank;
int tag = 10;
MPI_Status status;
MPI_Datatype MPI_STATE_T;

// Some constants and global variables
int total_states;
int time_steps;
const double dt = 1e-3, epsilon = 1e-1, grav = 0.04; // Grav should be 100 / N
struct node_t *root;
struct state_t *states;

/*
 * Main function.
 */
int main(int argc, char *argv[])
{
    // Start MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create MPI type for struct state_t
    int count = 7;
    int blocklengths[7] = {1, 1, 1, 1, 1, 1, 1};
    MPI_Aint displacements[7];
    displacements[0] = offsetof(struct state_t, x);
    displacements[1] = offsetof(struct state_t, y);
    displacements[2] = offsetof(struct state_t, u);
    displacements[3] = offsetof(struct state_t, v);
    displacements[4] = offsetof(struct state_t, force_x);
    displacements[5] = offsetof(struct state_t, force_y);
    displacements[6] = offsetof(struct state_t, mass);
    MPI_Datatype types[7] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};

    MPI_Type_create_struct(count, blocklengths, displacements, types, &MPI_STATE_T);
    MPI_Type_commit(&MPI_STATE_T);

    if (rank == 0)
    {
        //The second argument sets the name of input file
        if (argc > 1)
        {
            char *filename = argv[1];

            if (read_case(filename) == 1)
            {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            if (procs > 1)
            {
                if ((total_states % procs) != 0)
                {
                    printf("Error: Unable to split processing with this input and number of processes.\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }
        else
        {
            printf("Error: The argument with the name of input file is missing.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (procs > 1)
    {
        MPI_Bcast(&time_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&total_states, 1, MPI_INT, 0, MPI_COMM_WORLD);

        delta = total_states / procs;

        if (rank != 0)
        {
            // Initiate memory for the vector
            states = malloc(sizeof(struct state_t) * total_states);

            if (states == NULL)
            {
                printf("Error: Some malloc won't work on rank = %d.\n", rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
    else
    {
        delta = total_states;
    }

    // Allocate memory for root
    root = malloc(sizeof(struct node_t));

    if (root == NULL)
    {
        printf("Error: Some malloc won't work on rank = %d.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // The main loop
    for (int i = 0; i < time_steps; i++)
    {
        time_step();
    }

    if (rank == 0)
    {
        // Compute final statistics
        double vu = 0;
        double vv = 0;
        double sumx = 0;
        double sumy = 0;
        double total_mass = 0;

        for (int i = 0; i < total_states; i++)
        {
            sumx += states[i].mass * states[i].x;
            sumy += states[i].mass * states[i].y;
            vu += states[i].u;
            vv += states[i].v;
            total_mass += states[i].mass;
        }

        double cx = sumx / total_mass;
        double cy = sumy / total_mass;
        print_statistics(vu, vv, cx, cy);
    }

    // Free memory
    free(root);
    free(states);

    MPI_Finalize();
    return 0;
}

/*
 * Updates the positions of the particles of a time step.
 */
void time_step(void)
{
    MPI_Bcast(states, total_states, MPI_STATE_T, 0, MPI_COMM_WORLD);

    set_node(root);
    root->min_x = 0;
    root->max_x = 1;
    root->min_y = 0;
    root->max_y = 1;

    // Put particles in tree
    for (int i = 0; i < total_states; i++)
    {
        put_particle_in_tree(i, root);
    }

    // Calculate mass and center of mass
    calculate_mass(root);
    calculate_center_of_mass_x(root);
    calculate_center_of_mass_y(root);

    if (rank == 0)
    {
        point = 0;

        for (int i = 1; i < procs; i++)
        {
            point += delta;
            MPI_Send(&point, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
        }

        point = 0;
        calculate_states(point, point + delta);

        for (int i = 1; i < procs; i++)
        {
            point += delta;
            MPI_Recv(&states[point], delta, MPI_STATE_T, i, tag, MPI_COMM_WORLD, &status);
        }
    }
    else
    {
        MPI_Recv(&point, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

        calculate_states(point, point + delta);

        MPI_Send(&states[point], delta, MPI_STATE_T, 0, tag, MPI_COMM_WORLD);
    }

    // Free memory
    free_node(root);
}

/*
 * Calculate forces and update velocities and positions.
 */
void calculate_states(int begin, int end)
{
    // Update forces
    for (int i = begin; i < end; i++)
    {
        states[i].force_x = 0;
        states[i].force_y = 0;
        update_forces(i, root);
    }

    // Update velocities and positions
    for (int i = begin; i < end; i++)
    {
        double ax = states[i].force_x / states[i].mass;
        double ay = states[i].force_y / states[i].mass;
        states[i].u += ax * dt;
        states[i].v += ay * dt;
        states[i].x += states[i].u * dt;
        states[i].y += states[i].v * dt;

        /*
         * This of course doesn't make any sense physically,
         * but makes sure that the particles stay within the
         * bounds. Normally the particles won't leave the
         * area anyway.
         */

        bounce(&states[i].x, &states[i].y, &states[i].u, &states[i].v);
    }
}

/*
 * Help function for calculating the forces recursively
 * using the Barnes Hut quad tree.
 */
void update_forces(int particle, struct node_t *node)
{
    // The node is a leaf node with a particle and not the particle itself
    if (!node->has_children && node->has_particle && node->particle != particle)
    {
        double r = sqrt((states[particle].x - node->c_x) * (states[particle].x - node->c_x) + (states[particle].y - node->c_y) * (states[particle].y - node->c_y));
        calculate_force(particle, node, r);
    }
    // The node has children
    else if (node->has_children)
    {
        // Calculate r and theta
        double r = sqrt((states[particle].x - node->c_x) * (states[particle].x - node->c_x) + (states[particle].y - node->c_y) * (states[particle].y - node->c_y));
        double theta = (node->max_x - node->min_x) / r;

        /* 
         * If the distance to the node's centre of mass is far enough, calculate the force,
         * otherwise traverse further down the tree.
         */

        if (theta < 0.5)
        {
            calculate_force(particle, node, r);
        }
        else
        {
            update_forces(particle, &node->children[0]);
            update_forces(particle, &node->children[1]);
            update_forces(particle, &node->children[2]);
            update_forces(particle, &node->children[3]);
        }
    }
}

/*
 * Calculates and updates the force of a particle from a node.
 */
void calculate_force(int particle, struct node_t *node, double r)
{
    double temp = -grav * states[particle].mass * node->total_mass / ((r + epsilon) * (r + epsilon) * (r + epsilon));
    states[particle].force_x += (states[particle].x - node->c_x) * temp;
    states[particle].force_y += (states[particle].y - node->c_y) * temp;
}

/*
 * If a particle moves beyond any of the boundaries then bounce it back.
 */
void bounce(double *x, double *y, double *u, double *v)
{
    double W = 1.0f, H = 1.0f;
    if (*x > W)
    {
        *x = 2 * W - *x;
        *u = -*u;
    }

    if (*x < 0)
    {
        *x = -*x;
        *u = -*u;
    }

    if (*y > H)
    {
        *y = 2 * H - *y;
        *v = -*v;
    }

    if (*y < 0)
    {
        *y = -*y;
        *v = -*v;
    }
}

/*
 * Puts a particle recursively in the Barnes Hut quad-tree.
 */
void put_particle_in_tree(int new_particle, struct node_t *node)
{
    // If no particle is assigned to the node
    if (!node->has_particle)
    {
        node->particle = new_particle;
        node->has_particle = 1;
    }
    // If the node has no children
    else if (!node->has_children)
    {
        // Allocate and initiate children
        node->children = malloc(4 * sizeof(struct node_t));

        if (node->children == NULL)
        {
            printf("Error: Some malloc won't work on rank = %d.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < 4; i++)
        {
            set_node(&node->children[i]);
        }

        // Set boundaries for the children
        node->children[0].min_x = node->min_x;
        node->children[0].max_x = (node->min_x + node->max_x) / 2;
        node->children[0].min_y = node->min_y;
        node->children[0].max_y = (node->min_y + node->max_y) / 2;

        node->children[1].min_x = (node->min_x + node->max_x) / 2;
        node->children[1].max_x = node->max_x;
        node->children[1].min_y = node->min_y;
        node->children[1].max_y = (node->min_y + node->max_y) / 2;

        node->children[2].min_x = node->min_x;
        node->children[2].max_x = (node->min_x + node->max_x) / 2;
        node->children[2].min_y = (node->min_y + node->max_y) / 2;
        node->children[2].max_y = node->max_y;

        node->children[3].min_x = (node->min_x + node->max_x) / 2;
        node->children[3].max_x = node->max_x;
        node->children[3].min_y = (node->min_y + node->max_y) / 2;
        node->children[3].max_y = node->max_y;

        // Put old particle into the appropriate child
        place_particle(node->particle, node);

        // Put new particle into the appropriate child
        place_particle(new_particle, node);

        // It now has children
        node->has_children = 1;
    }
    // Add the new particle to the appropriate children
    else
    {
        // Put new particle into the appropriate child
        place_particle(new_particle, node);
    }
}

/*
 * Puts a particle in the right child of a node with children
 */
void place_particle(int particle, struct node_t *node)
{
    if (states[particle].x <= (node->min_x + node->max_x) / 2 && states[particle].y <= (node->min_y + node->max_y) / 2)
    {
        put_particle_in_tree(particle, &node->children[0]);
    }
    else if (states[particle].x > (node->min_x + node->max_x) / 2 && states[particle].y < (node->min_y + node->max_y) / 2)
    {
        put_particle_in_tree(particle, &node->children[1]);
    }
    else if (states[particle].x < (node->min_x + node->max_x) / 2 && states[particle].y > (node->min_y + node->max_y) / 2)
    {
        put_particle_in_tree(particle, &node->children[2]);
    }
    else
    {
        put_particle_in_tree(particle, &node->children[3]);
    }
}

/*
 * Sets initial values for a new node.
 */
void set_node(struct node_t *node)
{
    node->has_particle = 0;
    node->has_children = 0;
}

/*
 * Frees memory for a node and its children recursively.
 */
void free_node(struct node_t *node)
{
    if (node->has_children)
    {
        free_node(&node->children[0]);
        free_node(&node->children[1]);
        free_node(&node->children[2]);
        free_node(&node->children[3]);
        free(node->children);
    }
}

/*
 * Calculates the total mass for the node. It recursively updates the mass
 * of itself and all of its children.
 */
double calculate_mass(struct node_t *node)
{
    if (!node->has_particle)
    {
        node->total_mass = 0;
    }
    else if (!node->has_children)
    {
        node->total_mass = states[node->particle].mass;
    }
    else
    {
        node->total_mass = 0;

        for (int i = 0; i < 4; i++)
        {
            node->total_mass += calculate_mass(&node->children[i]);
        }
    }

    return node->total_mass;
}

/*
 * Calculates the x-position of the centre of mass for the 
 * node. It recursively updates the position of itself and 
 * all of its children.
 */
double calculate_center_of_mass_x(struct node_t *node)
{
    if (!node->has_children)
    {
        node->c_x = states[node->particle].x;
    }
    else
    {
        node->c_x = 0;
        double m_tot = 0;

        for (int i = 0; i < 4; i++)
        {
            if (node->children[i].has_particle)
            {
                node->c_x += node->children[i].total_mass * calculate_center_of_mass_x(&node->children[i]);
                m_tot += node->children[i].total_mass;
            }
        }

        node->c_x /= m_tot;
    }

    return node->c_x;
}

/*
 * Calculates the y-position of the centre of mass for the 
 * node. It recursively updates the position of itself and 
 * all of its children.
 */
double calculate_center_of_mass_y(struct node_t *node)
{
    if (!node->has_children)
    {
        node->c_y = states[node->particle].y;
    }
    else
    {
        node->c_y = 0;
        double m_tot = 0;

        for (int i = 0; i < 4; i++)
        {
            if (node->children[i].has_particle)
            {
                node->c_y += node->children[i].total_mass * calculate_center_of_mass_y(&node->children[i]);
                m_tot += node->children[i].total_mass;
            }
        }

        node->c_y /= m_tot;
    }

    return node->c_y;
}

/*
 * Function to read a case.
 */
int read_case(char *filename)
{
    int i, s;
    FILE *arq = fopen(filename, "r");

    if (arq == NULL)
    {
        printf("Error: The file %s could not be opened.\n", filename);
        return 1;
    }

    s = fscanf(arq, "%d", &total_states);

    if (s != 1)
    {
        printf("Error: The file %s could not be read for number of particles.\n", filename);
        fclose(arq);
        return 1;
    }

    // Initiate memory for the vector
    states = malloc(sizeof(struct state_t) * total_states);

    if (states == NULL)
    {
        printf("Error: Some malloc won't work on rank = %d.\n", rank);
        fclose(arq);
        return 1;
    }

    for (i = 0; i < total_states; i++)
    {
        s = fscanf(arq, "%lf %lf %lf %lf %lf", &states[i].mass, &states[i].x, &states[i].y, &states[i].u, &states[i].v);

        if (s != 5)
        {
            printf("Error: Some reading won't work at line %d (%d).\n", i + 1, s);
            fclose(arq);
            return 1;
        }
    }

    fscanf(arq, "%d", &time_steps);

    if (filename)
    {
        fclose(arq);
    }

    return 0;
}

/* 
 * Prints statistics: time, N, final velocity, final center of mass.
 */
void print_statistics(float vu, float vv, float cx, float cy)
{
    printf("%.5f %.5f\n", vu, vv);
    printf("%.5f %.5f\n", cx, cy);
}
