/**
*This program runs the classic cellular automaton model of Conway's Game of Life.
* This supports multithreading for faster simulation speeds.
*
* @author Kaidan Fullerton
* @version 1.0
*/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <pthread.h>

/****************** Definitions **********************/
/* Three possible modes in which the GOL simulation can run */
#define OUTPUT_NONE   0   // with no animation
#define OUTPUT_ASCII  1   // with ascii animation

/* Used to slow down animation run modes: usleep(SLEEP_USECS);
 * This value can make the animation run faster or slower 
 */
#define SLEEP_USECS    100000

/* A global variable to keep track of the number of live cells in the world */
static int total_live = 0;         
/* Mutex and barrier used for thread synchronization */
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_barrier_t barrier;

/* This struct represents all the data needed to keep track of the GOL
 * simulation.  Rather than passing individual arguments into each function,
 * we pass in everything in just one of these structs.
 * this is passed to play_gol, the main gol playing loop  
 */
struct gol_data {

   // struct fields
   int rows;  // the row dimension
   int cols;  // the column dimension
   int iters; // number of iterations to run the gol simulation
   int output_mode; // set to:  OUTPUT_NONE or OUTPUT_ASCII  

   // 2-d array of the board
   // *copy points to a copy of the board used for calculations
   // *c is used for swapping 
   int *grid, *copy, *c;

   int num_threads, partition, print_partition;
   // threading information
   int thread_num, r_min, r_max, c_min, c_max;
};

/****************** Function Prototypes **********************/

/* allocates memory for the dynamically allocated array */
void initialize_grid(struct gol_data *data);
/* returns the index in the flattened array for coordinates (i,j) */
int grid_index(struct gol_data *data, int i, int j);
/* runs a step in the GOL board (for a given thread) */
void step(struct gol_data *data);
/* creates a clone of the gol_data object. Helper function for step. */
struct gol_data clone(struct gol_data *data);
/* adjusts coordinates based on how the grid is structured like a torus */
void torus_adjust(struct gol_data *data, int *x, int *y);
/* the main Game of Life playing loop */
void *play_gol(void *args);
/* init gol data from the input file and run mode cmdline args */
int init_game_data_from_args(struct gol_data *data, char *argv[]);

/* print board to the terminal (for OUTPUT_ASCII mode) */ 
void print_board(struct gol_data *data, int round);

int main(int argc, char *argv[]) {
  int ret;
  struct gol_data *data;
  struct timeval start_time, end_time;
  double secs;
  pthread_t *tids;
  struct gol_data *tid_args;
  srand(time(0));

  /* check number of command line arguments */
  if (argc < 5) { 
    printf("usage: ./gol <infile> <0|1|> <num_threads> <0|1> <0|1>\n");
    printf("(0: with no visi, 1: with ascii visi\n");
    exit(1);
  }

  /* Initialize game state (all fields in data) from information 
   * read from input file */
  data = malloc(sizeof(struct gol_data));
  ret = init_game_data_from_args(data, argv);
  if(ret != 0) {
    printf("Error init'ing with file %s, mode %s\n", argv[1], argv[2]);
    exit(1);
  }

  // create the thread id array and array for thread arguments
  tids = malloc(sizeof(pthread_t)*data->num_threads);
  if (!tids) {
      perror("malloc pthread_t array");
      exit(1);
  }
  tid_args = malloc(sizeof(struct gol_data)*data->num_threads);
  if (!tid_args) {
      perror("malloc pthread args array");
      exit(1);
  }

  if (pthread_barrier_init(&barrier,0,data->num_threads)) {
    printf("pthread_barrier_init error\n");
    exit(1);
  }

  // partition the board evenly among the threads
  if (data->partition == 0) { //row-wise
    if (data->num_threads > data->rows) {
        printf("Error: more threads than rows\n");
        exit(1);
    }
    int row = 0;  
    for (int i = 0; i < data->num_threads; i++) {
      int sz = data->rows/data->num_threads;
      if (i < (data->rows % data->num_threads)) sz++;
      tid_args[i] = *data;
      tid_args[i].thread_num = i;
      tid_args[i].r_min = row;
      tid_args[i].r_max = row+sz-1;
      tid_args[i].c_min = 0;
      tid_args[i].c_max = data->cols-1;

      row += sz;
    }
  }
  else if (data->partition == 1) { // column-wise
    if (data->num_threads > data->cols) {
        printf("Error: more threads than cols\n");
        exit(1);
    }
    int col = 0;  
    for (int i = 0; i < data->num_threads; i++) {
      int sz = data->cols/data->num_threads;
      if (i < (data->cols % data->num_threads)) sz++;
      tid_args[i] = *data;
      tid_args[i].thread_num = i;
      tid_args[i].r_min = 0;
      tid_args[i].r_max = data->rows-1;
      tid_args[i].c_min = col;
      tid_args[i].c_max = col+sz-1;

      col += sz;
    }
  }

  ret = gettimeofday(&start_time,NULL);
  if (ret != 0) {
    printf("Error getting time of day\n");
    exit(1);
  }

  /* Invoke play_gol in different ways based on the run mode */
  if(data->output_mode == OUTPUT_NONE) {  // run with no animation
    for (int i = 0; i < data->num_threads; i++) {
        //printf("making thread %d\n",i);
        ret = pthread_create(&tids[i],0,play_gol,&tid_args[i]);
        if (ret) {
            perror("Error pthread_create\n");
            exit(1);
        }
    }
    for (int i = 0; i < data->num_threads; i++) {
        pthread_join(tids[i],0);
    }
  } 
  else if (data->output_mode == OUTPUT_ASCII) { // run with ascii animation
    for (int i = 0; i < data->num_threads; i++) {
        //printf("making thread %d\n",i);
        ret = pthread_create(&tids[i],0,play_gol,&tid_args[i]);
        if (ret) {
            perror("Error pthread_create\n");
            exit(1);
        }
    }
    for (int i = 0; i < data->num_threads; i++) {
        pthread_join(tids[i],0);
    }
    // clear the previous print_board output from the terminal:
    if(system("clear")) { perror("clear"); exit(1); }

    /* If the # of iterations is even, then data will be pointing to the wrong board, so we need to swap them for the final print */
    if (data->iters % 2 == 0) {
        data->c = data->grid;
        data->grid = data->copy;
        data->copy = data->c;
    }
    print_board(data, data->iters); 
  } 

  if (data->output_mode == OUTPUT_ASCII || data->output_mode == OUTPUT_NONE) {
    ret = gettimeofday(&end_time,NULL);
    if (ret != 0) {
      printf("Error getting time of day\n");
      exit(1);
    }
    secs = (double)((end_time.tv_sec*1000000+end_time.tv_usec)-(start_time.tv_sec*1000000+start_time.tv_usec))/1000000.0;

    /* Print the total runtime, in seconds. */
    fprintf(stdout, "Total time: %0.3f seconds\n", secs);
    /*
    fprintf(stdout, "Number of live cells after %d rounds: %d\n\n", 
        data.iters, total_live);
        */
    printf("After %d rounds on %dx%d, number of live cells is %d\n",data->iters,data->rows,data->cols,total_live);
  }

  free(data->grid);
  free(data->copy);
  free(tids);
  free(tid_args);
  free(data);

  return 0;
}

/* initialize the gol game state from command line arguments
 *       argv[1]: name of file to read game config state from
 *       argv[2]: run mode value
 * data: pointer to gol_data struct to initialize
 * argv: command line args 
 *       argv[1]: name of file to read game config state from
 *       argv[2]: run mode
 * returns: 0 on success, 1 on error
 */

// creates the grid by dynamically allocating a 2-d array
void initialize_grid(struct gol_data *data) {
    data->grid = malloc(data->rows*data->cols*sizeof(int));
    data->copy = malloc(data->rows*data->cols*sizeof(int));
    int i, j, idx;
    // initialize everything in the grid to be 0
    for (i = 0; i < data->rows; i++) {
        for (j = 0; j < data->cols; j++) {
            idx = grid_index(data,i,j);
            data->grid[idx] = 0;
            data->copy[idx] = 0;
        }
    }
}

// returns the index of the corresponding point in the flattened array
int grid_index(struct gol_data *data, int i, int j) {
    return i*data->cols + j;
}

// this calculates each round.
// we create a copy of the data struct, and we use this
// to modify the original grid in place.
// to calculate whether a square should be dead or alive, we
// look at the 3x3 square centered around it, accounting for the torus shape,
// and count how many are alive.
// now since it's multithreaded, each thread is only responsible for updating its own section.
void step(struct gol_data *data) {
    //pthread_mutex_lock(&mutex);
    int i, j, idx;
    int dx, dy;
    int nx, ny;
    int count;
    for (i = data->r_min; i <= data->r_max; i++) {
        for (j = data->c_min; j <= data->c_max; j++) {
            count = 0;
            for (dx = -1; dx <= 1; dx++) {
                for (dy = -1; dy <= 1; dy++) {
                    // this looks at the 3x3 square centered around (i,j)
                    if (dx == 0 && dy == 0) continue;
                    nx = i + dx;
                    ny = j + dy;

                    torus_adjust(data,&nx,&ny);
                    idx = grid_index(data,nx,ny);
                    if (data->grid[idx] == 1) {
                        count++;    
                    }
                }
            }
            idx = grid_index(data,i,j);
            if (data->grid[idx] == 1) {
                // logic for a currently alive cell
                if (count <= 1 || count >= 4) {
                    data->copy[idx] = 0;
                }
                else {
                    // mutex lock when updating total_live
                    pthread_mutex_lock(&mutex);
                    total_live++;
                    pthread_mutex_unlock(&mutex);
                    data->copy[idx] = 1;
                }
            }
            else {
                // logic for a currently dead cell
                if (count == 3) {
                    pthread_mutex_lock(&mutex);
                    total_live++;
                    pthread_mutex_unlock(&mutex);
                    data->copy[idx] = 1;
                }
                else {
                    data->copy[idx] = 0;
                }
            }
        }
    }
    //pthread_mutex_unlock(&mutex);
}

// this only creates a copy that has the same rows, cols, grid
struct gol_data clone(struct gol_data *data) {
    struct gol_data copy;
    copy.rows = data->rows;
    copy.cols = data->cols;
    int i, j, idx;
    initialize_grid(&copy);
    for (i = 0; i < copy.rows; i++) {
        for (j = 0; j < copy.cols; j++) {
            idx = grid_index(&copy,i,j);
            copy.grid[idx] = data->grid[idx];
        }
    }
    return copy;
}

// adjusts the indices so they don't go off the grid
void torus_adjust(struct gol_data *data, int *x, int *y) {
    if (*x < 0) *x += data->rows;
    if (*x >= data->rows) *x -= data->rows;
    if (*y < 0) *y += data->cols;
    if (*y >= data->cols) *y -= data->cols;
}

int init_game_data_from_args(struct gol_data *data, char *argv[]) {

  FILE *infile;
  int output_mode;
  int ret;
  int n_points;
  int i;
  int x, y;
  int idx;
  infile = fopen(argv[1],"r");
  
  output_mode = atoi(argv[2]);
  data->num_threads = atoi(argv[3]);
  data->partition = atoi(argv[4]);
  data->print_partition = atoi(argv[5]);

  if (data->num_threads < 1) {
      printf("Error: invalid # of threads\n");
      exit(1);
  }
  if (data->partition != 0 && data->partition != 1) {
    printf("Error: invalid partition\n");
    exit(1);
  }
  if (data->print_partition != 0 && data->print_partition != 1) {
      printf("Error: invalid print_partition\n");
      exit(1);
  }
  if (infile == NULL) {
      printf("Error: file open %s\n",argv[1]);
      exit(1);
  }
  ret = fscanf(infile,"%d%d%d%d",&data->rows,&data->cols,&data->iters,&n_points);
  if (ret != 4) {
      printf("Error: reading input file\n");
      exit(1);
  }
  
  data->output_mode = output_mode;
  //printf("rows = %d, cols = %d, iters = %d, n_points = %d, output_mode = %d\n",data->rows,data->cols,data->iters,n_points,data->output_mode);
  initialize_grid(data);
  for (i = 0; i < n_points; i++) {
    ret = fscanf(infile,"%d%d",&x,&y);
    if (ret != 2) {
        printf("Error: reading input points\n");
        exit(1);
    }
    idx = grid_index(data,x,y);
    data->grid[idx] = 1;
    data->copy[idx] = 1;
  }
  fclose(infile);

  return 0;
}
/**********************************************************/
/* the gol application main loop function:
 *  runs rounds of GOL, 
 *    * updates program state for next round (world and total_live)
 *    * performs any animation step based on the output/run mode
 *
 *   data: pointer to a struct gol_data initialized with
 *         all GOL game playing state
 */
void *play_gol(void *args) {
  struct gol_data *data;
  data = (struct gol_data *)args;
  int ret;
  if (data->print_partition == 1) {
    printf("tid %d: rows %d:%d (%d) cols: %d:%d (%d)\n",data->thread_num,data->r_min,data->r_max,data->r_max-data->r_min+1,data->c_min,data->c_max,data->c_max-data->c_min+1);
    //printf("Hi I am thread #%d\nI am assigned rows [%d, %d], and columns [%d, %d]\n",data->thread_num,data->r_min,data->r_max,data->c_min,data->c_max);
    //printf("dimensions are %d by %d\n",data->rows,data->cols);
  }
  int t;
  for (t = 0; t < data->iters; t++) {
    // synch pthreads
    pthread_barrier_wait(&barrier);

    data->c = data->grid;
    data->grid = data->copy;
    data->copy = data->c;
    if (data->thread_num == 0) {
      // thread 0 handles printing the board, sleeping, and resetting total_live
      if (data->output_mode == OUTPUT_ASCII) {
          ret = system("clear");
          if (ret == -1) {
              printf("System call failed\n");
              exit(1);
          }
          print_board(data,t);
          usleep(SLEEP_USECS);
      }
      total_live = 0;
    }
    pthread_barrier_wait(&barrier);
    step(data);
  }
  return NULL;
  //  at the end of each round of GOL, determine if there is an 
  //  animation step to take based on the ouput_mode, 
  //   if ascii animation:
  //     (a) call system("clear") to clear previous world state from terminal
  //     (b) call print_board function to print current world state
  //     (c) call usleep(SLEEP_USECS) to slow down the animation 
  
}
/**********************************************************/
/* Print the board to the terminal. 
 *   data: gol game specific data
 *   round: the current round number
 */
void print_board(struct gol_data *data, int round) {
    int i, j, idx;

    /* Print the round number. */
    fprintf(stderr, "Round: %d\n", round);

    for (i = 0; i < data->rows; ++i) {
        for (j = 0; j < data->cols; ++j) {
            idx = grid_index(data,i,j);
            if (data->grid[idx] == 1)
                fprintf(stderr, " @");
            else 
                fprintf(stderr, " .");
        }
        fprintf(stderr, "\n");
    }

    /* Print the total number of live cells. */
    fprintf(stderr, "Live cells: %d\n\n", total_live);
}