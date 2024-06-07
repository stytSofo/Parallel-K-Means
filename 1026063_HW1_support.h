#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <errno.h>
#include <pthread.h>

#define GRAPH_COLUMNS 50
#define MAX_NUM_OF_STAR 20
#define SET_SIZE 1000
#define SEED 123456

#define	RADIUS				10.0

 
typedef struct {
	double				x;
	double				y;
	//int group;
	unsigned long		group; // To aligne with Double
} POINT;


// Support for thread safe timers
static pthread_mutex_t timer_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
	struct timeval startTime;
	struct timeval endTime;
} Timer;

Timer timer[1000];

void startTime(int i);
void stopTime(int i);
void elapsedTime(int i);

void startTime(int i) {
	//printf("Start Timer...");
	pthread_mutex_lock(&timer_mutex);
	gettimeofday(&(timer[i].startTime), NULL);
	pthread_mutex_unlock(&timer_mutex);
}

void stopTime(int i) {
	//printf("Stop Timer...");
	pthread_mutex_lock(&timer_mutex);
	gettimeofday(&(timer[i].endTime), NULL);
	pthread_mutex_unlock(&timer_mutex);
}

void elapsedTime(int i) {
	float elapseTime = (float) ((timer[i].endTime.tv_sec
			- timer[i].startTime.tv_sec)
			+ (timer[i].endTime.tv_usec - timer[i].startTime.tv_usec) / 1.0e6);
	printf("%%Time: %4.2f Sec.\n", elapseTime);
}

/* Fill in array with values from 0 to setSize */
void initData(int size,  int setSize, int * theArray)
{
  int i;
  srand(SEED);
  printf("Initializing the array with Uniformly Distributed Data...\n");
  for (i = 0; i < size; i++)
	theArray[i] = rand()%setSize;
}
/* Fill in array with values from 0 to setSize */
void initDataSKEWED(int size, int setSize, int * theArray, int countNumber)
{
  int i;
  srand(SEED);
  printf("Initializing the array with SKEWED !!!! Distribution Data...\n");
  for (i = 0; i < size; i++)
	  if (i<size/2)
		  theArray[i]=countNumber;
	  else
		  theArray[i] = rand()%SET_SIZE;
	
}
void printArray(int * theArray, int size){
	int i;
	for (i = 0; i < size; i++)printf("%d,",theArray[i]);
	printf("\n");
}

/*-------------------------------------------------------
	Kmeans_initData
	This function allocates a block of memory for data points,
	gives the data points random values and returns a pointer to them.
	The data points fall within a circle of the radius passed to
	the function. This does not create a uniform 2-dimensional
	distribution.
-------------------------------------------------------*/
POINT * Kmeans_initData(int num_pts, double radius)
{
	int		i;
	double	ang, r;
	POINT * pts;
 
	pts = (POINT*) malloc(sizeof(POINT) * num_pts);
    srand(SEED);
	for ( i = 0; i < num_pts; i++ ) {
		ang = 2.0 * M_PI * rand() / (RAND_MAX - 1.);
		r = radius * rand() / (RAND_MAX - 1.);
		pts[i].x = r * cos(ang);
		pts[i].y = r * sin(ang);
	}
	return pts;	
}
double ** Kmeans_initData_v3(int num_pts, double radius)
{
	int	i, nrows = 2;
	double	ang, r;
	double ** pts;
	pts = malloc(nrows * sizeof(double *));
	for ( i = 0; i < nrows; i++ )
		pts[i] = malloc(num_pts * sizeof(double));

    srand(SEED);
	for ( i = 0; i < num_pts; i++ ) {
		ang = 2.0 * M_PI * rand() / (RAND_MAX - 1.);
		r = radius * rand() / (RAND_MAX - 1.);
		pts[0][i] = r * cos(ang);
		pts[1][i] = r * sin(ang);
	}
	return pts;	
}
double ** Kmeans_initData_v3_1(int num_pts)
{
	int	i, nrows = 2;
	double ** pts;
	pts = malloc(nrows * sizeof(double *));
	for ( i = 0; i < nrows; i++ )
		pts[i] = malloc(num_pts * sizeof(double));

    srand(SEED);
	for ( i = 0; i < num_pts; i++ ) {
		//ang = 2.0 * M_PI * rand() / (RAND_MAX - 1.);
		//r = radius * rand() / (RAND_MAX - 1.);
		pts[0][i] = ((double)rand()/(double)(RAND_MAX))*10;
		pts[1][i] = ((double)rand()/(double)(RAND_MAX))*10;
	}
	return pts;	
}
/*-------------------------------------------------------
	print_eps
	this function prints the results.
-------------------------------------------------------*/
void print_eps(POINT * pts, int num_pts, POINT * centroids, int num_clusters)
{
#	define W 400
#	define H 400
 
	int i, j;
	double min_x, max_x, min_y, max_y, scale, cx, cy;
	double *colors = (double *) malloc(sizeof(double) * num_clusters * 3);
 
	for (i = 0; i < num_clusters; i++) {
		colors[3*i + 0] = (3 * (i + 1) % 11)/11.;
		colors[3*i + 1] = (7 * i % 11)/11.;
		colors[3*i + 2] = (9 * i % 11)/11.;
	}
 
	max_x = max_y = - HUGE_VAL;
	min_x = min_y = HUGE_VAL;
	for (j = 0; j < num_pts; j++) {
		if (max_x < pts[j].x) max_x = pts[j].x;
		if (min_x > pts[j].x) min_x = pts[j].x;
		if (max_y < pts[j].y) max_y = pts[j].y;
		if (min_y > pts[j].y) min_y = pts[j].y;
	}
 
	scale = W / (max_x - min_x);
	if (scale > H / (max_y - min_y))
		scale = H / (max_y - min_y);
	cx = (max_x + min_x) / 2;
	cy = (max_y + min_y) / 2;
 
	printf("%%!PS-Adobe-3.0\n%%%%BoundingBox: -5 -5 %d %d\n", W + 10, H + 10);
	printf( "/l {rlineto} def /m {rmoveto} def\n"
		"/c { .25 sub exch .25 sub exch .5 0 360 arc fill } def\n"
		"/s { moveto -2 0 m 2 2 l 2 -2 l -2 -2 l closepath "
		"	gsave 1 setgray fill grestore gsave 3 setlinewidth"
		" 1 setgray stroke grestore 0 setgray stroke }def\n"
	);
 
 
	for (i = 0; i < num_clusters; i++) {
		printf("%g %g %g setrgbcolor\n",
			colors[3*i], colors[3*i + 1], colors[3*i + 2]);
 
		for (j = 0; j < num_pts; j++) {
			if (pts[j].group != i) continue;
			printf("%.3f %.3f c\n",
				(pts[j].x - cx) * scale + W / 2,
				(pts[j].y - cy) * scale + H / 2);
		}
		printf("\n0 setgray %g %g s\n",
			(centroids[i].x - cx) * scale + W / 2,
			(centroids[i].y - cy) * scale + H / 2);
	}
	printf("\n%%%%EOF");
 
	free(colors);
 
	return;
}	/* end print_eps */
void print_eps_v3(double ** pts, unsigned int * pts_group, int num_pts, double ** centroids, unsigned int * centroids_group, int num_clusters, char *filename)
{
	FILE *fptr;
	fptr = fopen(filename,"w");

#	define W 400
#	define H 400
 
	int i, j;
	double min_x, max_x, min_y, max_y, scale, cx, cy;
	double *colors = (double *) malloc(sizeof(double) * num_clusters * 3);
 
	for (i = 0; i < num_clusters; i++) {
		colors[3*i + 0] = (3 * (i + 1) % 11)/11.;
		colors[3*i + 1] = (7 * i % 11)/11.;
		colors[3*i + 2] = (9 * i % 11)/11.;
	}
 
	max_x = max_y = - HUGE_VAL;
	min_x = min_y = HUGE_VAL;
	for (j = 0; j < num_pts; j++) {
		if (max_x < pts[0][j]) max_x = pts[0][j];
		if (min_x > pts[0][j]) min_x = pts[0][j];
		if (max_y < pts[1][j]) max_y = pts[1][j];
		if (min_y > pts[1][j]) min_y = pts[1][j];
	}
 
	scale = W / (max_x - min_x);
	if (scale > H / (max_y - min_y))
		scale = H / (max_y - min_y);
	cx = (max_x + min_x) / 2;
	cy = (max_y + min_y) / 2;
 
	fprintf(fptr,"%%!PS-Adobe-3.0\n%%%%BoundingBox: -5 -5 %d %d\n", W + 10, H + 10);
	fprintf( fptr,"/l {rlineto} def /m {rmoveto} def\n"
		"/c { .25 sub exch .25 sub exch .5 0 360 arc fill } def\n"
		"/s { moveto -2 0 m 2 2 l 2 -2 l -2 -2 l closepath "
		"	gsave 1 setgray fill grestore gsave 3 setlinewidth"
		" 1 setgray stroke grestore 0 setgray stroke }def\n"
	);
 
 
	for (i = 0; i < num_clusters; i++) {
		fprintf(fptr,"%g %g %g setrgbcolor\n",
			colors[3*i], colors[3*i + 1], colors[3*i + 2]);
 
		for (j = 0; j < num_pts; j++) {
			if (pts_group[j] != i) continue;
			fprintf(fptr,"%.3f %.3f c\n",
				(pts[0][j] - cx) * scale + W / 2,
				(pts[1][j] - cy) * scale + H / 2);
		}
		fprintf(fptr,"\n0 setgray %g %g s\n",
			(centroids[0][i] - cx) * scale + W / 2,
			(centroids[1][i] - cy) * scale + H / 2);
	}
	fprintf(fptr,"\n%%%%EOF");
 
	free(colors);
 
	return;
}	/* end print_eps */  
/*-------------------------------------------------------
	dist2
 
	This function returns the squared euclidean distance
	between two data points.
-------------------------------------------------------*/
double dist2(POINT * a, POINT * b)
{
	double x = a->x - b->x;
	double y = a->y - b->y;
	return x*x + y*y;
}
/*------------------------------------------------------
	nearest
 
  This function returns the index of the cluster centroid
  nearest to the data point passed to this function.
------------------------------------------------------*/
int nearest(POINT * pt, POINT * cent, int n_cluster)
{
	int i, clusterIndex;
	double d, min_d;
 
	min_d = HUGE_VAL;
	clusterIndex = pt->group;	
	for (i = 0; i < n_cluster; i++) {
		d = dist2(&cent[i], pt);
		if ( d < min_d ) {
			min_d = d;
			clusterIndex = i;
		}
	}	
	return clusterIndex;
}
  

/*------------------------------------------------------
	nearestDistance
  This function returns the distance of the cluster centroid
  nearest to the data point passed to this function.
------------------------------------------------------*/
double nearestDistance(POINT * pt, POINT * cent, int n_cluster)
{
	int i;
	double d, min_d;
 
	min_d = HUGE_VAL;
	for (i = 0; i < n_cluster; i++) {
		d = dist2(&cent[i], pt);
		if ( d < min_d ) {
			min_d = d;
		}
	}
 
	return min_d;
}
 /*----------------------------------------------------------------------
  bisectionSearch
  This function makes a bisectional search of an array of values that are
  ordered in increasing order, and returns the index of the first element
  greater than the search value passed as a parameter.
 
  This code is adapted from code by Andy Allinger given to the public
  domain.
 
  Input:
		x	A pointer to an array of values in increasing order to be searched.
		n	The number of elements in the input array x.
		v	The search value.
  Output:
		Returns the index of the first element greater than the search value, v.
----------------------------------------------------------------------*/
int bisectionSearch(double *x, int n, double v)
{
	int il, ir, i;
 
 
	if (n < 1) {
		return 0;
	}
	/* If v is less than x(0) or greater than x(n-1)  */
	if (v < x[0]) {
		return 0;
	}
	else if (v > x[n-1]) {
		return n - 1;
	}
 
	/*bisection search */
	il = 0;
	ir = n - 1;
 
	i = (il + ir) / 2;
	while ( i != il ) {
		if (x[i] <= v) {
			il = i;
		} else {
			ir = i;
		}
		i = (il + ir) / 2;		
	}		
 
	if (x[i] <= v)
		i = ir;
	return i;
} /* end of bisectionSearch */

/*-------------------------------------------------------
	kppAllinger
	This function uses the K-Means++ method to select
	the cluster centroids.
 
	This code is adapted from code by Andy Allinger given to the
	public domain.
 
	Input:
		pts		A pointer to an array of data points.
		num_pts		The number of points in the pts array.
		centroids	A pointer to an array to receive the centroids.
		num_clusters	The number of clusters to be found.
 
	Output:
		centroids	A pointer to the array of centroids found.	
-------------------------------------------------------*/
void kppAllinger(POINT * pts, int num_pts, POINT * centroids,
		 int num_clusters)
{
	int j;
	int selectedIndex;
	int cluster;
	double sum;
	double d;
	double random;	
	double * cumulativeDistances;
	double * shortestDistance;
 
 
	cumulativeDistances = (double*) malloc(sizeof(double) * num_pts);
	shortestDistance = (double*) malloc(sizeof(double) * num_pts);	
 
 
	/* Pick the first cluster centroids at random. */
	selectedIndex = rand() % num_pts;
	centroids[0] = pts[ selectedIndex ];
 
	for (j = 0; j < num_pts; ++j)
		shortestDistance[j] = HUGE_VAL;	
 
	/* Select the centroids for the remaining clusters. */
	for (cluster = 1; cluster < num_clusters; cluster++) {
 
		/* For each point find its closest distance to any of
		   the previous cluster centers */
		for ( j = 0; j < num_pts; j++ ) {
			d = dist2(&pts[j], &centroids[cluster-1] );
 
			if (d < shortestDistance[j])
				shortestDistance[j] = d;
		}
 
		/* Create an array of the cumulative distances. */
		sum = 0.0;
		for (j = 0; j < num_pts; j++) {
			sum += shortestDistance[j];
			cumulativeDistances[j] = sum;
		}
 
		/* Select a point at random. Those with greater distances
		   have a greater probability of being selected. */
		random = (float) rand() / (float) RAND_MAX * sum;
		selectedIndex = bisectionSearch(cumulativeDistances, num_pts, random);
 
		/* assign the selected point as the center */
		centroids[cluster] = pts[selectedIndex];
	}
 
	/* Assign each point the index of it's nearest cluster centroid. */
	for (j = 0; j < num_pts; j++)
		pts[j].group = nearest(&pts[j], centroids, num_clusters);
 
	free(shortestDistance);
	free(cumulativeDistances);
 
	return;
}	/* end, kppAllinger */
 
/*-------------------------------------------------------
	kpp
	This function uses the K-Means++ method to select
	the cluster centroids.
-------------------------------------------------------*/
void kpp(POINT * pts, int num_pts, POINT * centroids,
		 int num_clusters)
{
	int j;
	int cluster;
	double sum;
	double * distances;
 
 
	distances = (double*) malloc(sizeof(double) * num_pts);
 
	/* Pick the first cluster centroids at random. */
	centroids[0] = pts[ rand() % num_pts ];
 
 
	/* Select the centroids for the remaining clusters. */
	for (cluster = 1; cluster < num_clusters; cluster++) {
 
		/* For each data point find the nearest centroid, save its
		   distance in the distance array, then add it to the sum of
		   total distance. */
		sum = 0.0;
		for ( j = 0; j < num_pts; j++ ) {
			distances[j] = 
				nearestDistance(&pts[j], centroids, cluster);
			sum += distances[j];
		}
 
		/* Find a random distance within the span of the total distance. */
		sum = sum * rand() / (RAND_MAX - 1);
 
		/* Assign the centroids. the point with the largest distance
			will have a greater probability of being selected. */
		for (j = 0; j < num_pts; j++ ) {
			sum -= distances[j];
			if ( sum <= 0)
			{
				centroids[cluster] = pts[j];
				break;
			}
		}
	}
 
	/* Assign each observation the index of it's nearest cluster centroid. */
	for (j = 0; j < num_pts; j++)
		pts[j].group = nearest(&pts[j], centroids, num_clusters);
 
	free(distances);
 
	return;
}	/* end, kpp */
 
 