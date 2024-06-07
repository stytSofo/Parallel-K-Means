/* How to Run
Compile Using:
	gcc -Wall -Werror -fopenmp -lm 1026063_HW2_K-meansSerial_std.c
Run Using:
	./a.out [NumberOfPoints NumberOfClusters NumberOfIterations NumberOfThreads]
	./a.out 1000000 100 1000 12
For gprof:
	gcc -Werror -Wall -pg -lm 1026063_HW2_K-meansSerial_std.c
	./a.out
	gprof ./a.out > analysis.txt
	gprof ./a.out | ./gprof2dot.py | dot -Tpng -o output.png
For perf:
	 perf record -g -- ./a.out
	 perf script | c++filt | ./gprof2dot.py -f perf | dot -Tpng -o output.png
eps-to-jpg
  https://cloudconvert.com/eps-to-jpg
Reference:
 https://github.com/jrfonseca/gprof2dot
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "1026063_HW1_support.h"

#define NUMBER_OF_POINTS 100000
#define NUMBER_OF_CLUSTERS 20
#define MAXIMUM_ITERATIONS 100
#define SIMD_WIDTH 256
#define SIMD_STEP 256 / 64
#define NO_OF_THREADS 4

void AssignInitialClusteringRandomly(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes)
{
	int i; /*Assign initial clustering randomly using the Random Partition method*/
	for (i = 0; i < num_pts; i++)
		pts_group[i] = i % num_clusters;
}

void AssignInitialClusteringRandomly_omp_static(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int num_threads)
{
	int i; /*Assign initial clustering randomly using the Random Partition method*/

#pragma omp parallel for schedule(static, num_pts / num_threads) private(i)
	for (i = 0; i < num_pts; i++)
	{
		pts_group[i] = i % num_clusters;
	}
}

void AssignInitialClusteringRandomly_omp_dynamic(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int num_threads)
{
	int i; /*Assign initial clustering randomly using the Random Partition method*/
	#pragma omp parallel for schedule(dynamic, num_pts / num_threads) private(i)
	for (i = 0; i < num_pts; i++)
		pts_group[i] = i % num_clusters;
}

void AssignInitialClusteringRandomly_omp_guided(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int num_threads)
{
	int i; /*Assign initial clustering randomly using the Random Partition method*/
	#pragma omp parallel for schedule(guided, num_pts / num_threads) private(i)
	for (i = 0; i < num_pts; i++)
		pts_group[i] = i % num_clusters;
}

void InitializeClusterXY(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes)
{
	int i;
	for (i = 0; i < num_clusters; i++)
	{
		centroids_group[i] = 0; /* used to count the cluster members. */
		centroids[0][i] = 0;	/* used for x value totals. */
		centroids[1][i] = 0;	/* used for y value totals. */
	}
}

void InitializeClusterXY_omp_static(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int num_threads)
{
	int i;
#pragma omp parallel for schedule(static, num_clusters / num_threads) private(i)
	for (i = 0; i < num_clusters; i++)
	{
		centroids_group[i] = 0; /* used to count the cluster members. */
		centroids[0][i] = 0;	/* used for x value totals. */
		centroids[1][i] = 0;	/* used for y value totals. */
	}
}

void InitializeClusterXY_omp_dynamic(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int num_threads)
{
	int i;
#pragma omp parallel for schedule(dynamic, num_clusters / num_threads) private(i)
	for (i = 0; i < num_clusters; i++)
	{
		centroids_group[i] = 0; /* used to count the cluster members. */
		centroids[0][i] = 0;	/* used for x value totals. */
		centroids[1][i] = 0;	/* used for y value totals. */
	}
}

void InitializeClusterXY_omp_guided(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int num_threads)
{
	int i;
#pragma omp parallel for schedule(guided, num_clusters / num_threads) private(i)
	for (i = 0; i < num_clusters; i++)
	{
		centroids_group[i] = 0; /* used to count the cluster members. */
		centroids[0][i] = 0;	/* used for x value totals. */
		centroids[1][i] = 0;	/* used for y value totals. */
	}
}

void AddToClusterTotal(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes)
{
	int i, clusterIndex;
	for (i = 0; i < num_pts; i++)
	{
		clusterIndex = pts_group[i];
		centroids_group[clusterIndex]++;
		centroids[0][clusterIndex] += pts[0][i];
		centroids[1][clusterIndex] += pts[1][i];
	}
}

void DivideEachClusterTotal(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes)
{
	int i;
	for (i = 0; i < num_clusters; i++)
	{
		centroids[0][i] /= centroids_group[i];
		centroids[1][i] /= centroids_group[i];
	}
}

void DivideEachClusterTotal_omp_static(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int num_threads)
{

	int i;
#pragma omp for schedule(static, num_clusters / num_threads) private(i)
	for (i = 0; i < num_clusters; i++)
	{
		centroids[0][i] /= centroids_group[i];
		centroids[1][i] /= centroids_group[i];
	}
}

void DivideEachClusterTotal_omp_dynamic(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int num_threads)
{

	int i;
#pragma omp for schedule(dynamic, num_clusters / num_threads) private(i)
	for (i = 0; i < num_clusters; i++)
	{
		centroids[0][i] /= centroids_group[i];
		centroids[1][i] /= centroids_group[i];
	}
}

void DivideEachClusterTotal_omp_guided(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int num_threads)
{

	int i;
#pragma omp for schedule(guided, num_clusters / num_threads) private(i)
	for (i = 0; i < num_clusters; i++)
	{
		centroids[0][i] /= centroids_group[i];
		centroids[1][i] /= centroids_group[i];
	}
}

int FindNearestCentroid(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes)
{
	int i, j, clusterIndex = 0;
	int changes = 0;
	double x, y, d, min_d = 0x7f800000;
	for (i = 0; i < num_pts; i++)
	{
		min_d = 0x7f800000; // IEEE 754 +infinity
		for (j = 0; j < num_clusters; j++)
		{
			x = centroids[0][j] - pts[0][i];
			y = centroids[1][j] - pts[1][i];
			d = x * x + y * y;
			if (d < min_d)
			{
				min_d = d;
				clusterIndex = j;
			}
		}
		if (clusterIndex != pts_group[i])
		{
			pts_group[i] = clusterIndex;
			changes++;
		}
	}
	return changes;
}

int FindNearestCentroid_omp_static(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int num_threads)
{
	int clusterIndex = 0;
	int changes = 0;
	double x, y, d, min_d = 0x7f800000;
	int i;
	int j;
#pragma omp parallel for reduction(+: min_d, changes) private(clusterIndex, x, y, d, i, j) schedule(static, num_pts / num_threads)
	for (i = 0; i < num_pts; i++)
	{
		min_d = 0x7f800000; // IEEE 754 +infinity

		for (j = 0; j < num_clusters; j++)
		{
			x = centroids[0][j] - pts[0][i];
			y = centroids[1][j] - pts[1][i];
			d = x * x + y * y;
			if (d < min_d)
			{
				min_d = d;
				clusterIndex = j;
			}
		}

		if (clusterIndex != pts_group[i])
		{
			pts_group[i] = clusterIndex;
			changes++;
		}
	}

	return changes;
}

int FindNearestCentroid_omp_dynamic(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int num_threads)
{
	int clusterIndex = 0;
	int changes = 0;
	double x, y, d, min_d = 0x7f800000;
	int i = 0;
	int j = 0;

#pragma omp parallel for reduction(+:changes) private(clusterIndex,d,i,j,x,y,min_d) schedule(dynamic, num_pts / num_threads) 
	for (i = 0; i < num_pts; i++)
	{
		min_d = 0x7f800000; // IEEE 754 +infinity

		for (j = 0; j < num_clusters; j++)
		{
			x = centroids[0][j] - pts[0][i];
			y = centroids[1][j] - pts[1][i];
			d = x * x + y * y;
			if (d < min_d)
			{
				min_d = d;
				clusterIndex = j;
			}
		}
		
		if (clusterIndex != pts_group[i])
		{
			pts_group[i] = clusterIndex;
			changes++;
		}
	}

	return changes;
}

int FindNearestCentroid_omp_guided(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int num_threads)
{
	int clusterIndex = 0;
	int changes = 0;
	double x, y, d, min_d = 0;
	int i;
	int j;

#pragma omp parallel for reduction(+: min_d, changes) private(clusterIndex, x, y, d,i,j) schedule(guided, num_pts / num_threads) 
	for (i = 0; i < num_pts; i++)
	{
		min_d = 0x7f800000; // IEEE 754 +infinity

		for (j = 0; j < num_clusters; j++)
		{
			x = centroids[0][j] - pts[0][i];
			y = centroids[1][j] - pts[1][i];
			d = x * x + y * y;
			if (d < min_d)
			{
				min_d = d;
				clusterIndex = j;
			}
		}

		if (clusterIndex != pts_group[i])
		{
			pts_group[i] = clusterIndex;
			changes++;
		}
	}

	return changes;
}

/*------------------------------------------------------------------
	lloyd
	This function clusters the data using Lloyd's K-Means algorithm
	after selecting the intial centroids.
	It returns a pointer to the memory it allocates containing
	the array of cluster centroids.
---------------------------------------------------------------------*/
void lloyd(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes)
{
	int i;
	int changes;
	int acceptable = num_pts / 1000; /* The maximum point changes acceptable. */

	if (num_clusters == 1 || num_pts <= 0 || num_clusters > num_pts)
		return;

	if (maxTimes < 1)
		maxTimes = 1;
	/*Assign initial clustering randomly using the Random Partition method*/

	AssignInitialClusteringRandomly(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);

	do
	{
		/* Calculate the centroid of each cluster.
		  ----------------------------------------*/
		/* Initialize the x, y and cluster totals. */
		InitializeClusterXY(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		/* Add each observation's x and y to its cluster total. */
		AddToClusterTotal(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		/* Divide each cluster's x and y totals by its number of data points. */
		DivideEachClusterTotal(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		/* Find each data point's nearest centroid */
		changes = FindNearestCentroid(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		// if(maxTimes == 100)
		// print_eps_v3(pts, pts_group, num_pts, centroids, centroids_group, num_clusters,"before.eps");
		maxTimes--;
	} while ((changes > acceptable) && (maxTimes > 0));

	/* Set each centroid's group index */
	for (i = 0; i < num_clusters; i++)
		centroids_group[i] = i;
} /* end, lloyd */

/**
 * @brief Multithreaded version of lloyd's algorithm with static distribution of work
 *
 * @param pts
 * @param pts_group
 * @param num_pts
 * @param centroids
 * @param centroids_group
 * @param num_clusters
 * @param maxTimes
 * @param noThreads
 */
void lloyd_omp_static(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int noThreads)
{
	/**
	 * @brief Construct a new omp set num threads object with noThread threads
	 *
	 */
	omp_set_num_threads(noThreads);
	omp_set_nested(1);

	int i;
	int changes;
	int acceptable = num_pts / 1000; /* The maximum point changes acceptable. */

	if (num_clusters == 1 || num_pts <= 0 || num_clusters > num_pts)
		return;

	if (maxTimes < 1)
		maxTimes = 1;
	/*Assign initial clustering randomly using the Random Partition method*/

	AssignInitialClusteringRandomly_omp_static(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, noThreads);

	//AssignInitialClusteringRandomly(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);

	do
	{
		/* Calculate the centroid of each cluster.
		  ----------------------------------------*/
		/* Initialize the x, y and cluster totals. */
		InitializeClusterXY(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		//InitializeClusterXY_omp_static(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, noThreads);
		/* Add each observation's x and y to its cluster total. */
		AddToClusterTotal(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		//AddToClusterTotal_omp_static(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, noThreads);

		/* Divide each cluster's x and y totals by itss number of data points. */

		DivideEachClusterTotal(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		//DivideEachClusterTotal_omp_static(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes,noThreads);
		/* Find each data point's nearest centroid */

		changes = FindNearestCentroid_omp_static(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, noThreads);
		// changes = FindNearestCentroid(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);

		// if(maxTimes == 100)
		// print_eps_v3(pts, pts_group, num_pts, centroids, centroids_group, num_clusters,"before.eps");
		//#pragma omp barrier

		maxTimes--;
		//#pragma omp barrier
	} while ((changes > acceptable) && (maxTimes > 0));

	for (i = 0; i < num_clusters; i++)
		centroids_group[i] = i;

} /* end, lloyd */

void lloyd_omp_dynamic(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int noThreads)
{
	/**
	 * @brief Construct a new omp set num threads object with noThread threads
	 *
	 */
	omp_set_num_threads(noThreads);
	omp_set_nested(1);

	int i;
	int changes;
	int acceptable = num_pts / 1000; /* The maximum point changes acceptable. */

	if (num_clusters == 1 || num_pts <= 0 || num_clusters > num_pts)
		return;

	if (maxTimes < 1)
		maxTimes = 1;
	/*Assign initial clustering randomly using the Random Partition method*/

	// AssignInitialClusteringRandomly_omp_static(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, noThreads);

	AssignInitialClusteringRandomly_omp_dynamic(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes,noThreads);

	do
	{
		/* Calculate the centroid of each cluster.
		  ----------------------------------------*/
		/* Initialize the x, y and cluster totals. */
		InitializeClusterXY(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		// InitializeClusterXY_omp_static(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, noThreads);
		/* Add each observation's x and y to its cluster total. */
		AddToClusterTotal(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		// AddToClusterTotal_omp_static(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, noThreads);

		/* Divide each cluster's x and y totals by its number of data points. */

		//DivideEachClusterTotal_omp_dynamic(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes,noThreads);
		DivideEachClusterTotal(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		/* Find each data point's nearest centroid */

		changes = FindNearestCentroid_omp_dynamic(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, noThreads);
		// changes = FindNearestCentroid(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);

		// if(maxTimes == 100)
		// print_eps_v3(pts, pts_group, num_pts, centroids, centroids_group, num_clusters,"before.eps");
		//#pragma omp barrier

		maxTimes--;
		//#pragma omp barrier
	} while ((changes > acceptable) && (maxTimes > 0));
	for (i = 0; i < num_clusters; i++)
		centroids_group[i] = i;

} /* end, lloyd */

void lloyd_omp_guided(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int noThreads)
{
	/**
	 * @brief Construct a new omp set num threads object with noThread threads
	 *
	 */
	omp_set_num_threads(noThreads);
	omp_set_nested(1);

	int i;
	int changes;
	int acceptable = num_pts / 1000; /* The maximum point changes acceptable. */

	if (num_clusters == 1 || num_pts <= 0 || num_clusters > num_pts)
		return;

	if (maxTimes < 1)
		maxTimes = 1;
	/*Assign initial clustering randomly using the Random Partition method*/

	// AssignInitialClusteringRandomly_omp_static(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, noThreads);

	AssignInitialClusteringRandomly_omp_guided(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, noThreads);

	do
	{
		/* Calculate the centroid of each cluster.
		  ----------------------------------------*/
		/* Initialize the x, y and cluster totals. */
		//InitializeClusterXY_omp_guided(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, noThreads);
		InitializeClusterXY(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		/* Add each observation's x and y to its cluster total. */
		AddToClusterTotal(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		DivideEachClusterTotal(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
	
		changes = FindNearestCentroid_omp_guided(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, noThreads);
	
		maxTimes--;
		
	} while ((changes > acceptable) && (maxTimes > 0));


	for (i = 0; i < num_clusters; i++)
		centroids_group[i] = i;

} /* end, lloyd */

void print_centroids_v3(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters)
{
	int i;
	for (i = 0; i < num_pts; i++)
		centroids_group[(int)pts_group[i]]++;
	for (i = 0; i < num_clusters; i++)
		printf("\n%% Group:%d #ofPoints %d:\t\tcentroids.x:%f\tcentroids.y:%f", i, centroids_group[i], centroids[0][i], centroids[1][i]);
	printf("\n");
}
/*-------------------------------------------------------
	main
-------------------------------------------------------*/
int main(int argc, char **argv)
{
	int num_pts = NUMBER_OF_POINTS;
	int num_clusters = NUMBER_OF_CLUSTERS;
	int maxTimes = MAXIMUM_ITERATIONS;
	int noOfThreads = NO_OF_THREADS;
	int i, nrows = 2;
	double radius = RADIUS;
	double **pts;
	unsigned int *pts_group;
	double **centroids;
	unsigned int *centroids_group;
	if (argc == 4)
	{
		num_pts = atoi(argv[1]);
		num_clusters = atoi(argv[2]);
		noOfThreads = atoi(argv[3]);
	}
	else if (argc == 5)
	{
		num_pts = atoi(argv[1]);
		num_clusters = atoi(argv[2]);
		maxTimes = atoi(argv[3]);
		noOfThreads = atoi(argv[4]);
	}
	else
	{
		printf("%%*** RUNNING WITH DEFAULT VALUES ***\n");
		printf("%%Execution: ./a.out Number_of_Points Number_of_Clusters\n");
	}
	printf("%%Number of Points:%d, Number of Clusters:%d, maxTimes:%d,  radious:%4.2f\n", num_pts, num_clusters, maxTimes, radius);
	/* Generate the observations */
	printf("%%SERIAL: Kmeans_initData\n");
	pts = Kmeans_initData_v3(num_pts, radius);
	pts_group = malloc(num_pts * sizeof(unsigned int));
	centroids = malloc(nrows * sizeof(double *));
	for (i = 0; i < nrows; i++)
		centroids[i] = malloc(num_clusters * sizeof(double));
	centroids_group = malloc(num_clusters * sizeof(unsigned int));

	startTime(0);
	/* Cluster using the Lloyd algorithm and K-Means++ initial centroids. */
	// lloyd(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);

	lloyd_omp_static(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, noOfThreads);

	stopTime(0);
	/* Print the results */
	print_centroids_v3(pts, pts_group, num_pts, centroids, centroids_group, num_clusters);
	//print_eps_v3(pts, pts_group, num_pts, centroids, centroids_group, num_clusters,"after.eps");
	printf("Static :");
	elapsedTime(0);
	free(pts);
	free(centroids);

	return 0;
}