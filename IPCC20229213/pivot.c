#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<mpi.h>
#include<omp.h>

// Calculate sum of distance while combining different pivots. Complexity : O( n^2 )
double SumDistance(const int k, const int n, const int dim, double* coord, int* pivots, double *redist){

    double chebyshevSum = 0;
    // #pragma omp parallel for reduction (+: chebyshevSum)
    for(int i=0; i<n; i++){
        int j;
        double idist[k];
        for (int l = 0; l < k; l++) {
            idist[l] = redist[i*n+pivots[l]];
        }
        for(j=0; j<i; j++){
            double chebyshev = 0;
            int ki;
            for(ki=0; ki<k; ki++){
                double dis = fabs(idist[ki] - redist[j*n+pivots[ki]]);
                chebyshev = dis>chebyshev ? dis : chebyshev;
            }
            chebyshevSum += chebyshev;
           
        }
    }
    chebyshevSum*=2;
   
    return chebyshevSum;
}

// Recursive function Combination() : combine pivots and calculate the sum of distance while combining different pivots.
// ki  : current depth of the recursion
// k   : number of pivots
// n   : number of points
// dim : dimension of metric space
// M   : number of combinations to store
// coord  : coordinates of points
// pivots : indexes of pivots
// maxDistanceSum  : the largest M distance sum
// maxDisSumPivots : the top M pivots combinations
// minDistanceSum  : the smallest M distance sum
// minDisSumPivots : the bottom M pivots combinations
int cnt = 0;
void Combination(const int worldSize, const int worldRank, int ki, const int k, const int n, const int dim, const int M, double* coord, int* pivots,
                 double* maxDistanceSum, int* maxDisSumPivots, double* minDistanceSum, int* minDisSumPivots, double *redist){
    if(ki==k) {
        if (cnt % worldSize != worldRank) {
            cnt = (cnt + 1) % worldSize;
            return;
        }
        cnt = (cnt + 1) % worldSize;

        // Calculate sum of distance while combining different pivots.
        double distanceSum = SumDistance(k, n, dim, coord, pivots, redist);

        // put data at the end of array
        maxDistanceSum[M] = distanceSum;
        minDistanceSum[M] = distanceSum;
        int kj;
        for(kj=0; kj<k; kj++){
            maxDisSumPivots[M*k + kj] = pivots[kj];
        }
        for(kj=0; kj<k; kj++){
            minDisSumPivots[M*k + kj] = pivots[kj];
        }
        // sort
        int a;
        for(a=M; a>0; a--){
            if(maxDistanceSum[a] > maxDistanceSum[a-1]){
                double temp = maxDistanceSum[a];
                maxDistanceSum[a] = maxDistanceSum[a-1];
                maxDistanceSum[a-1] = temp;
                int kj;
                for(kj=0; kj<k; kj++){
                    int temp = maxDisSumPivots[a*k + kj];
                    maxDisSumPivots[a*k + kj] = maxDisSumPivots[(a-1)*k + kj];
                    maxDisSumPivots[(a-1)*k + kj] = temp;
                }
            }
        }
        for(a=M; a>0; a--){
            if(minDistanceSum[a] < minDistanceSum[a-1]){
                double temp = minDistanceSum[a];
                minDistanceSum[a] = minDistanceSum[a-1];
                minDistanceSum[a-1] = temp;
                int kj;
                for(kj=0; kj<k; kj++){
                    int temp = minDisSumPivots[a*k + kj];
                    minDisSumPivots[a*k + kj] = minDisSumPivots[(a-1)*k + kj];
                    minDisSumPivots[(a-1)*k + kj] = temp;
                }
            }
        }
        
        return;
    }

    // Recursively call Combination() to combine pivots
    // int i;
    // #pragma omp parallel for
    for(int i=pivots[ki-1]+1; i<n; i++) {
        pivots[ki] = i;
        Combination(worldSize, worldRank, ki+1, k, n, dim, M, coord, pivots, 
                    maxDistanceSum, maxDisSumPivots, minDistanceSum, minDisSumPivots, redist);
    }
}

int main(int argc, char* argv[]){
    // filename : input file namespace
    char* filename = (char*)"uniformvector-2dim-5h.txt";
    if( argc==2 ) {
        filename = argv[1];
    }  else if(argc != 1) {
        printf("Usage: ./pivot <filename>\n");
        return -1;
    }
    // M : number of combinations to store
    const int M = 1000;
    // dim : dimension of metric space
    int dim;
    // n : number of points
    int n;
    // k : number of pivots
    int k;

    // Read parameter
    FILE* file = fopen(filename, "r");
    if( file == NULL ) {
        printf("%s file not found.\n", filename);
        return -1;
    }
    fscanf(file, "%d", &dim);
    fscanf(file, "%d", &n);
    fscanf(file, "%d", &k);
    printf("dim = %d, n = %d, k = %d\n", dim, n, k);

    // Start timing
    struct timeval start;

    // Read Data
    double* coord = (double*)malloc(sizeof(double) * dim * n);
    int i;
    for(i=0; i<n; i++){
        int j;
        for(j=0; j<dim; j++){
            fscanf(file, "%lf", &coord[i*dim + j]);
        }
    }
    fclose(file);
    gettimeofday(&start, NULL);

    MPI_Init(NULL, NULL);
    int worldSize, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // printf("worldSize = %d, rank = %d, max_threads = %d\n", worldSize, rank, omp_get_max_threads());
    omp_set_num_threads(omp_get_max_threads());


    // struct timeval start1;
    // gettimeofday(&start1, NULL);

    double *redist;
    redist=(double *)malloc(sizeof(double)*n*n);
    // if (rank == 1) {
        for(int i=0;i<n;i++){
            redist[i*n+i]=0;
        }
        #pragma omp parallel for
        for(int i=0;i<n;i++){
            for(int j=i+1;j<n;j++){
                double distance=0;
                for(int k=0;k<dim;k++){
                    distance+=(coord[i*dim+k]-coord[j*dim+k])*(coord[i*dim+k]-coord[j*dim+k]);
                }
                redist[i*n+j]=redist[j*n+i]=sqrt(distance);
                
            }
        }
    // }
    // MPI_Bcast(redist, n*n, MPI_DOUBLE, 1, MPI_COMM_WORLD);

    // struct timeval end1;
    // gettimeofday(&end1, NULL);
    // printf("RE: Rank %d Using time : %f ms\n", rank, (end1.tv_sec-start1.tv_sec)*1000.0+(end1.tv_usec-start1.tv_usec)/1000.0);


    // maxDistanceSum : the largest M distance sum
    double* maxDistanceSum = (double*)malloc(sizeof(double) * (M+1));
    #pragma omp parallel for
    for(i=0; i<M; i++){
        maxDistanceSum[i] = 0;
    }
    // maxDisSumPivots : the top M pivots combinations
    int* maxDisSumPivots = (int*)malloc(sizeof(int) * k * (M+1));
    #pragma omp parallel for
    for(i=0; i<M; i++){
        int ki;
        for(ki=0; ki<k; ki++){
            maxDisSumPivots[i*k + ki] = 0;
        }
    }
    // minDistanceSum : the smallest M distance sum
    double* minDistanceSum = (double*)malloc(sizeof(double) * (M+1));
    #pragma omp parallel for
    for(i=0; i<M; i++){
        minDistanceSum[i] = __DBL_MAX__;
    }
    // minDisSumPivots : the bottom M pivots combinations
    int* minDisSumPivots = (int*)malloc(sizeof(int) * k * (M+1));
    #pragma omp parallel for
    for(i=0; i<M; i++){
        int ki;
        for(ki=0; ki<k; ki++){
            minDisSumPivots[i*k + ki] = 0;
        }
    }

    // temp : indexes of pivots with dummy array head
    int* temp = (int*)malloc(sizeof(int) * (k+1));
    temp[0] = -1;

    // Main loop. Combine different pivots with recursive function and evaluate them. Complexity : O( n^(k+2) )
    Combination(worldSize, rank, 0, k, n, dim, M, coord, &temp[1], maxDistanceSum, maxDisSumPivots, minDistanceSum, minDisSumPivots, redist);

    double* maxDistanceSumAll;
    int* maxDisSumPivotsAll;
    double* minDistanceSumAll;
    int* minDisSumPivotsAll;

    if (rank == 0) {
        maxDistanceSumAll = (double*)malloc(sizeof(double) * (M+1) * worldSize);
        maxDisSumPivotsAll = (int*)malloc(sizeof(int) * k * (M+1) * worldSize);
        minDistanceSumAll = (double*)malloc(sizeof(double) * (M+1) * worldSize);
        minDisSumPivotsAll = (int*)malloc(sizeof(int) * k * (M+1) * worldSize);
    }

    MPI_Gather(maxDistanceSum, M, MPI_DOUBLE, maxDistanceSumAll, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(maxDisSumPivots, k * M, MPI_INT, maxDisSumPivotsAll, k * M, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(minDistanceSum, M, MPI_DOUBLE, minDistanceSumAll, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(minDisSumPivots, k * M, MPI_INT, minDisSumPivotsAll, k * M, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        int *p, *q;
        p = (int *) malloc(sizeof(int) * (worldSize + 1));
        q = (int *) malloc(sizeof(int) * (worldSize + 1));
        for (int i = 0; i < worldSize; i++) {
            p[i] = q[i] = i * M;
        }
        for (int i = 0; i < M; i++) {
            int maxi = 0, mini = 0;
            for (int j = 1; j < worldSize; j++) {
                maxi = (maxDistanceSumAll[p[j]] > maxDistanceSumAll[p[maxi]]) ? j : maxi;
                mini = (minDistanceSumAll[q[j]] < minDistanceSumAll[q[mini]]) ? j : mini;
            }
            maxDistanceSum[i] = maxDistanceSumAll[p[maxi]];
            for (int ki = 0; ki < k; ki++) {
                maxDisSumPivots[i * k + ki] = maxDisSumPivotsAll[p[maxi] * k + ki];
            }
            p[maxi]++;

            minDistanceSum[i] = minDistanceSumAll[q[mini]];
            for (int ki = 0; ki < k; ki++) {
                minDisSumPivots[i * k + ki] = minDisSumPivotsAll[q[mini] * k + ki];
            }
            q[mini]++;
        }
    }

    MPI_Finalize();

    // End timing
    struct timeval end;
    gettimeofday (&end, NULL);
    printf("Rank %d Using time : %f ms\n", rank, (end.tv_sec-start.tv_sec)*1000.0+(end.tv_usec-start.tv_usec)/1000.0);

    if (rank == 0) {
        // Store the result
        FILE* out = fopen("result.txt", "w");
        for(i=0; i<M; i++){
            int ki;
            for(ki=0; ki<k-1; ki++){
                fprintf(out, "%d ", maxDisSumPivots[i*k + ki]);
            }
            fprintf(out, "%d\n", maxDisSumPivots[i*k + k-1]);
        }
        for(i=0; i<M; i++){
            int ki;
            for(ki=0; ki<k-1; ki++){
                fprintf(out, "%d ", minDisSumPivots[i*k + ki]);
            }
            fprintf(out, "%d\n", minDisSumPivots[i*k + k-1]);
        }
        fclose(out);

        // Log
        int ki;
        printf("max : ");
        for(ki=0; ki<k; ki++){
            printf("%d ", maxDisSumPivots[ki]);
        }
        printf("%lf\n", maxDistanceSum[0]);
        printf("min : ");
        for(ki=0; ki<k; ki++){
            printf("%d ", minDisSumPivots[ki]);
        }
        printf("%lf\n", minDistanceSum[0]);
    }


    return 0;
}