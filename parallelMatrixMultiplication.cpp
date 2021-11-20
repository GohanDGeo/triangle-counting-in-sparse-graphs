// parallelMatrixMultiplication.cpp : Contains sequential and parallel functions to compute triangles of a graph, using its adjacency matrix.
// GEORGIOS KOUTROUMPIS, AEM: 9668 
// TO RUN THE PROGRAM: g++ -o parallelMatrixMultiplication.exe parallelMatrixMultiplication.cpp  -fcilkplus -lgomp -O3 -fopenmp -lpthread
//					   ./parallelMatrixMultiplication.exe
//cmd arguments: 
//1st -> Method to use (0: cilk, 1: openMP, 2: pThreads, 3: sequential)
//2nd -> No of threads to use (does not matter for sequential)
//3rd -> Matrix to load	(0: "com-Youtube.mtx" , 1: "belgium_osm.mtx" , 2: "dblp-2010.mtx" , 3: "mycielskian13.mtx" , 4: "NACA0015.mtx")
//!WARNING! No checks on the legitimacy of the cmd arguments is made, please use them with care

#include <iostream>
#include <fstream>
#include <algorithm> 
#include <math.h>
#include "helper.h"
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/SparseExtra>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <numeric>
#include "csv_helper.h"
#include<string>  
using namespace std::chrono;
using namespace std;

//A struct containing arguments to pass to the pThread callback function
struct pThreads_args {
	vector<int> row_ind;		//Row indices vector of the CSC
	vector<int> col_ptr;		//Col ptr vector of the CSC
	vector<int> *mult_results;	//Pointer to the vector that holds the multiplication results
	int start;					//The start index of nnz element for the thread to start
	int end;					//The end index of the nnz element of the thread to end
};

//Callback function for the pThread implementation
//Args;
//@arg -> a pThreads_args struct, explained above
void* multi(void* arg)
{
	//Extract the arguments from teh struct
	struct pThreads_args* args = (struct pThreads_args*)arg;
	
	vector<int> row_ind = args->row_ind;
	vector<int> col_ptr = args->col_ptr;
	vector<int> *mult_results = (args->mult_results);

	int start = args->start;
	int end = args->end;

	//Implements the first sequential algorithm for triangle counting, for a batch of size end-start
	int colIdx = 0;
	for (int k = start; k < end; k++)
	{

		//Find the column index of the current nnz element
		while (k >= col_ptr[colIdx + 1])
			colIdx++;
		
		//Get its row
		int row = row_ind[k];

		//Construct a vector of the row of the element and its column
		vector<int>	v1 = vector<int>(row_ind.begin() + col_ptr[colIdx], row_ind.begin() + col_ptr[colIdx + 1]);

		vector<int> v2(row_ind.begin() + col_ptr[row], row_ind.begin() + col_ptr[row + 1]);

		//Find the common elements of the two vectors
		vector<int> v_intersection;
		std::set_intersection(
			v1.begin(), v1.end(),
			v2.begin(), v2.end(),
			std::back_inserter(v_intersection)
		);

		int common_elements_size = v_intersection.size();

		//Put the common element count to the result vector
		if (common_elements_size > 0)
		{
			mult_results->at(k) = common_elements_size;
		}
	}


}


//A sequential implementation for triangle counting
//Args:
//@mat -> The adjacency matrix of the graph on which to perform triangle counting on, in CSC format
int sequential_triangle_counting(CSC mat)
{

	//Extract the CSC matrix' row indices and col ptr vectors
	vector<int> row_ind = mat.row_ind;
	vector<int> col_ptr = mat.col_ptr;

	//Holds the column index the alg is in
	int colIdx = 0;

	//Vector holding the results of the multiplication
	vector<int> mult_results = vector<int>(row_ind.size());

	//Loop through each nnz element
	for (int k = 0; k < row_ind.size(); k++)
	{
		//Find the correct column index of the current element
		//by increasing it until the next column has more elements 
		//than the current k (nnz element number)
		while (k >= col_ptr[colIdx + 1])
		{
			colIdx++;
		}

		//Get the row of the current element
		int row = row_ind[k];

		//Get current element's row's and column's column and row indices, respectively
		//(matrix is symmetric)
		vector<int> v1 (row_ind.begin() + col_ptr[colIdx], row_ind.begin() + col_ptr[colIdx + 1]);
		vector<int> v2(row_ind.begin() + col_ptr[row], row_ind.begin() + col_ptr[row + 1]);

		//Find the number of common indices. 
		//Since all values are 1, the number of common indices is the value of element (row, colIdx) of the resulting matrix
		vector<int> v_intersection;
		std::set_intersection(
			v1.begin(), v1.end(),
			v2.begin(), v2.end(),
			std::back_inserter(v_intersection)
		);

		int common_elements_size = v_intersection.size();

		if (common_elements_size > 0)
		{
			mult_results.at(k) = common_elements_size;
		}
	}

	//Calculate the vector c =  C * e/2, where the vector mult_results holds the matrix C values.
	vector<int>c = vector<int>(col_ptr.size() - 1, 0);
	for (int i = 0; i < col_ptr.size() - 1; i++) {
		vector<int> v = vector<int>(mult_results.begin() + col_ptr[i], mult_results.begin() + col_ptr[i + 1]);
		c.at(i) = accumulate(v.begin(), v.end(), 0) / 2;
	}

	//The number of triangles is the sum of vector c's elements divided by 3.
	int triangles = accumulate(c.begin(), c.end(), 0) / 3;
	return triangles;
}

//A parallel implementation for triangle counting using cilk
//Args:
//@mat -> The adjacency matrix of the graph on which to perform triangle counting on, in CSC format
//@threads -> The number of threads to use. THIS IS NOT USED IN THE FUNCTION, BUT IS CHANGED IN THE MAIN FUNCTION.
//			  Here for consistency with the other parallel implementations.
int parallel_clik_triangle_counting(CSC mat, int threads)
{
	//Extract the CSC matrix' row indices and col ptr vectors
	vector<int> row_ind = mat.row_ind;
	vector<int> col_ptr = mat.col_ptr;

	//Vector holding the results of the multiplication
	vector<int> mult_results = vector<int>(row_ind.size());

	//Asynchonously loop through all nnz elements of the matrix
	cilk_for(int l = 0; l < col_ptr.size() - 1; l++)
	{
		cilk_for(int k = col_ptr[l]; k < col_ptr[l + 1]; k++)
		{
			//Get current element's row index
			int row = row_ind[k];
			
			//Get current element's row's and column's column and row indices, respectively
			//(matrix is symmetric)
			vector<int> v1 = vector<int>(row_ind.begin() + col_ptr[l], row_ind.begin() + col_ptr[l + 1]);
			vector<int> v2(row_ind.begin() + col_ptr[row], row_ind.begin() + col_ptr[row + 1]);
			vector<int> v_intersection;
			std::set_intersection(
				v1.begin(), v1.end(),
				v2.begin(), v2.end(),
				std::back_inserter(v_intersection)
			);

			// Find the number of common indices.
			//Since all values are 1, the number of common indices is the value of element (row, colIdx) of the resulting matrix
			int common_elements_size = v_intersection.size();

			if (common_elements_size > 0)
			{
				mult_results.at(k) = common_elements_size;
			}
		}
	}

	//Calculate the vector c =  C * e/2, where the vector mult_results holds the matrix C values.
	vector<int>c = vector<int>(col_ptr.size() - 1, 0);
	for (int i = 0; i < col_ptr.size() - 1; i++) {
		vector<int> v = vector<int>(mult_results.begin() + col_ptr[i], mult_results.begin() + col_ptr[i + 1]);
		c.at(i) = accumulate(v.begin(), v.end(), 0) / 2;
	}

	//The number of triangles is the sum of vector c's elements divided by 3.
	int triangles = accumulate(c.begin(), c.end(), 0) / 3;
	return triangles;
}


//A parallel implementation for triangle counting using openMP
//Args:
//@mat -> The adjacency matrix of the graph on which to perform triangle counting on, in CSC format
//@threads -> Number of threads to use
int parallel_openMP_triangle_counting(CSC mat, int threads)
{
	//Extract the CSC matrix' row indices and col ptr vectors
	vector<int> row_ind = mat.row_ind;
	vector<int> col_ptr = mat.col_ptr;

	//Vector holding the results of the multiplication
	vector<int> mult_results = vector<int>(row_ind.size());

	omp_set_dynamic(0);			  // Explicitly disable dynamic teams
	omp_set_num_threads(threads); // Use *int threads* threads for all consecutive parallel regions
	
	//Parallelize using openMp. Don't wait for second loop, to begin next iteration
	#pragma omp parallel
	for(int l = 0; l < col_ptr.size() - 1; l++)
	{
		#pragma omp for nowait
		for(int k = col_ptr[l]; k < col_ptr[l + 1]; k++)
		{
			//Get current element's row index
			int row = row_ind[k];

			//Get current element's row's and column's column and row indices, respectively
			//(matrix is symmetric)
			vector<int> v1 = vector<int>(row_ind.begin() + col_ptr[l], row_ind.begin() + col_ptr[l + 1]);
			vector<int> v2(row_ind.begin() + col_ptr[row], row_ind.begin() + col_ptr[row + 1]);



			// Find the number of common indices.
			//Since all values are 1, the number of common indices is the value of element (row, colIdx) of the resulting matrix
			vector<int> v_intersection;
			std::set_intersection(
				v1.begin(), v1.end(),
				v2.begin(), v2.end(),
				std::back_inserter(v_intersection)
			);

			int common_elements_size = v_intersection.size();

			if (common_elements_size > 0)
			{
				mult_results.at(k) = common_elements_size;
			}
		}
	}

	//Calculate the vector c =  C * e/2, where the vector mult_results holds the matrix C values.
	vector<int>c = vector<int>(col_ptr.size() - 1, 0);
	for (int i = 0; i < col_ptr.size() - 1; i++) {
		vector<int> v = vector<int>(mult_results.begin() + col_ptr[i], mult_results.begin() + col_ptr[i + 1]);
		c.at(i) = accumulate(v.begin(), v.end(), 0) / 2;

	}

	//The number of triangles is the sum of vector c's elements divided by 3.
	int triangles = accumulate(c.begin(), c.end(), 0) / 3;
	return triangles;
}


//A parallel implementation for triangle counting using openMP
//Args:
//@mat -> The adjacency matrix of the graph on which to perform triangle counting on, in CSC format
//@threads -> Number of threads to use
int parallel_pthreads_triangle_counting(CSC mat, int numOfThreads)
{
	//Extract the CSC matrix' row indices and col ptr vectors
	vector<int> row_ind = mat.row_ind;
	vector<int> col_ptr = mat.col_ptr;

	//Vector holding the results of the multiplication
	vector<int> mult_results = vector<int>(row_ind.size());

	//Get the number of nnz elements
	int n = row_ind.size();
	//Determine the batch size for each thread
	int threadLoops = n / numOfThreads;

	//Create an array of threads
	pthread_t threads[numOfThreads];
	//Create a vector of pThread_args, one element for each thread
	vector<pThreads_args> arguments(numOfThreads);

	//Create and start each thread, except the last
	for (int i = 0; i < numOfThreads-1; i++)
	{
		//Pass the CSC's vectors
		arguments.at(i).col_ptr = col_ptr;
		arguments.at(i).row_ind = row_ind;

		//Pass a pointer to the result vector
		arguments.at(i).mult_results = &mult_results;

		//Pass the start and end elements of the batch
		arguments.at(i).start = i * threadLoops;
		arguments.at(i).end = (i+1) * threadLoops;

		//Initialize and start thread, check for errors.
		int ret = pthread_create(&threads[i], NULL, multi, (void*)&arguments.at(i));
		if (ret != 0) {
			cout << "Error: pthread_create() failed\n";
			exit(EXIT_FAILURE);
		}
	}

	//Do the same for the last thread, but with an end == n
	int lastThreadIdx = numOfThreads - 1;
	arguments.at(lastThreadIdx).col_ptr = col_ptr;
	arguments.at(lastThreadIdx).row_ind = row_ind;
	arguments.at(lastThreadIdx).mult_results = &mult_results;

	arguments.at(lastThreadIdx).start = lastThreadIdx * threadLoops;
	arguments.at(lastThreadIdx).end = n;
	int ret = pthread_create(&threads[lastThreadIdx], NULL, multi, (void*)&arguments.at(lastThreadIdx));
	if (ret != 0) {
		cout << "Error: pthread_create() failed\n";
		exit(EXIT_FAILURE);
	}

	//Join all threads (Wait for each thread to finish before moving on)
	for (int i = 0; i < numOfThreads; i++)
		pthread_join(threads[i], NULL);
	
	//Since the mult_results vector is populated, continue to find c

	//Calculate the vector c =  C * e/2, where the vector mult_results holds the matrix C values.
	vector<int>c = vector<int>(col_ptr.size() - 1, 0);
	for (int i = 0; i < col_ptr.size() - 1; i++) {
		vector<int> v = vector<int>(mult_results.begin() + col_ptr[i], mult_results.begin() + col_ptr[i + 1]);
		c.at(i) = accumulate(v.begin(), v.end(), 0) / 2;

	}

	//The number of triangles is the sum of vector c's elements divided by 3.
	int triangles = accumulate(c.begin(), c.end(), 0) / 3;
	return triangles;
}

//A second sequential implementation for triangle counting
//Args:
//@mat -> The adjacency matrix of the graph on which to perform triangle counting on, in CSC format
int second_sequential_triangle_counting(CSC mat)
{
	//Extract the CSC matrix' row indices and col ptr vectors
	vector<int> row_ind = mat.row_ind;
	vector<int> col_ptr = mat.col_ptr;

	//Vector holding the results of the multiplication
	vector<int> mult_results = vector<int>(row_ind.size());

	//Loop through all nnz elements of the matrix
	for(int l = 0; l < col_ptr.size() - 1; l++)
	{
		for(int k = col_ptr[l]; k < col_ptr[l + 1]; k++)
		{
			//Get current element's row index
			int row = row_ind[k];

			//Get current element's row's and column's column and row indices, respectively
			//(matrix is symmetric)
			vector<int> v1 = vector<int>(row_ind.begin() + col_ptr[l], row_ind.begin() + col_ptr[l + 1]);
			vector<int> v2(row_ind.begin() + col_ptr[row], row_ind.begin() + col_ptr[row + 1]);
			vector<int> v_intersection;
			std::set_intersection(
				v1.begin(), v1.end(),
				v2.begin(), v2.end(),
				std::back_inserter(v_intersection)
			);

			// Find the number of common indices.
			//Since all values are 1, the number of common indices is the value of element (row, colIdx) of the resulting matrix
			int common_elements_size = v_intersection.size();
			if (common_elements_size > 0)
			{
				mult_results.at(k) = common_elements_size;
			}
		}

	}

	//Calculate the vector c =  C * e/2, where the vector mult_results holds the matrix C values.
	vector<int>c = vector<int>(col_ptr.size() - 1, 0);
	for (int i = 0; i < col_ptr.size() - 1; i++) {
		vector<int> v = vector<int>(mult_results.begin() + col_ptr[i], mult_results.begin() + col_ptr[i + 1]);
		c.at(i) = accumulate(v.begin(), v.end(), 0) / 2;

	}

	//The number of triangles is the sum of vector c's elements divided by 3.
	int triangles = accumulate(c.begin(), c.end(), 0) / 3;
	return triangles;
}

//The driver function of the program
//cmd arguments: 
//1st -> Method to use (0: cilk, 1: openMP, 2: pThreads, 3: sequential)
//2nd -> No of threads to use (does not matter for sequential)
//3rd -> Matrix to load	(0: "com-Youtube.mtx" , 1: "belgium_osm.mtx" , 2: "dblp-2010.mtx" , 3: "mycielskian13.mtx" , 4: "NACA0015.mtx")
//!WARNING! No checks on the legitimacy of the cmd arguments is made, please use them with care
int main(int argc, char** argv)
{
	//Initialize openmp get time, used to measure duraton of functions
	double omp_get_wtime(void);

	//Vectors holding the filenames for the benchmark data and implementations that can be used
	vector<string> filenames{ "com-Youtube.mtx" , "belgium_osm.mtx" , "dblp-2010.mtx" , "mycielskian13.mtx" , "NACA0015.mtx" };
	vector<string> techniques{  "cilk" , "openMP" , "pThreads", "sequential" };

	//Create an array of the parallel functions
	typedef int (*TriangleCountingFunction) (CSC mat, int threads);
	TriangleCountingFunction functions[] =
	{
		parallel_clik_triangle_counting,
		parallel_openMP_triangle_counting,
		parallel_pthreads_triangle_counting
	};

	//Index of matrix to use
	int fileNo = 0;
	//Index of implementation to use
	int techniqueNo = 0;
	//How many threads to be used (irrelevant for sequential implementations
	int threadCount = 1;


	//Check if command line arguments were used, and use the appropriate implementations and matrix.
	if (argc == 2)
	{
		techniqueNo = atoi(argv[1]);
	}
	else if (argc == 3)
	{
		techniqueNo = atoi(argv[1]);
		threadCount = atoi(argv[2]);
	}
	else if (argc == 4)
	{
		techniqueNo = atoi(argv[1]);
		threadCount = atoi(argv[2]);
		fileNo = atoi(argv[3]);
	}

	string filename = filenames[fileNo];
	cout << "Loading matrix " << filename << "\n";
	CSC mat = readMTXSymPatToCSC(filename);
	cout << "Loaded " << filename << "\n";

	if (techniqueNo == 3)
		cout << "Triangle calculation using sequential algorithm\n";
	else
		cout << "Triangle calculation using " << techniques[techniqueNo] << " and " << threadCount << " threads" << "\n";

	if (techniqueNo == 3)
	{
		double start = omp_get_wtime();
		int triangles = sequential_triangle_counting(mat);
		double stop = omp_get_wtime();
		double duration = stop - start;
		cout << "Triangles: " << triangles << "\n";
		cout << fixed
			<< "Time taken by function: "
			<< stop - start << " seconds" << endl;
	}
	else
	{
		if (techniqueNo == 0)
		{
			string workers = to_string(threadCount);
			__cilkrts_end_cilk();
			__cilkrts_set_param("nworkers", workers.c_str());
		}


		double start = omp_get_wtime();
		int triangles = functions[techniqueNo](mat, threadCount);
		double stop = omp_get_wtime();
		double duration = stop - start;
		cout << "Triangles: " << triangles << "\n";
		cout << fixed
			<< "Time taken by "
			<< techniques[techniqueNo]
			<< ": "
			<< stop - start << " seconds" << endl;
	}
}

