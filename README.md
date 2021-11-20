# Triangle Counting In Sparse Graphs

Contains 2 sequential implementations (that work similary) and 3 parallel implementations for calculating number of triangles in a graph.
Input is a Pattern Symmetrix market matrix file.

To run:
$ g++ -o parallelMatrixMultiplication.exe parallelMatrixMultiplication.cpp  -fcilkplus -lgomp -O3 -fopenmp -lpthread
$ ./parallelMatrixMultiplication.exe

Command line arguments can be used:
1st -> Method to use (0: cilk, 1: openMP, 2: pThreads, 3: sequential)
2nd -> No of threads to use (does not matter for sequential)
3rd -> Matrix to load	(0: "com-Youtube.mtx" , 1: "belgium_osm.mtx" , 2: "dblp-2010.mtx" , 3: "mycielskian13.mtx" , 4: "NACA0015.mtx")
The 3rd argument only if you have the above matrix files in the same directory. If that's not the case, change the filename variable in the main function of the cpp file.
