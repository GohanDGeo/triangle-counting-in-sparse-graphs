//Helper functions to read a symmetric pattern sparse matrix from Matrix Market format

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <math.h>
#include <sstream>

using namespace std;

struct CSC {
	vector<int> row_ind;
	vector<int> col_ptr;
};

void printArray(vector<int> v) {
	for (int i = 0; i < v.size(); i++)
		cout << v[i] << ' ';
	cout << '\n';
}

void printArray(vector<double> v) {
	for (int i = 0; i < v.size(); i++)
		cout << v[i] << ' ';
	cout << '\n';
}

void printCSCMatrix(CSC csc) {
	int idx = 0;
	int n = csc.col_ptr.size() - 1;

	vector<int> col_ptr = csc.col_ptr;
	vector<int> row_ind = csc.row_ind;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (idx < col_ptr[i + 1] && j == row_ind[idx])
			{
				cout << 1 << " ";
				idx++;
			}
			else
				cout << "0 ";
		}
		cout << "\n";
	}
}


CSC readMTXSymPatToCSC(string fileName) {

	vector<int> col_ptr;
	vector<int> row_ind;


	ifstream file(fileName);

	int M, N, L;
	bool isPattern = false;
	bool isSymmetric = false;

	bool foundHeader = false;
	while (file.peek() == '%')
	{
		if (file.peek() == '%' && !foundHeader)
		{
			string line;
			getline(file, line);
			if (line.find("pattern") != std::string::npos)
				isPattern = true;
			if (line.find("symmetric") != std::string::npos)
			{
				isSymmetric = true;
			}
			foundHeader = true;
		}
		else
			file.ignore(2048, '\n');

	}

	file >> M >> N >> L;

	vector<vector<int>> mtxElements(L);

	for (int l = 0; l < L; l++) {

		int row, col;

		file >> row >> col;
		row--;
		col--;

		mtxElements[row].push_back(col);

		if (isSymmetric)
			mtxElements[col].push_back(row);
		
	}

	col_ptr.push_back(0);
	for (int i = 0; i < L; i++) {
		int size = mtxElements[i].size();

		if (size > 0) {
			col_ptr.push_back(size + col_ptr[i]);
			for (int j = 0; j < size; j++) {
				row_ind.push_back(mtxElements[i][j]);
			}
		}
		else {
			col_ptr.push_back(col_ptr[i]);
		}
	}

	CSC matrix;
	matrix.col_ptr = col_ptr;
	matrix.row_ind = row_ind;
	
	return matrix;
}