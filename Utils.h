#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>

using namespace cv;
using namespace std;

static class Utils
{
public:
	Utils(void);
	virtual ~Utils(void);
	static Mat toGrayscale(InputArray);
	static void Utils::read_csv(const string&, vector<Mat>&, vector<int>&, char separator = ';');
	static Mat norm_0_255(const Mat&);
	static Mat Utils::tan_triggs_preprocessing(InputArray src,
        float alpha = 0.1, float tau = 10.0, float gamma = 0.2, int sigma0 = 1,
        int sigma1 = 2);
	static void split(const string& str, const string& delim, vector<string>& parts);
	static int readOption(int argc, char** argv, int *i, const char* opt);
	static int readOptionString(int argc, char** argv, int *i, const char* opt, char** arg);
	static void clParseError(int argc, char **argv, int i, char* message);
	static int checkBadOption(int argc, char** argv, int *i);

	// used to retrieve image fullpaths from a csv file
	static void getPathsFromCSV( const string& train_csv_path, vector<string>& paths, char separator = ';' );

	static std::vector<std::pair<double,int>> get_pairs(std::vector<double> vec);

	static void getCurrentDateTime( char* result, int sizeInBytes );
};

