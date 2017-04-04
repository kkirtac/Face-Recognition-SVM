#include "Utils.h"


Utils::Utils(void)
{
}


Utils::~Utils(void)
{
}


Mat Utils::toGrayscale(InputArray _src) {
    Mat src = _src.getMat();
    // only allow one channel
    if(src.channels() != 1) {
        CV_Error(1, "Only Matrices with one channel are supported");
    }
    // create and return normalized image
    Mat dst;
    cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}


void Utils::read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator) {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(1, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) 
		{
			//Mat im = imread
			images.push_back(imread(path,0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

void Utils::getPathsFromCSV( const string& csv_path, vector<string>& paths, char separator )
{
	std::ifstream file(csv_path.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(1, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) 
		{
			paths.push_back(path);
        }
    }
}


// Normalizes a given image into a value range between 0 and 255.
Mat Utils::norm_0_255(const Mat& src) {
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

//
// Calculates the TanTriggs Preprocessing as described in:
//
//      Tan, X., and Triggs, B. "Enhanced local texture feature sets for face
//      recognition under difficult lighting conditions.". IEEE Transactions
//      on Image Processing 19 (2010), 1635–650.
//
// Default parameters are taken from the paper.
//
Mat Utils::tan_triggs_preprocessing(InputArray src,
        float alpha, float tau, float gamma, int sigma0, int sigma1) {

    // Convert to floating point:
    Mat X = src.getMat();
    X.convertTo(X, CV_32FC1);
    // Start preprocessing:
    Mat I;
    pow(X, gamma, I);
    // Calculate the DOG Image:
    {
        Mat gaussian0, gaussian1;
        // Kernel Size:
        int kernel_sz0 = (3*sigma0);
        int kernel_sz1 = (3*sigma1);
        // Make them odd for OpenCV:
        kernel_sz0 += ((kernel_sz0 % 2) == 0) ? 1 : 0;
        kernel_sz1 += ((kernel_sz1 % 2) == 0) ? 1 : 0;
        GaussianBlur(I, gaussian0, Size(kernel_sz0,kernel_sz0), sigma0, sigma0, BORDER_CONSTANT);
        GaussianBlur(I, gaussian1, Size(kernel_sz1,kernel_sz1), sigma1, sigma1, BORDER_CONSTANT);
        subtract(gaussian0, gaussian1, I);
    }

    {
        double meanI = 0.0;
        {
            Mat tmp;
            pow(abs(I), alpha, tmp);
            meanI = mean(tmp).val[0];

        }
        I = I / pow(meanI, 1.0/alpha);
    }

    {
		double meanI = 0.0;
        {
            Mat tmp;
            pow(min(abs(I), tau), alpha, tmp);
            meanI = mean(tmp).val[0];
        }
        I = I / pow(meanI, 1.0/alpha);
    }

    // Squash into the tanh:
    {
        for(int r = 0; r < I.rows; r++) {
            for(int c = 0; c < I.cols; c++) {
                I.at<float>(r,c) = tanh(I.at<float>(r,c) / tau);
            }
        }
        I = tau * I;
    }
    return I;
}

/* split the string str using delimeter delim and fill the parts vector with the string parts */
void Utils::split(const string& str, const string& delim, vector<string>& parts) {
  size_t start, end = 0;
  while (end < str.size()) {
    start = end;
    while (start < str.size() && (delim.find(str[start]) != string::npos)) {
      start++;  // skip initial whitespace
    }
    end = start;
    while (end < str.size() && (delim.find(str[end]) == string::npos)) {
      end++; // skip to end of word
    }
    if (end-start != 0) {  // just ignore zero-length strings.
      parts.push_back(string(str, start, end-start));
    }
  }
}

/* returns true if opt is parsed and i+=0 */
int Utils::readOption(int argc, char** argv, int *i, const char* opt){
    if((*i) < argc){
        if(_stricmp(argv[(*i)],opt) == 0){
            return 1;
        }
    }
    else{
        clParseError(argc, argv, (*i), "Error parsing command line.");
    }
    
    return 0;
}

/* returns true if opt is parsed and sets arg to point to location in argv that is after flag (the next string) i+=1 */
int Utils::readOptionString(int argc, char** argv, int *i, const char* opt, char** arg){
    if((*i) < argc){
        if(_stricmp(argv[(*i)],opt) == 0){
            (*i) += 1;
            if(*i < argc){
                (*arg) = argv[(*i)];
            } else {
                clParseError(argc, argv, (*i), "Option expects one argument.");
            }

            return 1;
        }
    }
    else{
        clParseError(argc, argv, (*i), "Error parsing command line.");
    }

    return 0;    
}

/* check to see if current argument starts with a dash and if it does output an error and exit. otherwize return 0 */
int Utils::checkBadOption(int argc, char** argv, int *i){
    if((*i) < argc){
        if(argv[(*i)][0] == '-'){
            clParseError(argc, argv, (*i), "Unrecognized option.");
        }
    }
    else{
        clParseError(argc, argv, (*i), "Error parsing command line.");
    }

    return 0;
}

/* Output a command line parsing error */
void Utils::clParseError(int argc, char **argv, int i, char* message){
    fprintf(stdout, "Error: %s\n", message);
    fprintf(stdout, "       for command at command line argument <%d: %s >\n", i, (i < argc) ? argv[i] : "(Error: passed end of line)");
    //usage((0 < argc) ? argv[0] : "(Error: No program name)");
    exit(1);
}

std::vector<std::pair<double,int>> Utils::get_pairs(std::vector<double> vec)
{
	std::vector<std::pair<double,int>> temp;				// vector of pairs

	for(int i = 0; i < vec.size() ; i++)
		temp.push_back(std::pair<double,int>(vec[i],i));

	return temp;
}

void Utils::getCurrentDateTime( char* result, int sizeInBytes )
{
	// get current date and time
	time_t now = time(0);
	tm tstruct = *localtime(&now);

	//char buf[80];

	strftime(result, sizeInBytes, "%Y-%m-%d.%X", &tstruct);

	//return buf;
}
