
#include <opencv2\ml\ml.hpp>
#include <ctime>
#include <cstdlib>
#include <direct.h>
#include "Utils.h"
#include "Logger.h"

using namespace cv;
using namespace std;



class FaceRecognizerSVM
{

public:

	FaceRecognizerSVM();

	FaceRecognizerSVM( bool );


	CvSVMParams param;
	CvSVM svm;

	bool isPerformPCA;

	int train( Mat& trainingData, Mat& trainingClasses, CvSVMParams params );

	float predict( Mat );

};

class Arguments
{

public:

	Arguments();


	void setAlgorithm(string algorithm);
	string getAlgorithm();

	void setInfo(string info);
	string getInfo();

	void setHistType(string histType);
	string getHistType();

	void setLogPath(string logPath);
	string getLogPath();

	void setTrainCsvPath(string trainPath);
	string getTrainCsvPath();

	void setTestCsvPath(string testPath);
	string getTestCsvPath();

	void setModelPath(string modelPath);
	string getModelPath();

	int save( string fullPath );

	int load( string fullPath );

private:

	// the algoritm
	string algorithm;

	// information about the current setup
	string info;

	// illumination normalization type
	string histType;

	// full path of the log file
	string log_path;

	// full path of the training csv file
	string train_csv_path;

	// full path of the test csv file
	string test_csv_path;

	// model external path
	string model_path;

};

Arguments::Arguments()
{

}

void Arguments::setAlgorithm( string _alg)
{
	algorithm = _alg;
}

string Arguments::getAlgorithm()
{
	return algorithm;
}

void Arguments::setHistType( string _histType )
{
	histType = _histType;
}

string Arguments::getHistType()
{
	return histType;
}

void Arguments::setInfo( string _info )
{
	info = _info;
}

string Arguments::getInfo()
{
	return info;
}

void Arguments::setLogPath( string _logPath )
{
	log_path = _logPath;
}

string Arguments::getLogPath()
{
	return log_path;
}

void Arguments::setTestCsvPath( string _testPath )
{
	test_csv_path = _testPath;
}

string Arguments::getTestCsvPath()
{
	return test_csv_path;
}

void Arguments::setTrainCsvPath( string _trainPath )
{
	train_csv_path = _trainPath;
}

string Arguments::getTrainCsvPath()
{
	return train_csv_path;
}

void Arguments::setModelPath( string _modelPath )
{
	model_path = _modelPath;
}

string Arguments::getModelPath()
{
	return model_path;
}

int Arguments::load( string fullPath )
{
	ifstream args(fullPath);

	if (!args.is_open())
	{
		cout << "Unable to open file";
		return EXIT_FAILURE;
	}

	getline(args,this->algorithm);
	getline(args,this->histType);
	getline(args,this->train_csv_path);
	getline(args,this->test_csv_path);

	return EXIT_SUCCESS;
}


int Arguments::save( string fullPath )
{
	ofstream args(fullPath);

	if (!args.is_open())
	{
		cout << "Unable to open file";
		return EXIT_FAILURE;
	}

	args << this->algorithm<<endl;
	args << this->histType<<endl;
	args << this->train_csv_path<<endl;
	args << this->test_csv_path<<endl;

	args.close();

	return EXIT_SUCCESS;
}


FaceRecognizerSVM::FaceRecognizerSVM()
{
	isPerformPCA = false;

	CvSVMParams param = CvSVMParams () ;
	param.svm_type = CvSVM :: C_SVC ;
	param.kernel_type = CvSVM :: RBF ; //CvSVM::RBF, CvSVM::LINEAR ...
	param.degree = 0; // for poly
	param.gamma = 20; // for poly/rbf/sigmoid
	param.coef0 = 0; // for poly/sigmoid
	param.C = 7; // for CV_SVM_C_SVC , CV_SVM_EPS_SVR and CV_SVM_NU_SVR
	param.nu = 0.0; // for CV_SVM_NU_SVC , CV_SVM_ONE_CLASS , and CV_SVM_NU_SVR
	param.p = 0.0; // for CV_SVM_EPS_SVR
	param.class_weights = NULL ; // for CV_SVM_C_SVC
	param.term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS ;
	param.term_crit.max_iter = 1000;
	param.term_crit.epsilon = 1e-6;
}


FaceRecognizerSVM::FaceRecognizerSVM( bool _isPerformPCA )
{
	isPerformPCA = _isPerformPCA;

	CvSVMParams param = CvSVMParams () ;
	param.svm_type = CvSVM :: C_SVC ;
	param.kernel_type = CvSVM :: RBF ; //CvSVM::RBF, CvSVM::LINEAR ...
	param.degree = 0; // for poly
	param.gamma = 20; // for poly/rbf/sigmoid
	param.coef0 = 0; // for poly/sigmoid
	param.C = 7; // for CV_SVM_C_SVC , CV_SVM_EPS_SVR and CV_SVM_NU_SVR
	param.nu = 0.0; // for CV_SVM_NU_SVC , CV_SVM_ONE_CLASS , and CV_SVM_NU_SVR
	param.p = 0.0; // for CV_SVM_EPS_SVR
	param.class_weights = NULL ; // for CV_SVM_C_SVC
	param.term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS ;
	param.term_crit.max_iter = 1000;
	param.term_crit.epsilon = 1e-6;
}


int FaceRecognizerSVM::train( Mat& trainingData, Mat& trainingClasses, CvSVMParams params )
{

	if( svm.train_auto( trainingData, trainingClasses, Mat(), Mat(), params, 5) )
	{
		return EXIT_SUCCESS;
	}
	else
	{
		return EXIT_FAILURE;
	}


}


float FaceRecognizerSVM::predict ( Mat sample )
{

	return svm.predict (sample);
}


typedef struct 
{
	 Mat* mat; 
		 
	 Mat* labels;

	 bool performHistNorm;

} MyStruct;


void asRowMatrix(const vector<Mat> &data, Mat* dst)
{

    for(unsigned int i = 0; i < data.size(); i++)
    {
        Mat image_row = data[i].clone().reshape(1,1);
        Mat row_i = dst->row(i);
        image_row.convertTo(row_i, CV_32FC1, 1/255.);
    }

}

void getLabels(const vector<int> &data, Mat* labels)
{             
	   
	for(int i = 0; i <data.size() ; i++)              
	{                                                                     		   
		labels->at<float>(i, 0) = (float) data[i];                       	   
	}            
	 
}

static void prepareData(const string& csv_file, string dataPath, string modelPath, 
						string testTrainPath, char separator)
{
	   
	std::ifstream file(csv_file.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(1, error_message);
    }
    string line, path, classlabel;


	// csv path inden csv dosya ismini cekmek icin
	vector<string> csvFileNameParts;
	Utils::split(csv_file, "/", csvFileNameParts);


	string fileName;

	vector<string> parts;


	mkdir(dataPath.c_str());
	mkdir(modelPath.c_str());
	mkdir(testTrainPath.c_str());

	string newImgPath;

	std::ofstream out;
	out.open( modelPath + "/" + csvFileNameParts[csvFileNameParts.size()-1] );


    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) 
		{		
			Utils::split(path, "\\", parts);

			fileName = parts[parts.size()-1];

			vector<string> partsFileName;

			Utils::split(fileName, ".", partsFileName);

			newImgPath = testTrainPath + "/" + partsFileName[partsFileName.size()-2] + "_" + classlabel + "." + partsFileName[partsFileName.size()-1];

			// yeni csv dosyasina yeni path ler yaziliyor
			out << newImgPath << ";" << classlabel << endl;

			Mat img = imread(path);

			imwrite(newImgPath, img);
        }
    }

	   
   if (out.is_open())
	   out.close();
 

}


static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator) {
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
			images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}


int Csv2Mat( string csvPath, MyStruct* param )
{

	vector<Mat> images;
	vector<int> lbls;

	Mat* dataMat;
	Mat* dataLabels;

	try
	{
		//Utils::read_csv( csvPath, images, lbls );
		read_csv(csvPath, images, lbls, ';');

		if( param->performHistNorm )
		{
			Mat img, preprocessed, normalized;

			cout << "performing illumination normalization..." << endl; 

			for(int i=0; i<images.size(); i++)
			{
				img = images[i];
				preprocessed = Utils::tan_triggs_preprocessing(img);
				normalized = Utils::norm_0_255(preprocessed);

				images.erase(images.begin()+i);

				images.insert(images.begin()+i, normalized);
			}

			cout << "illumination normalization finished." << endl; 

		}


	} catch(cv::Exception& e)
	{
		//cout << " Error: " << endl;
		cerr << "Error opening file \"" << csvPath << "\". Reason: " << e.msg << endl;

		return EXIT_FAILURE;
	}

	int imgSize = images[0].rows * images[0].cols;

	dataMat = new Mat( images.size(), imgSize, CV_32FC1 );

	
	// get the data Matrix
	Mat* dst = new Mat(static_cast<int>(images.size()), imgSize, CV_32FC1);        // m,r*c büyüklüðünde data hazýrlanacak ---> m training data count
	
	asRowMatrix(images, dst);

	param->mat = dst;


	// get Labels
	Mat* labels = new Mat(lbls.size(), 1, CV_32FC1);

	getLabels(lbls, labels);

	param->labels = labels;


	return EXIT_SUCCESS;
}


// accuracy
float evaluate ( Mat& predicted , Mat& actual ) {
	assert ( predicted.rows == actual.rows ) ;
	int t = 0;
	int f = 0;
	for(int i = 0; i < actual.rows ; i ++) {
		float p = predicted.at <float >(i ,0) ;
		float a = actual.at <float >(i ,0) ;
		if( (int)p == (int)a ) {
			t ++;
		} else {
			f ++;
		}
	}
	return (t * 1.0) / (t + f );
}



// perform PCA to the data matrix
static PCA performPCA( Mat& src, Mat& dst, int numComponents )
{

	PCA pca(src, Mat(), CV_PCA_DATA_AS_ROW, numComponents);
 

	//    
	//// copy the PCA results
	//mean = pca.mean.reshape(1,1); // store the mean vector
	//Mat _eigenvectors;
	//transpose(pca.eigenvectors, _eigenvectors); // eigenvectors by column


	for(int i = 0; i < src.rows; i++) {

		pca.project( src.row(i), dst.row(i) );

	}

	//asRowMatrix( _projections, &dst);

	
	return pca;
}

static const string currentDateTime()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);

    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}


/*
* usage
*
* Display a standard usage parameters or help if there is a problem with the
* command line.
*/
void usage(const char* name){
	printf("Usage: %s [OPTIONS] \n",name);
	printf("  ALGORITHM OPTIONS:  (You must specify at least one algorithm)\n");
	printf("    -algorithm <ALG_NAME>   - algorithm name [EigenFaces|FisherFaces|LBPH] \n");
	printf("  NORMALIZATION OPTIONS:\n");
	printf("    -hist <TYPE>            - Select the type of histogram equalization.\n");
	printf("                              Default is TanTriggs, Options are Histeq.. \n");
	printf("                              Histeq		- equalize raw 0-255 pixels, like original NIST code.\n");
	printf("                              Tan_Triggs	- Tan&Triggs illumination normalization.\n");
	printf("  INPUT OPTIONS:  (You must specify at least one input csv path)\n");
	printf("    -train_csv_path <DIR_NAME>   - full path of the training csv file \n");
	printf("    -test_csv_path  <DIR_NAME>   - full path of the test csv file \n");
	printf("  OUTPUT OPTIONS:  (You must specify at least one output log path)\n");
	printf("    -log_path <DIR_NAME>   - full path of the log file \n");
	printf("  OTHER OPTIONS:\n");
	printf("    -info           - any information about the current setup.\n");
	exit(1);
}


void processCommand(int argc, char** argv, Arguments* args) {
	int i;
	int param_num = 0;

	/******* Set up default values *******/
	args->setAlgorithm("PCA_SVM");
	args->setHistType("Tan_Triggs");
	args->setInfo("");
	args->setLogPath("");
	args->setTestCsvPath("");
	args->setTrainCsvPath("");

	char* temp;
	/******* Read command line arguments *******/

	for (i = 1;i < argc;i++) {

		/* Catch common help requests */
		if      (Utils::readOption(argc, argv, &i, "-help" )) { usage(argv[0]); }
		else if (Utils::readOption(argc, argv, &i, "--help")) { usage(argv[0]); }

		/******* Read in normalization options *******/
		else if (Utils::readOptionString(argc, argv, &i, "-hist", &(temp))) { args->setHistType(string(temp)); }

		/******* Read in training csv file path *******/
		else if (Utils::readOptionString(argc, argv, &i, "-train_csv_path", &(temp))) { args->setTrainCsvPath(string(temp)); }

		/******* Read in test csv file path *******/
		else if (Utils::readOptionString(argc, argv, &i, "-test_csv_path", &(temp))) { args->setTestCsvPath(string(temp)); }

		/************** Read log path ***************/
		else if (Utils::readOptionString(argc, argv, &i, "-log_path", &(temp))) { args->setLogPath(string(temp)); }

		/************** Read info ***************/
		else if (Utils::readOptionString(argc, argv, &i, "-info", &(temp))) { args->setInfo(string(temp)); }

		/* check if the current argument is an unparsed option */
		else if (Utils::checkBadOption(argc,argv,&i)) {}

		else{ Utils::clParseError(argc,argv,i,"Wrong number of required arguments"); }

	}

}


int main(int argc, char **argv) {


	Arguments* args = new Arguments();
	processCommand(argc, argv, args);


	string logName = args->getLogPath();

	if( stricmp(logName.c_str(),"")==0 | stricmp(logName.c_str()," ")==0 )
	{
		srand(time(NULL));
		int id = rand() % 1000000;
		logName = string( "log_" + to_string(id) + ".log");
	}

	
	Logger myLog( logName );


	string train_csv_path = args->getTrainCsvPath();

	string test_csv_path = args->getTestCsvPath();


	///*****************************************/
	///*************prepare data***************/
	//	
	//string dataPath = "../data";

	//string modelPath_CMU = "../data/CMU" ;
	//string trainPath_CMU = "../data/CMU/train" ;
	//string testPath_CMU = "../data/CMU/test" ;
	//string trainCSVPath_CMU = "../5-fold/CMU_PIE/train1.txt";
	//string testCSVPath_CMU = "../5-fold/CMU_PIE/test1.txt";	

	//string modelPath_YALE = "../data/YALE" ;
	//string trainPath_YALE = "../data/YALE/train" ;
	//string testPath_YALE = "../data/YALE/test" ;
	//string trainCSVPath_YALE = "../5-fold/yaleB/train1.txt";
	//string testCSVPath_YALE = "../5-fold/yaleB/test1.txt";
	//	 
	//string modelPath_FERET = "../data/FERET" ;
	//string trainPath_FERET = "../data/FERET/train" ;
	//string testPath_FERET = "../data/FERET/test" ; 
	//string trainCSVPath_FERET = "../5-fold/Feret_gray_FaFbBaBjBk/train1.txt";
	//string testCSVPath_FERET = "../5-fold/Feret_gray_FaFbBaBjBk/test1.txt";

	//prepareData(trainCSVPath_CMU, dataPath, modelPath_CMU, trainPath_CMU, ';');
	//prepareData(testCSVPath_CMU, dataPath, modelPath_CMU, testPath_CMU, ';');

	//prepareData(trainCSVPath_YALE, dataPath, modelPath_YALE, trainPath_YALE, ';');
	//prepareData(testCSVPath_YALE, dataPath, modelPath_YALE, testPath_YALE, ';');

	//prepareData(trainCSVPath_FERET, dataPath, modelPath_FERET, trainPath_FERET, ';');
	//prepareData(testCSVPath_FERET, dataPath, modelPath_FERET, testPath_FERET, ';');

	///***************************************/


	if ( stricmp(train_csv_path.c_str(),"")==0 | stricmp(test_csv_path.c_str(),"")==0
		| stricmp(train_csv_path.c_str()," ")==0 | stricmp(test_csv_path.c_str()," ")==0 )
	{
				
		cout << " Some Arguments are missing.. Please check arguments. Train and test csv paths should be given." << endl ;
		myLog << myLog.currentDateTime() << "\t" << "Some Arguments are missing.. Please check arguments. Train and test csv paths should be given. " << endl;
				
		system("pause");
		return EXIT_FAILURE;
	}


	// apply PCA + SVM
	FaceRecognizerSVM mySVM(true);


	MyStruct* train_param = new MyStruct;
	MyStruct* test_param = new MyStruct;

		
	if( stricmp( (args->getHistType()).c_str(), "Tan_Triggs" ) == 0 )
	{
		train_param->performHistNorm = true;
		test_param->performHistNorm = true;
	}
	else
	{
		train_param->performHistNorm = false;
		test_param->performHistNorm = false;
	}


	cout << myLog.currentDateTime() << " Loading Training data Matrix..." << endl ;
	myLog << myLog.currentDateTime() << " Loading Training data Matrix..." << endl ;

	if ( Csv2Mat(train_csv_path, train_param) == EXIT_SUCCESS )
	{
		cout << myLog.currentDateTime() << "Training data Matrix is Ready! " << endl;
		myLog << myLog.currentDateTime() << "Training data Matrix is Ready! " << endl;
	}
	else
	{		
		cout << myLog.currentDateTime() << " Training data Matrix could not be loaded! " << endl ;
		myLog << myLog.currentDateTime() << " Training data Matrix could not be loaded! " << endl ;
		waitKey(0);
		return EXIT_FAILURE;
	}
	

	cout << myLog.currentDateTime() << " Loading Test data Matrix..." << endl ;
	myLog << myLog.currentDateTime() << " Loading Test data Matrix..." << endl ;

	if ( Csv2Mat(test_csv_path, test_param) == EXIT_SUCCESS )
	{
		cout << myLog.currentDateTime() << " Test data Matrix is Ready! " << endl;	
		myLog << myLog.currentDateTime() << " Test data Matrix is Ready! " << endl;	
	}
	else
	{
		cout << myLog.currentDateTime() << " Test data Matrix could not be loaded! " << endl;
		myLog << myLog.currentDateTime() << " Test data Matrix could not be loaded! " << endl;
		waitKey(0);
		return EXIT_FAILURE;
	}


	time_t start;
	time_t finish;

	double seconds, minutes, hours;


	int numComponents = 50; //train_param->mat->rows-1;
	Mat train_pca(train_param->mat->rows, numComponents, train_param->mat->type()); 		
	Mat test_pca(test_param->mat->rows, numComponents, test_param->mat->type()); 

	if (mySVM.isPerformPCA)
	{

		cout << myLog.currentDateTime() << " Performing PCA to the Training Data!" << endl ;
		myLog << myLog.currentDateTime() << " Performing PCA to the Training Data!" << endl ;

		time(&start);

		PCA pca = performPCA( *(train_param->mat), train_pca, numComponents );

		time(&finish);

		seconds = difftime(finish, start);
	    minutes = seconds/60;
	    hours = seconds/3600;

		cout << myLog.currentDateTime() << " PCA to Train data took " << seconds << " seconds, " << minutes << " minutes, " << hours << "hours." << endl;
		myLog << myLog.currentDateTime() << " PCA to Train data took " << seconds << " seconds, " << minutes << " minutes, " << hours << "hours." << endl;

		train_param->mat = &train_pca;


		cout << myLog.currentDateTime() << " Performing PCA to the Test Data!" << endl ;
		myLog << myLog.currentDateTime() << " Performing PCA to the Test Data!" << endl ;

		time(&start);

		for (int i = 0; i < test_param->mat->rows; i++)
		{
			pca.project( test_param->mat->row(i), test_pca.row(i));
		}
		
		
		time(&finish);

		seconds = difftime(finish, start);
	    minutes = seconds/60;
	    hours = seconds/3600;

		cout << myLog.currentDateTime() << " PCA to Test data took " << seconds << " seconds, " << minutes << " minutes, " << hours << "hours." << endl;
		myLog << myLog.currentDateTime() << " PCA to Test data took " << seconds << " seconds, " << minutes << " minutes, " << hours << "hours." << endl;

		test_param->mat = &test_pca;

	}



	cout << myLog.currentDateTime() << " Training SVM..." << endl ;
	myLog << myLog.currentDateTime() << " Training SVM..." << endl ;

	time(&start);


	if( true )
	{

		try{

			int s = mySVM.train( *(train_param->mat), *(train_param->labels), mySVM.param );
			 
		} catch( cv::Exception e )
		{
			cout << "Error: " << e.msg << endl;
		}

		time(&finish);

		seconds = difftime(finish, start);
	    minutes = seconds/60;
	    hours = seconds/3600;

		cout << myLog.currentDateTime() << " SVM Training took " << seconds << " seconds, " << minutes << " minutes, " << hours << "hours." << endl;
		myLog << myLog.currentDateTime() << " SVM Training took " << seconds << " seconds, " << minutes << " minutes, " << hours << "hours." << endl;
	}
	else
	{
		cout << myLog.currentDateTime() << " Error while Training SVM.."  << endl ;
		myLog << myLog.currentDateTime() << " Error while Training SVM.."  << endl ;
		waitKey(0);
		return EXIT_FAILURE;
	}


	// TEST
	Mat predicted ( test_param->labels->rows , 1, CV_32F );
	int num_test_sample = test_param->mat->rows;

	cout << myLog.currentDateTime() << " Predicting SVM..." << endl ;
	myLog << myLog.currentDateTime() << " Predicting SVM..." << endl ;
	time(&start);


	for ( int i = 0; i < num_test_sample; i++ ) {

		Mat sample = test_param->mat->row(i);

		cout << myLog.currentDateTime() << " Predicting sample " << i << "/" << num_test_sample << endl ;

		predicted.at<float>(i, 0) = mySVM.predict( sample );

		myLog << "actual: " << test_param->labels->at<float>(i, 0) << "  predicted: " << predicted.at<float>(i, 0) << endl ;
		
	}

	time(&finish);
			
	seconds = difftime(finish, start); 
	minutes = seconds/60;
	hours = seconds/3600;

	float accuracy = evaluate( predicted, *(test_param->labels) );
			
	cout << myLog.currentDateTime() << " SVM Prediction took " << seconds << " seconds, " << minutes << " minutes, " << hours << "hours." << endl;		
	myLog << myLog.currentDateTime() << " SVM Prediction took " << seconds << " seconds, " << minutes << " minutes, " << hours << "hours." << endl;		
	cout <<  myLog.currentDateTime() << " Accuracy_ {SVM} = " << accuracy << endl ;
	myLog <<  myLog.currentDateTime() << " Accuracy_ {SVM} = " << accuracy << endl ;
	
	waitKey(0);


	return EXIT_SUCCESS;
}