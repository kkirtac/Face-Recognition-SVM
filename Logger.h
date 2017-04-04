#ifndef LOGGER_H   
#define LOGGER_H


#include <iostream>
#include <fstream>
#include <ostream>
#include <string>


class Logger
{
   private:
	std::ofstream log;
	
   public:
	Logger();
	Logger(std::string dosya_adi);
	~Logger();

	const std::string currentDateTime();
	void writeln(std::string yazi);	

	std::ostream& operator<< (std::string str) 
	{ 
		log<<str;
		return log;
	}

};



#endif