#include <iostream>
#include <string>
#include <stdio.h>
#include <time.h>


#include "Logger.h"

using namespace std;


const string Logger::currentDateTime()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);

    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}


Logger::Logger(string dosya)
{
  log.open(dosya);

  if (log.is_open())
  {
	log<<currentDateTime()<<"\tDosya olusturuldu......"<<endl;
  }
  else 
	cout << "Unable to open file";
}


Logger::Logger()
{
   log.open("log.txt");

   if (log.is_open())
   {
	log<<currentDateTime()<<"\tDosya olusturuldu......"<<endl;
   }
   else 
	cout << "Unable to open file";
}

Logger::~Logger()
{
   if (log.is_open())
   {
 	log<<currentDateTime()<<"\tDosya  kapatildi......"<<endl;
   }

   log.close();
}


void Logger::writeln(string yazi)
{
   if (log.is_open())
   {
	log<<yazi<<endl;
   }
   else 
	cout << currentDateTime()<<"\tUnable to writeln to file";
}









