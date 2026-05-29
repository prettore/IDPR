/*
 * Passo4.h
 *
 *  Created on: 10 de abr. de 2025
 *      Author: luis
 */
#include "../COMMON/PassoMaster.h"
#include "../COMMON/individuo.h"
#include <chrono>
#include <thread>
#include <random>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>


#ifndef CLASS_PASSO4
#define CLASS_PASSO4

class Passo4 :  public PassoMaster {
public:
	using PassoMaster::PassoMaster;
	virtual ~Passo4();
	void exec();
};


#endif
