/*
 * Passo6.h
 *
 *  Created on: 10 de abr. de 2025
 *      Author: luis
 */
#include "../COMMON/PassoMaster.h"
#include <chrono>
#include <thread>
#include <random>
#include <iostream>

#ifndef CLASS_PASSO6
#define CLASS_PASSO6

class Passo6 :  public PassoMaster {
public:
	using PassoMaster::PassoMaster;
	virtual ~Passo6();
	void exec();
};


#endif
