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
#include <time.h>
#include <random>
#include <chrono>

#ifndef CLASS_PASSO1
#define CLASS_PASSO1

class Passo1 :  public PassoMaster {
public:
	using PassoMaster::PassoMaster;
	Passo1(std::string caminho);
	virtual ~Passo1();
	void exec();
};


#endif
