/*
 * Passo5.h
 *
 *  Created on: 10 de abr. de 2025
 *      Author: luis
 */
#include "../COMMON/PassoMaster.h"
#include <chrono>
#include <thread>
#include <random>
#include <iostream>

#ifndef CLASS_PASSO5
#define CLASS_PASSO5

class Passo5 :  public PassoMaster {
public:
	using PassoMaster::PassoMaster;
	virtual ~Passo5();
	void exec();
	void mutacao1(Individuo &individuo);
	void mutacao2(Individuo &individuo);
};


#endif
