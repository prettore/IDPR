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
#include <exception>
#include <typeinfo> // for typeid



#ifndef CLASS_PASSO3
#define CLASS_PASSO3

class Passo3 :  public PassoMaster {
public:
	using PassoMaster::PassoMaster;
	virtual ~Passo3();
	void exec();
	std::vector<Individuo> cruzamento(Individuo *individuoA,Individuo *individuoB);
};


#endif
