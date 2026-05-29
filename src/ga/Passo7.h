/*
 * Passo7.h
 *
 *  Created on: 10 de abr. de 2025
 *      Author: luis
 */
#include "../COMMON/PassoMaster.h"
#include <chrono>
#include <thread>
#include <random>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <exception>


#ifndef CLASS_PASSO7
#define CLASS_PASSO7

class Passo7 :  public PassoMaster {
private:
	 std::vector<Individuo> individuosIniciais;
	 std::vector<Individuo> individuosFinais;
	 int somaSaltos = 0;
	 double somaFitness = 0.0;
	 long int cicloUltimaTrocaMaiorFitness = this->getMaxCiclos();
public:
	using PassoMaster::PassoMaster;
	virtual ~Passo7();
	void exec();
	void calculaFintess();
	void elitizarIndividuos();
	void recuperaIndividuos(std::string filtro, std::vector<Individuo> &individuos);
	void persisteCicloUltimaTrocaMaiorFitness(std::string pathArquivo);
	void recuperaCicloUltimaTrocaMaiorFitness(std::string pathArquivo);
};


#endif
