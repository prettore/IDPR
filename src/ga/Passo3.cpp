/*
 * Passo3.cpp
 *
 *  Created on: 10 de abr. de 2025
 *      Author: luis
 */

#include "Passo3.h"
#include "../COMMON/individuo.h"

Passo3::~Passo3() {
	// TODO Auto-generated destructor stub
}

void Passo3::exec() {

	Individuo *individuoMerge1;
	Individuo *individuoMerge2;

	std::string novoNomeArquivo1 = "";
	std::string novoNomeArquivo2 = "";
	setFiltroRecupera("I__");
	setFiltroPersiste("IC_");

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);
	std::uniform_int_distribution<> distribution(1,500);


	std::string numeroArquivoA;
	std::string numeroArquivoB;
	std::string numeroArquivoC;
	std::string numeroArquivoD;


	try{
		if ((std::stoi(getNumeroContador1()) > 1001) || (std::stoi(getNumeroContador2()) > 1001)){
			int tempoEspera = rand() % 10 + 1;
			std::this_thread::sleep_for(std::chrono::milliseconds(tempoEspera*100));
			return;
		}
	}
	catch(...){
		std::cout <<"Exit " << std::stoi(getNumeroContador1()) << " " << std::stoi(getNumeroContador2()) << "\n";
		exit(-1);
	}



		novoNomeArquivo1 = getFiltroPersiste()+getNumeroContador1();
		novoNomeArquivo2 = getFiltroPersiste()+getNumeroContador2();


		numeroArquivoA = std::to_string(distribution(generator));
		numeroArquivoB = std::to_string(distribution(generator));
		numeroArquivoC = std::to_string(distribution(generator));
		numeroArquivoD = std::to_string(distribution(generator));

	if (numeroArquivoA.size()<3){
			std::string tempString(3-numeroArquivoA.size(),'0');
			std::cout << tempString << "\n";
			numeroArquivoA = tempString+numeroArquivoA;
	}
	if (numeroArquivoB.size()<3){
			std::string tempString(3-numeroArquivoB.size(),'0');
			std::cout << tempString << "\n";
			numeroArquivoB = tempString+numeroArquivoB;

	}
	if (numeroArquivoC.size()<3){
			std::string tempString(3-numeroArquivoC.size(),'0');
			std::cout << tempString << "\n";
			numeroArquivoC = tempString+numeroArquivoC;
	}
	if (numeroArquivoD.size()<3){
			std::string tempString(3-numeroArquivoD.size(),'0');
			std::cout << tempString << "\n";
			numeroArquivoD = tempString+numeroArquivoD;
	}


	std::string nomeArquivoProcessadoA =	getPathArquivo()+"/"+ getFiltroRecupera() + numeroArquivoA;
	std::string nomeArquivoProcessadoB =	getPathArquivo()+"/"+ getFiltroRecupera() + numeroArquivoB;
	std::string nomeArquivoProcessadoC =	getPathArquivo()+"/"+ getFiltroRecupera() + numeroArquivoC;
	std::string nomeArquivoProcessadoD =	getPathArquivo()+"/"+ getFiltroRecupera() + numeroArquivoD;

	while ((!arquivoExiste(nomeArquivoProcessadoA) || !arquivoExiste(nomeArquivoProcessadoB))
		|| (!arquivoExiste(nomeArquivoProcessadoC) || !arquivoExiste(nomeArquivoProcessadoD))){
		//Se não encontrou nenhum arquivo na pasta
		//aguarda um tempo aleatorio entre 1 e 10 segundos.
		int tempoEspera = rand() % 100 + 1;
		std::this_thread::sleep_for(std::chrono::milliseconds(tempoEspera*100));
		std::cout << "Aguardando arquivos::" << nomeArquivoProcessadoA << " " << nomeArquivoProcessadoB << " " << nomeArquivoProcessadoC << " " << nomeArquivoProcessadoD << "\n";
	}
	Individuo individuoA(nomeArquivoProcessadoA);
	Individuo individuoB(nomeArquivoProcessadoB);
	Individuo individuoC(nomeArquivoProcessadoC);
	Individuo individuoD(nomeArquivoProcessadoD);


		if (individuoA.getFitness() > individuoB.getFitness())
			individuoMerge1 = &individuoA;
		else
			individuoMerge1 = &individuoB;

		if (individuoC.getFitness() > individuoD.getFitness())
			individuoMerge2 = &individuoC;
		else
			individuoMerge2 = &individuoD;

		std::vector<Individuo> resultadosCruzamento;
		try {
			resultadosCruzamento = cruzamento(individuoMerge1,individuoMerge2);
		}catch (const std::exception &e){
	        std::cerr << "Capturada uma excecao: " << e.what() << std::endl;
	        std::cerr << "Tipo da excecao: " << typeid(e).name() << std::endl;
	        std::cerr << individuoMerge1->getGridSize()<< std::endl;
	        std::cerr << "Nome arquivo A::" << nomeArquivoProcessadoA << std::endl;
	        std::cerr << "Nome arquivo B::" << nomeArquivoProcessadoB << std::endl;
	        exit(-1);
	    } catch (...) {
	        // Catch any other types of exceptions (catch-all)
	        std::cerr << "Caught an unknown exception." << std::endl;
	        exit(-1);
	    }


		if (std::stoi(getNumeroContador1()) <= 1000)
			resultadosCruzamento[0].persisteIndividuo(novoNomeArquivo1,getPathArquivo());
		if (std::stoi(getNumeroContador2()) <= 1000)
			resultadosCruzamento[1].persisteIndividuo(novoNomeArquivo2,getPathArquivo());
		std::cout << "Arquivos gerados e persistidos "<< novoNomeArquivo1 << " "<< novoNomeArquivo2 <<"\n";


}

std::vector<Individuo> Passo3::cruzamento(Individuo *individuoA,Individuo *individuoB) {
	int gridSize = individuoA->getGridSize();
	Individuo individuoCalculado1(gridSize);
	Individuo individuoCalculado2(gridSize);
	bool individuo1Finalizado = false,individuo2Finalizado = false;
	int  limCaminhos1= 10000000, limCaminhos2 = 10000000;


	std::vector<Drone> drones1A = individuoA->getDrones();
	std::vector<Drone> drones1B = individuoB->getDrones();
	std::vector<Drone> drones2A = individuoA->getDrones();
	std::vector<Drone> drones2B = individuoB->getDrones();

	std::vector<Drone> *dronesEscolha1;
	std::vector<Drone> *dronesEscolha2;

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);
	std::uniform_int_distribution<int> distribution(0,1);

	while (!individuo1Finalizado || !individuo2Finalizado){

			//Seleciona o vetor
			if (distribution(generator)){
				dronesEscolha1 = &drones1A;
				if (!(dronesEscolha1->size()-2))dronesEscolha1 = &drones1B;
				dronesEscolha2 = &drones2B;
				if (!(dronesEscolha2->size()-2))dronesEscolha2 = &drones2A;
			}
			else{
				dronesEscolha1 = &drones1B;
				if (!(dronesEscolha1->size()-2))dronesEscolha1 = &drones1A;
				dronesEscolha2 = &drones2A;
				if (!(dronesEscolha2->size()-2))dronesEscolha2 = &drones2B;
			}



			if (!individuo1Finalizado){
				//Garante que não possui só o inicial e final
				std::uniform_int_distribution<int> distribution1(1,dronesEscolha1->size()-1);

				int pos = distribution1(generator);
				while ((pos <1) || (pos>dronesEscolha1->size())){
					pos = distribution(generator);
				}

				Drone escolhido(dronesEscolha1->at(pos));

				dronesEscolha1->erase(dronesEscolha1->begin()+pos);
				individuoCalculado1.addDrone(escolhido.getX(),escolhido.getY());
				individuoCalculado1.criaCaminho(0);
				individuoCalculado1.calculaMetricas();


				if ((double)individuoCalculado1.getConfiabilidade() >= MIN_CONFIABILIDADE){
					individuo1Finalizado = true;
				}



				if (!--limCaminhos1){//Verifica se atingiu o maximo de caminhos permitidos
					individuo1Finalizado = true;
				}

				if ((individuoCalculado1.getSaltos() >= (MAX_SALTOS * 0.8)) && !individuo1Finalizado ){
					for(int i=1;i< (individuoCalculado1.getDrones().size()-1); i++){
						Drone drone = individuoCalculado1.getDrones()[i];
						individuoCalculado1.criaCaminho(drone.getId());
						limCaminhos1--;
						individuoCalculado1.calculaMetricas();
						if ((double)individuoCalculado1.getConfiabilidade() >= MIN_CONFIABILIDADE){//Verifica se atingiu a confiabilidade desejada
							individuo1Finalizado = true;
							break;
						}
						if (limCaminhos1--){//Verifica se atingiu o maximo de caminhos permitidos
							individuo1Finalizado = true;
							break;
						}
						if (individuoCalculado1.getSaltos() >= MAX_SALTOS){//Verifica se atingiu o numero maximo de saltos
							individuo1Finalizado = true;
							break;
						}

					}
				}
			}

			if (!individuo2Finalizado){
				//Garante que não possui só o inicial e final
				std::uniform_int_distribution<int> distribution2(1,dronesEscolha2->size()-1);

				int pos = distribution2(generator);
				while ((pos <1) || (pos>dronesEscolha2->size())){
					pos = distribution(generator);
				}


				Drone escolhido(dronesEscolha2->at(pos));
				dronesEscolha2->erase(dronesEscolha2->begin()+pos);


				individuoCalculado2.addDrone(escolhido.getX(),escolhido.getY());
				individuoCalculado2.criaCaminho(0);
				individuoCalculado2.calculaMetricas();


				if ((double)individuoCalculado2.getConfiabilidade() >= MIN_CONFIABILIDADE){
					individuo2Finalizado = true;
				}



				if (!--limCaminhos2){//Verifica se atingiu o maximo de caminhos permitidos
					individuo2Finalizado = true;
				}

				if ((individuoCalculado2.getSaltos() >= (MAX_SALTOS * 0.8)) && !individuo2Finalizado ){
					for(int i=1;i< (individuoCalculado2.getDrones().size()-1); i++){
						Drone drone = individuoCalculado2.getDrones()[i];
						individuoCalculado2.criaCaminho(drone.getId());
						limCaminhos2--;
						individuoCalculado2.calculaMetricas();
						if ((double)individuoCalculado2.getConfiabilidade() >= MIN_CONFIABILIDADE){//Verifica se atingiu a confiabilidade desejada
							individuo2Finalizado = true;
							break;
						}
						if (limCaminhos2--){//Verifica se atingiu o maximo de caminhos permitidos
							individuo2Finalizado = true;
							break;
						}
						if (individuoCalculado2.getSaltos() >= MAX_SALTOS){//Verifica se atingiu o numero maximo de saltos
							individuo2Finalizado = true;
							break;
						}

					}
				}
			}

	}//while
	std::vector<Individuo> resposta;
	resposta.push_back(individuoCalculado1);
	resposta.push_back(individuoCalculado2);
	return resposta;

}
