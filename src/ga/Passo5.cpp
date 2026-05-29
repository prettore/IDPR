/*
 * Passo5.cpp
 *
 *  Created on: 10 de abr. de 2025
 *      Author: luis
 */

#include "Passo5.h"

#define MAX 30

Passo5::~Passo5() {
	// TODO Auto-generated destructor stub
}

void Passo5::mutacao1(Individuo &individuo){
	//remove drone
	int tamanhoGrid = individuo.getCenario().size();
	double minimoOrigem = tamanhoGrid*tamanhoGrid;
	double minimoDestino = tamanhoGrid*tamanhoGrid;
	int menor = 1;
	int menorOrigem = 0;
	int menorDestino = individuo.getNumeroDrones() - 1;
	bool conecta = false;
	for(Drone drone: individuo.getDrones()){
		if(drone.getId() == 0 || drone.getId() == individuo.getNumeroDrones() - 1){
			continue;
		}
		int x1 = drone.getX();
		int y1 = drone.getY();
		for(int i = 0; i < individuo.getNumeroDrones(); i ++){
			int x2 = individuo.getDrone(i).getX();
			int y2 = individuo.getDrone(i).getY();
			if(individuo.getRota(i, drone.getId())){
				if(sqrt(pow(x1-x2, 2) + pow(y1-y2, 2)) < minimoOrigem){
					minimoOrigem = sqrt(pow(x1-x2, 2) + pow(y1-y2, 2));
					menorOrigem = i;
					if(minimoOrigem <= minimoDestino){
						menor = drone.getId();
					}
				}
			}
			else if(individuo.getRota(drone.getId(), i)){
				if(sqrt(pow(x1-x2, 2) + pow(y1-y2, 2)) < minimoDestino){
					minimoDestino = sqrt(pow(x1-x2, 2) + pow(y1-y2, 2));
					menorDestino = i;
					if(minimoDestino < minimoOrigem){
						menor = drone.getId();
					}
				}
			}
		}
	}
	if(minimoDestino < minimoOrigem){
		if(!individuo.existeCaminho(0, menorDestino)){
			//encontra origem
			menorOrigem = 0;
			minimoOrigem = tamanhoGrid*tamanhoGrid;
			Drone destino = individuo.getDrone(menorDestino);
			int x1 = destino.getX();
			int y1 = destino.getY();
			for(int i=0; i<individuo.getNumeroDrones()-1; i++){
				if(individuo.getRota(i, menor)){
					int x2 = individuo.getDrone(i).getX();
					int y2 = individuo.getDrone(i).getY();
					if(sqrt(pow(x1-x2, 2) + pow(y1-y2, 2)) < minimoOrigem){
						minimoOrigem = sqrt(pow(x1-x2, 2) + pow(y1-y2, 2));
						menorOrigem = i;
					}
				}
			}
			conecta = true;
		}
	}
	else{
		individuo.setRota(menorOrigem, menor, false);
		if(!individuo.existeCaminho(menorOrigem, individuo.getNumeroDrones()-1)){
			//encontra destino
			Drone origem = individuo.getDrone(menorOrigem);
			menorDestino = individuo.getNumeroDrones() - 1;
			minimoDestino = tamanhoGrid*tamanhoGrid;
			int x1 = origem.getX();
			int y1 = origem.getY();
			for(int i=1; i<individuo.getNumeroDrones(); i++){
				if(individuo.getRota(menor,i)){
					int x2 = individuo.getDrone(i).getX();
					int y2 = individuo.getDrone(i).getY();
					if(sqrt(pow(x1-x2, 2) + pow(y1-y2, 2)) < minimoDestino){
						minimoDestino = sqrt(pow(x1-x2, 2) + pow(y1-y2, 2));
						menorDestino = i;
					}
				}
			}
			conecta = true;
		}
	}
	if(conecta){
		individuo.setRota(menorOrigem, menorDestino, true);
	}
	individuo.removeDrone(menor);
	if(!individuo.existeCaminho(0, individuo.getNumeroDrones() - 1)){
		individuo.criaCaminho(0);
	}
}

void Passo5::mutacao2(Individuo &individuo){
	//desloca drone
	std::random_device rd; 
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> disu(1, individuo.getNumeroDrones()-2);
    std::normal_distribution<> disn(0.0, 2.5);
	int tamanhoGrid = individuo.getCenario().size();
	int drone = disu(gen);
	int x_ini = individuo.getDrones().at(drone).getX();
	int y_ini = individuo.getDrones().at(drone).getY();
	int x = x_ini;
	int y = y_ini;
	while(x >= tamanhoGrid || y >= tamanhoGrid || x < 0 || y < 0 || individuo.getCenario(y, x)){
		x = x_ini + disn(gen);
		y = y_ini + disn(gen);
	}
	individuo.setCenario(y_ini, x_ini, false);
	individuo.setCenario(y, x, true);
	individuo.setDrone(drone, x, y);
}

void Passo5::exec() {
	std::random_device rd; 
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.0, 0.99999);
	std::string novoNomeArquivo = "";
	setFiltroRecupera("IG_");
	setFiltroPersiste("M__");
	bool processouPrimeiroArquivo = false;
	long  tempoEsperaPorArquivo = 0;
	while(true){
		std::string nomeArquivoProcessado =	this->getFirstFilterFile(true);
		while (nomeArquivoProcessado.size()==0){
			std::cout << "Passo 5: Aguardando arquivo com extensão " << getFiltroRecupera() << "..0 \n";
			//Se não encontrou nenhum arquivo na pasta
			//aguarda um tempo aleatorio entre 1 e 10 segundos.
			int tempoEspera = rand() % 100 + 1;
			std::this_thread::sleep_for(std::chrono::milliseconds(tempoEspera*100));
			//Procura novo arquivo na pasta
			nomeArquivoProcessado =	this->getFirstFilterFile(true);
		}
		Individuo individuo(getPathArquivo()+"/"+nomeArquivoProcessado);
		if (individuo.getNumeroDrones()){
			std::string nomeTemporario = "temp__"+nomeArquivoProcessado;
			std::string nomeTemporarioCompleto = getPathArquivo() +"/"+ nomeTemporario;
			std::string nomeCompleto  = getPathArquivo()+"/"+nomeArquivoProcessado;
			if (std::rename(nomeCompleto.c_str(),nomeTemporarioCompleto.c_str()))
				return;

			try{
				//mutacao1: remove drone
				if(dis(gen) <= 0.15){
					mutacao1(individuo);
				}
				//mutacao2: desloca drone
				if(dis(gen) <= 0.30){
					mutacao2(individuo);
				}
			}
			catch(...){
				if (std::rename(nomeTemporarioCompleto.c_str(),nomeCompleto.c_str()))
								return;
			}

			individuo.persisteIndividuo(geraNomePersistencia(nomeArquivoProcessado),getPathArquivo());
			processouPrimeiroArquivo = true;
			removeFile(nomeTemporario,getPathArquivo());
		}
	}
}
