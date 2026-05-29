#include "Passo1.h"

#define numInicialDrones 5

void Passo1::exec(){
}

Passo1::~Passo1(){
}

Passo1::Passo1(std::string caminho):PassoMaster(caminho){
}

//int main(int argc, char *argv[]){
//	Passo1 passo1(".");
//	std::vector<Individuo> individuos;
//	int tamanhoPopulacao;
//	int tamanhoGrid;
//	if(argc == 3){
//		tamanhoPopulacao = std::atoi(argv[1]);
//		tamanhoGrid = std::atoi(argv[2]);
//	}
//	else{
//		tamanhoPopulacao = 500;
//		tamanhoGrid = 100;
//	}
//	//cria populaçao
//	int somaSaltos = 0;
//	for(int i = 0; i < tamanhoPopulacao; i ++){
//		Individuo individuo(tamanhoGrid);
//		individuo.geraIndividuo(numInicialDrones);
//		individuo.reparaIndividuo();
//		individuos.push_back(individuo);
//		somaSaltos += individuo.getSaltos();
//		printf("%s\n", passo1.getFileName("I__", i+1, std::to_string(tamanhoPopulacao).length()).c_str());
//	}
//	//avalia individuos
//	for(Individuo individuo: individuos){
//		individuo.setFitness(1.0 / (individuo.getNumeroDrones() + (double)individuo.getSaltos() / (double)somaSaltos));
//	}
//	//persiste individuos
//	for(int i = 0; i < tamanhoPopulacao; i ++){
//		individuos.at(i).persisteIndividuo(passo1.getFileName("I__", i+1, std::to_string(tamanhoPopulacao).length()));
//	}
//	return 0;
//}
