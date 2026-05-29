/*
 * Passo6.cpp
 *
 *  Created on: 10 de abr. de 2025
 *      Author: luis
 */

#include "Passo6.h"

Passo6::~Passo6() {
	// TODO Auto-generated destructor stub
}

void Passo6::exec() {
	std::string novoNomeArquivo = "";
	setFiltroRecupera("M__");
	setFiltroPersiste("MG_");
	while(true){
		std::string nomeArquivoProcessado =	this->getFirstFilterFile(true);
		while (nomeArquivoProcessado.size()==0){
			std::cout << "Passo 4: Aguardando arquivo com extensão " << getFiltroRecupera() << "..0 \n";
			//Se não encontrou nenhum arquivo na pasta
			//aguarda um tempo aleatorio entre 1 e 10 segundos.
			int tempoEspera = rand() % 100 + 1;
			std::this_thread::sleep_for(std::chrono::milliseconds(tempoEspera*100));
			//Procura novo arquivo na pasta
			nomeArquivoProcessado =	this->getFirstFilterFile(true);
		}
		//recupera individuo
		std::string nomeArquivoProcessadoF =	getPathArquivo()+"/"+nomeArquivoProcessado;
		Individuo individuo(nomeArquivoProcessadoF);
		if (individuo.getNumeroDrones()){
			std::string nomeTemporario = "temp__"+nomeArquivoProcessado;
			std::string nomeTemporarioCompleto = getPathArquivo() +"/"+ nomeTemporario;
			std::string nomeCompleto  = getPathArquivo()+"/"+nomeArquivoProcessado;
			if (std::rename(nomeCompleto.c_str(),nomeTemporarioCompleto.c_str()))
				return;

			//repara individuo
			try{
				individuo.reparaIndividuo();
			}
			catch(...){
				if (std::rename(nomeTemporarioCompleto.c_str(),nomeCompleto.c_str()))
								return;
			}
			//persiste individuo
			std::cout << geraNomePersistencia(nomeArquivoProcessado) << "\n";
			individuo.persisteIndividuo(geraNomePersistencia(nomeArquivoProcessado),getPathArquivo());
			removeFile(nomeTemporario,getPathArquivo());
		}
	}

}
