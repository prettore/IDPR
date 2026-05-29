/*
 * Passo7.cpp
 *
 *  Created on: 10 de abr. de 2025
 *      Author: luis
 */

 #include "Passo7.h"

 Passo7::~Passo7() {
	 // TODO Auto-generated destructor stub
 }
 
 void Passo7::recuperaIndividuos(std::string filtro, std::vector<Individuo> &individuos){
	 std::vector<std::string> listaNomeArquivos;
	 setFiltroRecupera(filtro);
	 listaNomeArquivos = getAllRecuperaFilterFiles();
	 while (listaNomeArquivos.size() < getMaxSortResults()){
		 std::cout << "Passo 7: Aguardando arquivo com o formato: " << getFiltroRecupera() << "..0 \n";
		 std::cout << "Passo 7: aguardando arquivos para processamento. \nTotal de arquivos na pasta: " << listaNomeArquivos.size() <<"\nTotal de arquivos necessários: "<< getMaxSortResults() <<" \n";
		 //Não atingiu o total de arquivos esperados
		 //aguarda um tempo aleatorio entre 1 e 10 segundos.
		 int tempoEspera = rand() % 100 + 1;
		 std::this_thread::sleep_for(std::chrono::milliseconds(tempoEspera*100));
		 //Soma tempo de espera.
		 //Procura novo arquivo na pasta
		 listaNomeArquivos.clear();
		 listaNomeArquivos = getAllRecuperaFilterFiles();
	 }
	 //recupera individuos
//	 individuos.clear();
	 for(std::string arquivo: listaNomeArquivos){
		 Individuo individuo(getPathArquivo()+"/"+arquivo);
		 individuos.push_back(individuo);
	 }
 }
void Passo7::calculaFintess() {
	 //avalia
	 int cicloAtual = getCicloAtual(); //Recupera valor da variavel cicloAtual

	 double penalidade = 0.0; //penaliza arquivos com vazao maior do que a vaza maxima

	 double topConfiabilidade = individuosIniciais[0].getConfiabilidade();
	 double minConfiabilidade = individuosIniciais[0].getConfiabilidade();

	 double topSaltos = individuosIniciais[0].getSaltos();
	 double minSaltos = individuosIniciais[0].getSaltos();

	 int topVazao = individuosIniciais[0].getVazao();
	 int minVazao = individuosIniciais[0].getVazao();

	 int topDrones = individuosIniciais[0].getNumeroDrones();
	 int minDrones = individuosIniciais[0].getNumeroDrones();



	 for(Individuo &individuo: individuosIniciais){
		 somaSaltos += individuo.getNumeroDrones();
		 individuo.setFitness(individuo.getSaltos());

		 if (individuo.getConfiabilidade() > topConfiabilidade)
			 topConfiabilidade = individuo.getConfiabilidade();
		 if (individuo.getConfiabilidade() < minConfiabilidade)
					 minConfiabilidade = individuo.getConfiabilidade();

		 if (individuo.getSaltos() > topSaltos)
			 topSaltos = individuo.getSaltos();
		 if (individuo.getSaltos() < minSaltos)
			 minSaltos = individuo.getSaltos();



		 if (individuo.getVazao() > topVazao)
			 topVazao = individuo.getVazao();
		 if (individuo.getVazao() < minVazao)
			 minVazao = individuo.getVazao();

		 if (individuo.getNumeroDrones() > topDrones)
			 topDrones = individuo.getNumeroDrones();
		 if (individuo.getNumeroDrones() < minDrones)
			 minDrones = individuo.getNumeroDrones();


	 }



	 for(Individuo &individuo: individuosIniciais){
		 penalidade = 0.0;

		 if (individuo.getVazao()> individuo.getMaxVazao()){
			 penalidade = 20.0;
			 penalidade += (double(getMaxCiclos() - cicloAtual)/double(getMaxCiclos()))
					 *(double(individuo.getVazao() - minVazao)/double(topVazao - minVazao))
					 *1000;
		 }

		 individuo.setFitness(1.0 /(
				 	 	 	 	 	 3.0*individuo.getFitness() +
									 (double(individuo.getNumeroDrones() - minDrones) / double(topDrones - minDrones)) +
									 (1.0 -((individuo.getConfiabilidade() - minConfiabilidade)/(topConfiabilidade - minConfiabilidade)))+
									 (double(individuo.getVazao() - minVazao)/double(topVazao - minVazao)) +
									 penalidade
						 	 	 	)
				 	 	 	 );
		 somaFitness += individuo.getFitness();
	 }


 }
void Passo7::elitizarIndividuos(){
	 //elitismo
	individuosFinais.clear();
	 for(int i = 0; i < 20; i ++){
		 int maior = 0;
		 double maximo = individuosIniciais.at(0).getFitness();
		 for(int j = 1; j < individuosIniciais.size(); j ++){
			 if(individuosIniciais.at(j).getFitness() > maximo){
				 maior = j;
				 maximo = individuosIniciais.at(j).getFitness();
			 }
		 }
		 individuosFinais.push_back(individuosIniciais.at(maior));
		 somaFitness -= individuosIniciais.at(maior).getFitness();
		 individuosIniciais.erase(individuosIniciais.begin() + maior);
	 }
}
 void Passo7::exec() {
 
	 	 setFiltroPersiste("I__");
	 	 Individuo oldFirstIndividuo(getPathArquivo()+"/I__001");

 
	 	 this->atualizaCiclo(getPathArquivo()+"/");//Recupera o ciclo (arquivo ciclo) e atualiza variavel cicloAtual
	 	 this->recuperaCicloUltimaTrocaMaiorFitness(getPathArquivo()+"/");


		 std::random_device rd;
		 std::mt19937 gen(rd());
		 std::uniform_real_distribution<> dis(0.0, 0.99999);
		 individuosIniciais.clear();
		 this->recuperaIndividuos("I__", individuosIniciais);
		 this->recuperaIndividuos("MG_", individuosIniciais);


		 //Remove todos os arquivos da pasta
		 removeAllFiles();

		 //avalia
		 this->calculaFintess();
		 this->elitizarIndividuos();
 
 
		 //roleta
		 while(individuosFinais.size() < getMaxSortResults()){
			 std::vector<double> chances(individuosIniciais.size(), 0.0);
			 chances[0] = individuosIniciais.at(0).getFitness() / somaFitness;
			 for(int i = 1; i < individuosIniciais.size(); i ++){
				 chances[i] = chances[i-1] + individuosIniciais.at(i).getFitness() / somaFitness;
			 }
			 double random = dis(gen);
			 int i;
			 for(i = 0; i < individuosIniciais.size(); i ++){
				 if(random < chances[i]){
					 break;
				 }
			 }
			 individuosFinais.push_back(individuosIniciais.at(i));
			 somaFitness -= individuosIniciais.at(i).getFitness();
			 individuosIniciais.erase(individuosIniciais.begin() + i);
		 }
 
		 //persiste
		 for(int i = 0; i < individuosFinais.size(); i ++){
			 individuosFinais.at(i).persisteIndividuo(getPathArquivo()+"/"+getFileName("I__", i+1, std::to_string(getMaxSortResults()).length()));
		 }
		 if ((individuosFinais[0].getVazao() != oldFirstIndividuo.getVazao()) ||
			 (individuosFinais[0].getNumeroDrones() != oldFirstIndividuo.getNumeroDrones()) ||
			 (individuosFinais[0].getConfiabilidade() != oldFirstIndividuo.getConfiabilidade()))
			 	 cicloUltimaTrocaMaiorFitness = this->getCicloAtual();


		 this->persisteCicloUltimaTrocaMaiorFitness(getPathArquivo()+"/");

 
 }

 void Passo7::persisteCicloUltimaTrocaMaiorFitness(std::string pathArquivo){
	ofstream MyFile;
	try {
		long int cicloM = cicloUltimaTrocaMaiorFitness;
		MyFile.open(pathArquivo+"/CicloUltimaTrocaMaiorFitness");
		MyFile << cicloM << '\n';
		MyFile.close();
	} catch (const std::exception& e) {
		 std::cerr << "Error opening or writing file: " << e.what() << "\n";
			if (MyFile.is_open()) {
				MyFile.close();
			}
			exit(-1); // Indicate failure
		}

 }

 void Passo7::recuperaCicloUltimaTrocaMaiorFitness(std::string pathArquivo){
		int cicloUltima = getMaxCiclos();
		std::ifstream MyFile;

		try {
			MyFile.open(pathArquivo+"/CicloUltimaTrocaMaiorFitness");
			if (MyFile.is_open()){
				std::string linha;
				std::getline(MyFile,linha);
				cicloUltima = std::stoi(linha);
				MyFile.close();
				cicloUltimaTrocaMaiorFitness = cicloUltima;
			}
		}catch(...){
			std::cout << "Arquivo CicloUltimaTrocaMaiorFitness nao encontrado\n";
		}


 }
