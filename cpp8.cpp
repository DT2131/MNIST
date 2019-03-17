#include <bits/stdc++.h>
#include <random>
#include <conio.h>
using namespace std;

typedef struct HiddenLayer {

	//	data structure

	int neuronNumber;
	string layerType; 
	string activationFunction;
	vector <double> weights;
	vector <double> neuronsAfterActive;
	vector <double> bias;
	vector <double> neuronsValue;
	vector <double> lossValue;
	vector <double> gradientsValue;
	double learningRate;
	vector <double> weightsadjust;
	vector <double> biasadjust;

	//	initialization

	HiddenLayer() {}
	HiddenLayer(string _layerType, string _activationFunction, int _neuronNumber, double _learningRate) {
		layerType = _layerType;
		activationFunction = _activationFunction;
		neuronNumber = _neuronNumber;
		learningRate = _learningRate;
	}


	//	activation functions

	void active(int _leak = 0.1) {
		if (activationFunction == "sigmoid") {
			for (int i = 0; i < neuronNumber; i++) {
				neuronsValue[i] = 1.0 / (1.0 + exp(-neuronsAfterActive[i]));
			}
		}
		else if (activationFunction == "relu") {
			for (int i = 0; i < neuronNumber; i++) {
				neuronsValue[i] = max(neuronsAfterActive[i], 0.0);
			}
		}
		else if(activationFunction == "leakyRelu") {
			for (int i = 0; i < neuronNumber; i++) {
				neuronsValue[i] = neuronsAfterActive[i] * (neuronsAfterActive[i] < 0 ? _leak : 1);
			}
		}
		else {
			//	.......
		}
	}
};

typedef struct OutputLayer {
	int neuronNumber;
	string lossFunction;
	string classifier;
	vector <double> neuronsAfterClassify;
	vector <double> neuronsValue;
	vector <double> lossValue;
	vector <double> gradientsValue;
	double costSum;
	double answer;
	vector <double> answerVector;
	double Accuracy;
	double Top1Accuracy;
	double Top3Accuracy;
	double Top5Accuracy;

	//	initialization

	OutputLayer() {}
	OutputLayer(string _lossFunction, string _classifier) {
		lossFunction = _lossFunction;
		classifier = _classifier;
	}

	//	calculate accracy

	void accracy() {
		Accuracy = 0;
		Top1Accuracy = 0;
		Top3Accuracy = 0;
		Top5Accuracy = 0;
		vector<pair<double, int> >outputs;
		for (int i = 0; i < neuronNumber; i++) {
			outputs.push_back({ neuronsValue[i],i });
		}
		sort(outputs.begin(), outputs.end());
		if (outputs[neuronNumber - 1].second == answer) {
			Accuracy = 1;
		}
		for (int i = neuronNumber - 1; i >= neuronNumber / 2; i--) {
			if (outputs[i].second == answer) {
				if (i >= (neuronNumber - neuronNumber * 0.1)) {
					Top1Accuracy = 1;
				}
				if (i >= (neuronNumber - neuronNumber * 0.3)) {
					Top3Accuracy = 1;
				}
				if (i >= (neuronNumber - neuronNumber * 0.5)) {
					Top5Accuracy = 1;
					break;
				}
			}
		}
	}

	//	classify

	void classify() {
		if (classifier == "softmax") {
			double sum = 0;
			for (int i = 0; i < neuronNumber; i++) {
				neuronsValue[i] = exp(neuronsAfterClassify[i]);
				sum += neuronsValue[i];
			}
			for (int i = 0; i < neuronNumber; i++) {
				neuronsValue[i] /= sum;
			}
		}
		else {
			//	......
		}
	}
};

typedef struct InputLayer {
	int neuronNumber;
	vector <double> neuronsValue;
	vector <double> neuronsAfterNormalize;
	
	//	initialization

	InputLayer() {}
	InputLayer(int _neuronNumber) {
		neuronNumber = _neuronNumber;
	}

	//	noramlization

	void normalize(int maximum) {
		for (int i = 0; i < neuronNumber; i++) {
			neuronsValue[i] = neuronsAfterNormalize[i] / maximum;
		}
	}

};

typedef struct Batch {
	int batchSize;
	int batchNumber;
	vector <vector<double> >weightsadjust;
	vector <vector<double> >biasadjust;
	double batchAccuracy;
	double batchTop1Accuracy;
	double batchTop3Accuracy;
	double batchTop5Accuracy;
	double batchCost;
	bool batchShift;
	int simpleId;
	//	initialization

	Batch() {}
	Batch(int _batchNumber, int _batchSize, bool _batchShift = 0) {
		batchNumber = _batchNumber;
		batchSize = _batchSize;
		batchShift = _batchShift;
		simpleId = 0;
	}

	void accuracy(OutputLayer& output) {
		batchCost += output.costSum;
		batchAccuracy += output.Accuracy;
		batchTop1Accuracy += output.Top1Accuracy;
		batchTop3Accuracy += output.Top3Accuracy;
		batchTop5Accuracy += output.Top5Accuracy;
	}

};

typedef struct DataSet {
	int number;
	int dimonsion;
	int checker;
	int singleDataSize;
	int singleLabelVectorSize;
	vector <int> size;
	vector <double> data;
	vector <double> labels;
	vector <double> labelsVector;
	unsigned int in(ifstream& icin, unsigned int size) {
		unsigned int ans = 0;
		for (int i = 0; i < size; i++) {
			unsigned char x;
			icin.read((char*)&x, 1);
			unsigned int temp = x;
			ans <<= 8;
			ans += temp;
		}
		return ans;
	}

	void MNISTInput(const char dataDir[],const char labelsDir[]) {
		ifstream icin;
		icin.open(dataDir, ios::binary);
		dimonsion = 2;
		checker = in(icin, 4), number = in(icin, 4), size.push_back(in(icin, 4)), size.push_back(in(icin, 4));
		for (int i = 0; i < number; i++) {
			for (int x = 0; x < size[0]; x++) {
				for (int y = 0; y < size[1]; y++) {
					data.push_back(in(icin, 1));
				}
			}
		}
		singleDataSize = data.size() / number;
		icin.close();
		icin.open(labelsDir, ios::binary);
		checker = in(icin, 4), number = in(icin, 4);
		for (int i = 0; i < number; i++) {
			int temp = in(icin, 1);
			labels.push_back(temp);
			for (int j = 0; j < temp; j++) labelsVector.push_back(0);
			labelsVector.push_back(1);
			for (int j = temp + 1; j < 10; j++) labelsVector.push_back(0);
		}
		singleLabelVectorSize = labelsVector.size() / number;
	}
};

typedef struct Model {

	InputLayer inputLayer;
	vector <HiddenLayer> hiddenLayers;
	OutputLayer outputLayer;
	Batch batch;
	//	initialization

	void ini() {
		for (int i = 0; i < inputLayer.neuronNumber; i++) {
			inputLayer.neuronsValue.push_back(0);
			inputLayer.neuronsAfterNormalize.push_back(0);
		}
		batch.weightsadjust.resize(hiddenLayers.size());
		batch.biasadjust.resize(hiddenLayers.size());
		for (int i = 0; i < hiddenLayers.size(); i++) {
			for (int j = 0; j < hiddenLayers[i].neuronNumber; j++) {
				hiddenLayers[i].neuronsValue.push_back(0);
				hiddenLayers[i].neuronsAfterActive.push_back(0);
				hiddenLayers[i].bias.push_back(0);
				hiddenLayers[i].biasadjust.push_back(0);
				hiddenLayers[i].gradientsValue.push_back(0);
				hiddenLayers[i].lossValue.push_back(0);
				batch.biasadjust[i].push_back(0);
			}
			if (i == 0) {
				for (int j = 0; j < hiddenLayers[i].neuronNumber*inputLayer.neuronNumber; j++) {
					hiddenLayers[i].weights.push_back(0);
					hiddenLayers[i].weightsadjust.push_back(0);
					batch.weightsadjust[i].push_back(0);
				}
			}
			else {
				for (int j = 0; j < hiddenLayers[i].neuronNumber*hiddenLayers[i - 1].neuronNumber; j++) {
					hiddenLayers[i].weights.push_back(0);
					hiddenLayers[i].weightsadjust.push_back(0);
					batch.weightsadjust[i].push_back(0);
				}
			}
		}
		outputLayer.neuronNumber = hiddenLayers[hiddenLayers.size() - 1].neuronNumber;
		for (int i = 0; i < outputLayer.neuronNumber; i++) {
			outputLayer.neuronsValue.push_back(0);
			outputLayer.neuronsAfterClassify.push_back(0);
			outputLayer.answerVector.push_back(0);
			outputLayer.lossValue.push_back(0);
			outputLayer.gradientsValue.push_back(0);
		}
	}

	//	data input

	void fill_data(DataSet& data, int id) {
		
		for (int i = 0; i < data.singleDataSize; i++) {
			inputLayer.neuronsValue[i] = data.data[id * data.singleDataSize + i] / 255;
		}
		outputLayer.answer = data.labels[id];
		for (int i = 0; i < outputLayer.neuronNumber; i++) {
			outputLayer.answerVector[i] = data.labelsVector[id * data.singleLabelVectorSize + i];
		}
		
	}

	//	normal distribution weights and bias

	void normalWB(int u, int o) {
		default_random_engine e(rand());
		normal_distribution<double> normal(u, o);
		for (int i = 0; i < hiddenLayers.size(); i++) {
			for (int j = 0; j < hiddenLayers[i].neuronNumber; j++) {
				hiddenLayers[i].bias[j] = normal(e);
				hiddenLayers[i].biasadjust[j] = normal(e);
			}
			if (i == 0) {
				for (int j = 0; j < hiddenLayers[i].neuronNumber*inputLayer.neuronNumber; j++) {
					hiddenLayers[i].weights[j] = normal(e);
					hiddenLayers[i].weightsadjust[j] = normal(e);
				}
			}
			else {
				for (int j = 0; j < hiddenLayers[i].neuronNumber*hiddenLayers[i - 1].neuronNumber; j++) {
					hiddenLayers[i].weights[j] = normal(e);
					hiddenLayers[i].weightsadjust[j] = normal(e);
				}
			}
		}
	}

	//	load weights and bias

	void loadWB(string dir) {

	}

	//	save weights and bias

	void saveWB(const char a[]) {
		freopen(a, "w", stdout);

	}
	//	calculate gradient

	void gradient() {
		
		//	outputlayer
		//	calculate cost and loss
		
		outputLayer.costSum = 0;
		if (outputLayer.lossFunction == "squaredError") {
			for (int i = 0; i < outputLayer.neuronNumber; i++) {
				outputLayer.costSum += (outputLayer.neuronsValue[i] - outputLayer.answerVector[i]) * (outputLayer.neuronsValue[i] - outputLayer.answerVector[i]);
				outputLayer.lossValue[i] = 2 * (outputLayer.neuronsValue[i] - outputLayer.answerVector[i]);
			}
		}
		else if (outputLayer.lossFunction == "crossEntropy") {
			for (int i = 0; i < outputLayer.neuronNumber; i++) {
				outputLayer.costSum += -outputLayer.answerVector[i] * log(outputLayer.neuronsValue[i]);
				outputLayer.lossValue[i] = -outputLayer.answerVector[i] / outputLayer.neuronsValue[i];
			}
		}
		else {
			//	......

		}
		
		//	calculate gradient

		if (outputLayer.classifier == "softmax") {
			for (int i = 0; i < outputLayer.neuronNumber; i++) {
				outputLayer.gradientsValue[i] = 0;
				for (int j = 0; j < outputLayer.neuronNumber; j++) {
					if (i == j) {
						outputLayer.gradientsValue[i] += outputLayer.neuronsValue[i] * (1 - outputLayer.neuronsValue[i]) * outputLayer.lossValue[i];
					}
					else {
						outputLayer.gradientsValue[i] += (-outputLayer.neuronsValue[j] * outputLayer.neuronsValue[i]) * outputLayer.lossValue[j];
					}
				}
			}
		}
		else {
			//	......
		}

		//	hiddenlayer
		
		for (int i = hiddenLayers.size() - 1; i >= 0; i--) {

			//	calculate loss

			if (i == hiddenLayers.size() - 1) {
				for (int u = 0; u < hiddenLayers[i].neuronNumber; u++) {
					hiddenLayers[i].lossValue[u] = outputLayer.gradientsValue[u];
				}
			}
			else {
				for (int u = 0; u < hiddenLayers[i].neuronNumber; u++) {
					hiddenLayers[i].lossValue[u] = 0;
					for (int v = 0; v < hiddenLayers[i + 1].neuronNumber; v++) {
						hiddenLayers[i].lossValue[u] += hiddenLayers[i + 1].weights[u * hiddenLayers[i + 1].neuronNumber + v] * hiddenLayers[i + 1].gradientsValue[v];
					}
				}
			}

			//	calculate gradient

			if (hiddenLayers[i].activationFunction == "sigmoid") {
				for (int j = 0; j < hiddenLayers[i].neuronNumber; j++) {
					hiddenLayers[i].gradientsValue[j] = hiddenLayers[i].neuronsValue[j] * (1 - hiddenLayers[i].neuronsValue[j]) * hiddenLayers[i].lossValue[j];
				}
			}
			else if (hiddenLayers[i].activationFunction == "relu") {
				//	......
			}
			else if (hiddenLayers[i].activationFunction == "prelu") {
				//	......
			}
			else {
				//	......
			}
			
		}
	}



	//	train

	void train(DataSet& data) {
		for (int i = 0; i < batch.batchNumber; i++) {

			//	batch start

			//	batch ini

			for (int j = 0; j < batch.weightsadjust.size(); j++) {
				for (int k = 0; k < batch.weightsadjust[j].size(); k++) {
					batch.weightsadjust[j][k] = 0;
				}
				for (int k = 0; k < batch.biasadjust[j].size(); k++) {
					batch.biasadjust[j][k] = 0;
				}
			}
			batch.batchAccuracy = 0;
			batch.batchTop1Accuracy = 0;
			batch.batchTop3Accuracy = 0;
			batch.batchTop5Accuracy = 0;
			batch.batchCost = 0;

			//	start point random


			if (batch.batchShift) {
				srand((unsigned)time(0));
				batch.simpleId = rand()*rand();
				batch.simpleId %= data.number;
			}

			for (int j = 0; j < batch.batchSize; j++, batch.simpleId++,batch.simpleId%=data.number) {

				//	data input

				fill_data(data, batch.simpleId);

				//	model ini

				for (int k = 0; k < hiddenLayers.size(); k++) {
					for (int u = 0; u < hiddenLayers[k].neuronNumber; u++) {
						hiddenLayers[k].neuronsAfterActive[u] = 0;
					}
				}

				//	forward propagation
				//	hiddenlayer foward

				for (int k = 0; k < hiddenLayers.size(); k++) {
					if (hiddenLayers[k].layerType == "fullconnect") {
						if (k == 0) {
							for (int u = 0; u < inputLayer.neuronNumber; u++) {
								for (int v = 0; v < hiddenLayers[k].neuronNumber; v++) {
									hiddenLayers[k].neuronsAfterActive[v] += inputLayer.neuronsValue[u] * hiddenLayers[k].weights[u * hiddenLayers[k].neuronNumber + v];
								}
							}
							for (int u = 0; u < hiddenLayers[k].neuronNumber; u++) {
								hiddenLayers[k].neuronsAfterActive[u] += hiddenLayers[k].bias[u];
							}
						}
						else {
							for (int u = 0; u < hiddenLayers[k - 1].neuronNumber; u++) {
								for (int v = 0; v < hiddenLayers[k].neuronNumber; v++) {
									hiddenLayers[k].neuronsAfterActive[v] += hiddenLayers[k - 1].neuronsValue[u] * hiddenLayers[k].weights[u * hiddenLayers[k].neuronNumber + v];
								}
							}
							for (int u = 0; u < hiddenLayers[k].neuronNumber; u++) {
								hiddenLayers[k].neuronsAfterActive[u] += hiddenLayers[k].bias[u];
							}
						}
					}
					else {
						//	......
					}
					hiddenLayers[k].active();
				}
				//	outputlayer forward

				for (int u = 0; u < outputLayer.neuronNumber; u++) {
					outputLayer.neuronsAfterClassify[u] = hiddenLayers[hiddenLayers.size() - 1].neuronsValue[u];
				}
				outputLayer.classify();
				
				//	calculate gradient

				gradient();

				//	back propagation
				//	hiddenlayer backward

				for (int k = hiddenLayers.size() - 1; k >= 0; k--) {
					if (hiddenLayers[k].layerType == "fullconnect") {

						//	calculate weightsadjust and biasadjust 

						if (k == 0) {
							for (int u = 0; u < inputLayer.neuronNumber; u++) {
								for (int v = 0; v < hiddenLayers[k].neuronNumber; v++) {
									hiddenLayers[k].weightsadjust[u * hiddenLayers[k].neuronNumber + v] = inputLayer.neuronsValue[u] * hiddenLayers[k].gradientsValue[v];
									batch.weightsadjust[k][u * hiddenLayers[k].neuronNumber + v] += hiddenLayers[k].weightsadjust[u * hiddenLayers[k].neuronNumber + v];
								}
							}
							for (int u = 0; u < hiddenLayers[k].neuronNumber; u++) {
								hiddenLayers[k].biasadjust[u] = hiddenLayers[k].gradientsValue[u];
								batch.biasadjust[k][u] += hiddenLayers[k].biasadjust[u];
							}
						}
						else {
							for (int u = 0; u < hiddenLayers[k - 1].neuronNumber; u++) {
								for (int v = 0; v < hiddenLayers[k].neuronNumber; v++) {
									hiddenLayers[k].weightsadjust[u * hiddenLayers[k].neuronNumber + v] = hiddenLayers[k - 1].neuronsValue[u] * hiddenLayers[k].gradientsValue[v];	
									batch.weightsadjust[k][u * hiddenLayers[k].neuronNumber + v] += hiddenLayers[k].weightsadjust[u * hiddenLayers[k].neuronNumber + v];
								}
							}
							for (int u = 0; u < hiddenLayers[k].neuronNumber; u++) {
								hiddenLayers[k].biasadjust[u] = hiddenLayers[k].gradientsValue[u];
								batch.weightsadjust[k][u] += hiddenLayers[k].biasadjust[u];
							}
						}
					}
					else {
						//	......
					}


				}

				//	calculate accuracy

				outputLayer.accracy();
				batch.accuracy(outputLayer);
			}

			//	batch end

			//	adjust weights and bias

			for (int j = 0; j < hiddenLayers.size(); j++) {
				if (j == 0) {
					for (int u = 0; u < inputLayer.neuronNumber; u++) {
						for (int v = 0; v < hiddenLayers[j].neuronNumber; v++) {
							batch.weightsadjust[j][u * hiddenLayers[j].neuronNumber + v] /= batch.batchSize;
							batch.weightsadjust[j][u * hiddenLayers[j].neuronNumber + v] *= hiddenLayers[j].learningRate;
							hiddenLayers[j].weights[u * hiddenLayers[j].neuronNumber + v] -= batch.weightsadjust[j][u * hiddenLayers[j].neuronNumber + v];
						}
					}
					for (int u = 0; u < hiddenLayers[j].neuronNumber; u++) {
						batch.biasadjust[j][u] /= batch.batchSize;
						batch.biasadjust[j][u] *= hiddenLayers[j].learningRate;
						hiddenLayers[j].bias[u] -= batch.biasadjust[j][u];
					}
				}
				else {
					for (int u = 0; u < hiddenLayers[j - 1].neuronNumber; u++) {
						for (int v = 0; v < hiddenLayers[j].neuronNumber; v++) {
							batch.weightsadjust[j][u * hiddenLayers[j].neuronNumber + v] /= batch.batchSize;
							batch.weightsadjust[j][u * hiddenLayers[j].neuronNumber + v] *= hiddenLayers[j].learningRate;
							hiddenLayers[j].weights[u * hiddenLayers[j].neuronNumber + v] -= batch.weightsadjust[j][u * hiddenLayers[j].neuronNumber + v];
						}
					}
					for (int u = 0; u < hiddenLayers[j].neuronNumber; u++) {
						batch.biasadjust[j][u] /= batch.batchSize;
						batch.biasadjust[j][u] *= hiddenLayers[j].learningRate;
						hiddenLayers[j].bias[u] -= batch.biasadjust[j][u];
					}
				}
			}

			//	output inf
			//if (i % 100 == 0) {
			//	cout << "Batch No. " << i << " Average Batch loss " << batch.batchCost / batch.batchSize << " Average Batch accuracy " << batch.batchAccuracy / batch.batchSize << endl;
			//	for (int i = 0; i < 10; i++) {
			//		cout << outputLayer.neuronsValue[i] << endl;
			//	}
			//}
		}
	}

	void test(DataSet& data) {
		double ok = 0;
		for (int i = 0; i < data.number; i++) {
			for (int j = 0; j < data.singleDataSize; j++) {
				inputLayer.neuronsValue[j] = data.data[i * data.singleDataSize + j] / 255;
			}
			outputLayer.answer = data.labels[i];
			for (int j = 0; j < outputLayer.neuronNumber; j++) {
				outputLayer.answerVector[j] = data.labelsVector[i * data.singleLabelVectorSize + j];
			}
			for (int j = 0; j < hiddenLayers.size(); j++) {
				for (int u = 0; u < hiddenLayers[j].neuronNumber; u++) {
					hiddenLayers[j].neuronsAfterActive[u] = 0;
				}
			}
			//	forward propagation
			//	hiddenlayer foward
			for (int j = 0; j < hiddenLayers.size(); j++) {
				if (hiddenLayers[j].layerType == "fullconnect") {
					if (j == 0) {
						for (int u = 0; u < inputLayer.neuronNumber; u++) {
							for (int v = 0; v < hiddenLayers[j].neuronNumber; v++) {
								hiddenLayers[j].neuronsAfterActive[v] += inputLayer.neuronsValue[u] * hiddenLayers[j].weights[u * hiddenLayers[j].neuronNumber + v];
							}
						}
						for (int u = 0; u < hiddenLayers[j].neuronNumber; u++) {
							hiddenLayers[j].neuronsAfterActive[u] += hiddenLayers[j].bias[u];
						}
					}
					else {
						for (int u = 0; u < hiddenLayers[j - 1].neuronNumber; u++) {
							for (int v = 0; v < hiddenLayers[j].neuronNumber; v++) {
								hiddenLayers[j].neuronsAfterActive[v] += hiddenLayers[j - 1].neuronsValue[u] * hiddenLayers[j].weights[u * hiddenLayers[j].neuronNumber + v];
							}
						}
						for (int u = 0; u < hiddenLayers[j].neuronNumber; u++) {
							hiddenLayers[j].neuronsAfterActive[u] += hiddenLayers[j].bias[u];
						}
					}
				}
				else {
					//	......
				}
				hiddenLayers[j].active();
			}
			//	outputlayer forward

			for (int u = 0; u < outputLayer.neuronNumber; u++) {
				outputLayer.neuronsAfterClassify[u] = hiddenLayers[hiddenLayers.size() - 1].neuronsValue[u];
			}
			outputLayer.classify();
			outputLayer.accracy();
			if (outputLayer.Accuracy) ok++;
		}
		cout << " ANN Accuracy:" << ok / data.number << endl << "Hello world";
	}
};


void buildModel(Model& model, Batch& batch) {
	model.inputLayer = {
		784
	};
	model.hiddenLayers.push_back({
		"fullconnect",
		"sigmoid",
		16,
		3
		});
	model.hiddenLayers.push_back({
		"fullconnect",
		"sigmoid",
		10,
		3
		});
	model.outputLayer = {
		"crossEntropy",
		"softmax"
	};
	model.batch = batch;
}
void buildBatch(Batch& batch) {
	batch = {
		3000,
		20,
		1
	};
}
DataSet mnistTrain,mnistTest;
Model myModel;
Batch myBatch;
int main()
{
	mnistTrain.MNISTInput("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
	mnistTest.MNISTInput("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	buildBatch(myBatch);
	buildModel(myModel, myBatch);
	myModel.ini();
	myModel.normalWB(0, 1);
	myModel.train(mnistTrain);
	myModel.test(mnistTest);
}
