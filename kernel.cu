#include <bits/stdc++.h>
#include <random>
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
#include <graphics.h>
#include <conio.h>
using namespace std;

typedef struct Layer {

	//	data structure

	string layerType;
	int neuronNumber;
	string activationFunction;
	vector <double> bias;
	vector <double> neuronsValue;
	vector <double> weights;
	vector <double> weightsSingleAdjust;
	vector <double> biasSingleAdjust;
	vector <double> neuronsadjust;
	vector <double> weightsBatchAdjust;
	vector <double> biasBatchAdjust;
	double learningRate;

	//	initialize

	Layer(string _layerType, int _neuronNumber, string _activationFunction, double _learningRate = 1) {
		layerType = _layerType;
		neuronNumber = _neuronNumber;
		activationFunction = _activationFunction;
		learningRate = _learningRate;
		for (int i = 0; i < neuronNumber; i++) neuronsValue.push_back(0);
		for (int i = 0; i < neuronNumber; i++) biasSingleAdjust.push_back(0);
		for (int i = 0; i < neuronNumber; i++) biasBatchAdjust.push_back(0);
		for (int i = 0; i < neuronNumber; i++) neuronsadjust.push_back(0);
	}

	//	activation functions

	void sigmoid() {
		for (int i = 0; i < neuronNumber; i++) {
			neuronsValue[i] = 1.0 / (1.0 + exp(-neuronsValue[i]));
			if (isnan(neuronsValue[i])) {
				neuronsValue[i] = 0;
			}
		}
	}
}Layer;

typedef struct DataSet {
	int number;
	int dimonsion;
	int checker;
	int singleDataSize;
	int singleLabelSize;
	vector <int> size;
	vector <double> data;
	vector <double> labels;
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
	void MNISTInputDebug() {
		for (int i = 0; i < number; i++) {
			initgraph(50, 50);
			i = 1200;
			for (int x = 0; x < size[0]; x++) {
				for (int y = 0; y < size[1]; y++) {
					if (data[i * size[0] * size[1] + x * size[1] + y]) putpixel(y, x, WHITE);
				}
			}
			Sleep(5000);
		}
	}
	void MNISTinput() {
		ifstream icin;
		icin.open("train-images.idx3-ubyte", ios::binary);
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
		icin.open("train-labels.idx1-ubyte", ios::binary);
		checker = in(icin, 4), number = in(icin, 4);
		for (int i = 0; i < number; i++) {
			int temp = in(icin, 1);
			for (int j = 0; j < temp; j++) labels.push_back(0);
			labels.push_back(1);
			for (int j = temp + 1; j < 10; j++) labels.push_back(0);
		}
		singleLabelSize = labels.size() / number;
	}
}DataSet;

typedef struct Model {

	vector <Layer> layers;

	int layerSize;

	string costFunction;

	Model(string _costFunction) {
		costFunction = _costFunction;
	}

	void train(int batchNumber, int batchSize, bool batchshift, DataSet& data) {

		//	initialize layerSize, weights and bias

		layerSize = layers.size();
		srand((unsigned)time(NULL));
		default_random_engine e(rand());
		normal_distribution<double> normal(0, 1);
		for (int i = 1; i < layerSize; i++) {
			for (int j = 0; j < layers[i].neuronNumber * layers[i - 1].neuronNumber; j++) {
				double temp = normal(e);
				temp = min(temp, 5.0);
				temp = max(temp, -5.0);
				layers[i].weights.push_back(temp);
				layers[i].weightsBatchAdjust.push_back(0);
				layers[i].weightsSingleAdjust.push_back(0);
			}
			for (int j = 0; j < layers[i].neuronNumber; j++) {
				double temp = normal(e);
				temp = min(temp, 5.0);
				temp = max(temp, -5.0);
				layers[i].bias.push_back(temp);
				layers[i].biasSingleAdjust.push_back(0);
				layers[i].biasBatchAdjust.push_back(0);
			}
		}

		//	train

		int simple = 0;
		for (int i = 0; i < batchNumber; i++) {

			//	initialize weightsBatchadjust and biasBatchadjust

			for (int j = 1; j < layerSize; j++) {
				for (int k = 0; k < layers[j].neuronNumber * layers[j - 1].neuronNumber; k++) {
					layers[j].weightsBatchAdjust[k] = 0;
				}
				for (int k = 0; k < layers[j].neuronNumber; k++) {
					layers[j].biasBatchAdjust[k] = 0;
				}
			}


			//	single batch begin

			double batchCost = 0, batchAccuracy = 0;		//	batch accuracy

			if (batchshift) simple = (rand() * rand()) % data.number;
			for (int j = 0; j < batchSize; j++) {

				//	dataInput
				
				//	NEED NORMALIZETION HERE ?

				for (int k = 0; k < data.singleDataSize; k++) {
					layers[0].neuronsValue[k] = data.data[(simple*data.singleDataSize + k) % (data.number*data.singleDataSize)] / 255.0;
				}

				//	initialize weightsSingleasjust and biasSingleAdjust

				for (int k = 1; k < layerSize; k++) {
					for (int u = 0; u < layers[k].neuronNumber*layers[k - 1].neuronNumber; u++) {
						layers[k].weightsSingleAdjust[u] = 0;
					}
					for (int u = 0; u < layers[k].neuronNumber; u++) {
						layers[k].biasSingleAdjust[u] = 0;
					}
				}

				//	forward propagation

				for (int k = 1; k < layers.size(); k++) {
					if (layers[k].layerType == "fullConnect") {

						//	times weights

						for (int u = 0; u < layers[k - 1].neuronNumber; u++) {
							for (int v = 0; v < layers[k].neuronNumber; v++) {

								//	TEST W 
								if (isnan(layers[k].weights[u * layers[k].neuronNumber + v]))
									layers[k].weights[u * layers[k].neuronNumber + v] = 0;

								layers[k].neuronsValue[v] = layers[k - 1].neuronsValue[u] * layers[k].weights[u * layers[k].neuronNumber + v];

								//	TEST BA 
								if (isnan(layers[k].neuronsValue[v])) {
									layers[k].neuronsValue[v] = 0;
								}
							}
						}

						//	adds bias

						for (int v = 0; v < layers[k].neuronNumber; v++) {
							layers[k].neuronsValue[v] += layers[k].bias[v];

							//	TEST B

							if (isnan(layers[k].neuronsValue[v])) {
								layers[k].neuronsValue[v] = 0;
							}
						}
						// activate

						if (layers[k].activationFunction == "sigmoid") {
							layers[k].sigmoid();
						}
						else {
							//	......
						}
					}
					else {
						//	......
					}
				}

				//	backforward propagation

				if (costFunction == "squaredError") {
					for (int k = layerSize - 1; k > 0; k--) {
						if (layers[k].layerType == "fullConnect") {
							if (layers[k].activationFunction == "sigmoid") {

								//	calculate single weights adjustment and save neurons adjustment

								for (int v = 0; v < layers[k].neuronNumber; v++) {
									if (k == layerSize - 1) {
										layers[k].neuronsadjust[v] = 2 * (layers[k].neuronsValue[v] - data.labels[simple * data.singleLabelSize + v]);
									}
									else {
										layers[k].neuronsadjust[v] = 0;
										for (int u = 0; u < layers[k + 1].neuronNumber; u++) {
											layers[k].neuronsadjust[v] += layers[k + 1].weights[v * layers[k + 1].neuronNumber + u] * layers[k + 1].neuronsValue[u] * (1 - layers[k + 1].neuronsValue[u]) * layers[k + 1].neuronsadjust[u];
										}
									}
									for (int u = 0; u < layers[k - 1].neuronNumber; u++) {
										if (isnan(layers[k].neuronsadjust[v])) {
											layers[k].neuronsadjust[v] = 0;
										}
										if (isnan(layers[k].neuronsValue[v])) {
											layers[k].neuronsValue[v] = 0;
										}
										if (isnan(layers[k - 1].neuronsValue[u])) {
											layers[k - 1].neuronsValue[u] = 0;
										}
										layers[k].weightsSingleAdjust[u * layers[k].neuronNumber + v] = layers[k].neuronsadjust[v] * layers[k].neuronsValue[v] * (1.0 - layers[k].neuronsValue[v]) * layers[k - 1].neuronsValue[u];
										if (isnan(layers[k].weightsSingleAdjust[u * layers[k].neuronNumber + v])) {
											layers[k].weightsSingleAdjust[u * layers[k].neuronNumber + v] = 0;
										}
										if (isinf(layers[k].weightsSingleAdjust[u * layers[k].neuronNumber + v])) {
											layers[k].weightsSingleAdjust[u * layers[k].neuronNumber + v] = 0;
										}
										layers[k].weightsBatchAdjust[u * layers[k].neuronNumber + v] += layers[k].weightsSingleAdjust[u * layers[k].neuronNumber + v];
										if (isnan(layers[k].weightsBatchAdjust[u * layers[k].neuronNumber + v])) {
											layers[k].weightsBatchAdjust[u * layers[k].neuronNumber + v] = 0;
										}
									}
								}

								// calculate single bias adjustment

								for (int v = 0; v < layers[k].neuronNumber; v++) {
									layers[k].biasSingleAdjust[v] = layers[k].neuronsadjust[v] * layers[k].neuronsValue[v] * (1 - layers[k].neuronsValue[v]);
									if (isnan(layers[k].biasSingleAdjust[v])) {
										layers[k].biasSingleAdjust[v] = 0;
									}
									if (fabs(layers[k].biasSingleAdjust[v]) > 1000)
										layers[k].biasSingleAdjust[v] = layers[k].biasSingleAdjust[v];
									layers[k].biasBatchAdjust[v] += layers[k].biasSingleAdjust[v];
								}
							}
							else {
								//	......
							}
						}
						else {
							// ......
						}
					}
				}
				else {
					//	......
				}
				
				//	accuracy count
				double tempmax = -1;
				int tempmaxnum = -1;
				for (int k = 0; k < layers[layerSize - 1].neuronNumber; k++) {
					batchCost += layers[layerSize - 1].neuronsadjust[k];
					if (layers[layerSize - 1].neuronsValue[k] > tempmax) {
						tempmax = layers[layerSize - 1].neuronsValue[k];
						tempmaxnum = k;
					}
				}
				if (tempmaxnum != -1 && data.labels[(simple * data.singleLabelSize + tempmaxnum) % data.number] == 1) batchAccuracy++;
				
				//	go to next simple
				
				simple++;
				simple %= data.number;
			}

			//	single bench end
			
			//	change weights and bias

			for (int k = 1; k < layerSize; k++) {
				for (int u = 0; u < layers[k - 1].neuronNumber; u++) {
					for (int v = 0; v < layers[k].neuronNumber; v++) {
						if (isnan(layers[k].weightsBatchAdjust[u * layers[k].neuronNumber + v])) {
							layers[k].weightsBatchAdjust[u * layers[k].neuronNumber + v] = 0;
						}
						layers[k].weightsBatchAdjust[u * layers[k].neuronNumber + v] /= batchSize;
						if (isnan(layers[k].weightsBatchAdjust[u * layers[k].neuronNumber + v])) {
							layers[k].weightsBatchAdjust[u * layers[k].neuronNumber + v] = 0;
						}
						layers[k].weightsBatchAdjust[u * layers[k].neuronNumber + v] *= layers[k].learningRate;
						if (isnan(layers[k].weightsBatchAdjust[u * layers[k].neuronNumber + v])) {
							layers[k].weightsBatchAdjust[u * layers[k].neuronNumber + v] = 0;
						}
						layers[k].weights[u * layers[k].neuronNumber + v] += layers[k].weightsBatchAdjust[u*layers[k].neuronNumber + v];
					}
				}
				for (int u = 0; u < layers[k].neuronNumber; u++) {
					layers[k].biasBatchAdjust[u] /= batchSize;
					layers[k].biasBatchAdjust[u] *= layers[k].learningRate;
					if (isnan(layers[k].biasBatchAdjust[u])) {
						layers[k].biasBatchAdjust[u] = 0;
					}
					if (fabs(layers[k].biasBatchAdjust[u]) > 1000)
						layers[k].biasBatchAdjust[u] = layers[k].biasBatchAdjust[u];
					layers[k].bias[u] += layers[k].biasBatchAdjust[u];
				}
			}

			//	calculate bench accurary
			//if (batchAccuracy == 0) {
			//	batchAccuracy = 0;
			//}
			cout << "Batch No. " << i << " Simple No. " << batchSize * i << " Cost: " << batchCost / batchSize << " Accuracy: " << batchAccuracy / batchSize << endl;
		}
	}
}Model;

Model myModel = {"squaredError"};
DataSet mnist;

void buildModel(Model& model) {
	model.layers.push_back({
		"fullConnect",
		28*28,
		"sigmoid",
		-0.5
		});
	model.layers.push_back({
		"fullConnect",
		25,
		"sigmoid",
		-0.5
		});
	model.layers.push_back({
		"fullConnect",
		10,
		"sigmoid",
		-0.5
		});
}

int main()
{
	mnist.MNISTinput();
	//mnist.MNISTInputDebug();
	buildModel(myModel);
	myModel.train(100000, 10, 0, mnist);
}