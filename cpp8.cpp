#include <bits/stdc++.h>
#include <random>
#include <conio.h>
using namespace std;

typedef struct hiddenLayer {

	//	data structure

	int neuronNumber;
	string layerType; 
	string activationFunction;
	vector <double> weights;
	vector <double> neuronsAfterActive;
	vector <double> bias;
	vector <double> neuronsValue;
	vector <double> gradient;
	double learningRate;
	vector <double> weightsadjust;
	vector <double> biasadjust;

	//	initialization

	hiddenLayer(string _layerType, string _activationFunction, int _neuronNumber, double _learningRate) {
		layerType = _layerType;
		activationFunction = _activationFunction;
		neuronNumber = _neuronNumber;
		learningRate = _learningRate;
	}


	//	activation functions

	void active(int _leak) {
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

typedef struct outputLayer {
	int neuronNumber;
	string lossFunction;
	string classifier;
	vector <double> neuronsAfterClassify;
	vector <double> neuronsValue;
	double learningRate;
	vector <double> lossValue;
	double lossSum;
	double answer;
	vector <double> answerVector;
	double Accuracy;
	double Top1Accuracy;
	double Top3Accuracy;
	double Top5Accuracy;

	//	initialization

	outputLayer(string _lossFunction, string _classifier, int _neuronNumber, double _learningRate) {
		lossFunction = _lossFunction;
		classifier = _classifier;
		neuronNumber = _neuronNumber;
		learningRate = _learningRate;
	}

	//	calculate loss

	void loss() {
		lossSum = 0;
		if (lossFunction == "squaredError") {
			for (int i = 0; i < neuronNumber; i++) {
				lossValue[i] = 2 * (neuronsValue[i] - answerVector[i]);
			}
		}
		else if (lossFunction=="crossEntropy") {
			for (int i = 0; i < neuronNumber; i++) {
				lossValue[i] = -answerVector[i] * log(neuronsValue[i]);
			}
		}
		else {
			//	......
		}
		for (int i = 0; i < neuronNumber; i++) {
			lossSum += lossValue[i];
		}
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
				else if (i >= (neuronNumber - neuronNumber * 0.3)) {
					Top3Accuracy = 1;
				}
				else if (i >= (neuronNumber - neuronNumber * 0.5)) {
					Top5Accuracy = 1;
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

typedef struct inputLayer {
	int neuronNumber;
	vector <double> neuronsValue;
	vector <double> neuronsAfterNormalize;
	
	//	initialization

	inputLayer(int _neuronNumber) {
		neuronNumber = _neuronNumber;
	}

	//	noramlization

	void normalize(int maximum) {
		for (int i = 0; i < neuronNumber; i++) {
			neuronsValue[i] = neuronsAfterNormalize[i] / maximum;
		}
	}

};

typedef struct batch {
	int batchSize;
	int batchNumber;
	double batchAccuracy;
	double batchTop1Accuracy;
	double batchTop3Accuracy;
	double batchTop5Accuracy;
	double batchLoss;
	bool batchShift;
	int batchPosition;
	//	initialization

	batch(int _batchNumber, int _batchSize, bool _batchShift = 0) {
		batchNumber = _batchNumber;
		batchSize = _batchSize;
		batchShift = _batchShift;
		batchPosition = 0;
	}

	void accuracy(outputLayer& output) {
		batchAccuracy += output.Accuracy;
		batchTop1Accuracy += output.Top1Accuracy;
		batchTop3Accuracy += output.Top3Accuracy;
		batchTop5Accuracy += output.Top5Accuracy;
	}

};

typedef struct dataSet {
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

	void MNISTInput() {
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
			labels.push_back(temp);
			for (int j = 0; j < temp; j++) labelsVector.push_back(0);
			labelsVector.push_back(1);
			for (int j = temp + 1; j < 10; j++) labelsVector.push_back(0);
		}
		singleLabelVectorSize = labelsVector.size() / number;
	}
};

typedef struct model {

	inputLayer inputLayer;
	vector <hiddenLayer> hiddenLayers;
	outputLayer outputLayer;

	//	initialization

	void ini() {
		for (int i = 0; i < inputLayer.neuronNumber; i++) {
			inputLayer.neuronsValue.push_back(0);
			inputLayer.neuronsAfterNormalize.push_back(0);
		}
		for (int i = 0; i < hiddenLayers.size(); i++) {
			for (int j = 0; j < hiddenLayers[i].neuronNumber; j++) {
				hiddenLayers[i].neuronsValue.push_back(0);
				hiddenLayers[i].neuronsAfterActive.push_back(0);
				hiddenLayers[i].bias.push_back(0);
				hiddenLayers[i].biasadjust.push_back(0);
				hiddenLayers[i].gradient.push_back(0);
			}
			if (i == 0) {
				for (int j = 0; j < hiddenLayers[i].neuronNumber*inputLayer.neuronNumber; j++) {
					hiddenLayers[i].weights.push_back(0);
					hiddenLayers[i].weightsadjust.push_back(0);
				}
			}
			else {
				for (int j = 0; j < hiddenLayers[i].neuronNumber*hiddenLayers[i - 1].neuronNumber; j++) {
					hiddenLayers[i].weights.push_back(0);
					hiddenLayers[i].weightsadjust.push_back(0);
				}
			}
		}
		for (int i = 0; i < outputLayer.neuronNumber; i++) {
			outputLayer.neuronsValue.push_back(0);
			outputLayer.neuronsAfterClassify.push_back(0);
			outputLayer.answerVector.push_back(0);
			outputLayer.lossValue.push_back(0);
		}
	}

	//	data input

	void fill_data(dataSet& data, int id) {
		
		for (int i = 0; i < data.singleDataSize; i++) {
			inputLayer.neuronsValue[i] = data.data[id * data.singleDataSize + i];
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

	//	train

	void train(batch& batch) {

		for (int i = 0; i < batch.batchNumber; i++) {
			for (int j = 0; j < batch.batchSize; j++) {
				
				if (batch.batchShift) {
					
				}
			}
		}

		//	forward propagation


		//	calculate loss


		//	back propagation

	}
};

dataSet mnist;
model myModel;
batch myBatch;
void buildModel(model& model) {
	model.inputLayer = {
		784
	};
	model.hiddenLayers.push_back({
		"fullconnect",
		"sigmoid",
		16,
		0.1
		});
	model.outputLayer = {
		"crossEntropy",
		"softmax",
		10,
		0.1
	};
}
void buildBatch(batch& batch) {
	batch = {
		100000,
		10,
		1
	};
}
int main()
{
	mnist.MNISTInput();
	buildModel(myModel);
	buildBatch(myBatch);
	myModel.ini();
	myModel.normalWB(0, 1);
	myModel.train(myBatch);
}
