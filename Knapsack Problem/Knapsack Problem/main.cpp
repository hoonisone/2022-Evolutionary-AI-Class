#include <vector>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <tuple>
using namespace std;


vector<pair<float, float>> LoadData() {
	vector<pair<float, float>> data;
	string line;
	ifstream file("Data(0-1Knapsack).txt"); // example.txt ������ ����. ������ ����. 
	float idx, weight, profit;

	if (file.is_open()) {
		for (int i = 0; i < 5; i++) {
			getline(file, line); // 5�� ������
		}
		while (getline(file, line)) {
			file >> idx >> weight >> profit;
			data.push_back(make_pair(weight, profit));
		}
		file.close(); // ������ ������ �ݴ´�. 
	}
	else {
		cout << "Unable to open file";
	}
	return data;
}

class EvolutionSearch {
public:
	int crossPoint;
	float mutationProbability;

	EvolutionSearch(int sampleNum, int crossPoint = 1, float mutationProbability = 0.01) {
		population = MakeNewSamples(sampleNum);
		this->crossPoint = crossPoint;
		this->mutationProbability = mutationProbability;

	}
	void NextGeneration() {
		//cout << "Selection" << endl;
		Selection(population);

		//cout << "CrossOver" << endl;
		CrossOver(population, crossPoint); // cross point = 3

		//cout << "Mutation" << endl;
		Mutation(population, mutationProbability); // mutation probability = 0.01
		generation++;
	}
	void PrintPopulation() {
		Print(population);
	}
	void PrintFitness() {
		Print(GetFitnessList(population));
	}

	void PrintAccFitness() {
		Print(GetAccFitnessList(population));
	}

	void PrintFitnessRatio() {
		Print(GetAccFitnessRatioList(population));
	}

	void PrintState() {
		tuple<float, float, float, int, int> statistic = FitnessStatistic(population);
		cout << "Generation[" << generation << "]" << endl;
		cout << "  * Max: " << get<0>(statistic) << endl;
		cout << "  * Avg: " << get<1>(statistic) << endl;
		cout << "  * Min: " << get<2>(statistic) << endl;
	}

	void PrintState2() {
		tuple<float, float, float, int, int> statistic = FitnessStatistic(population);
		cout << "Generation[" << generation << "]: " << "Max: " << get<0>(statistic) << ", Avg: " << get<1>(statistic) << ", Min : " << get<2>(statistic) << endl;

	}

protected:
	int generation; // ������ ���� Ƚ��
	vector<vector<bool>> population;

	// Utility
	vector<int> GetIndexList(int size) {
		vector<int> list;
		for (int i = 0; i < size; i++)
		{
			list.push_back(i);
		}
		return list;
	}
	vector<int> GetRandomPoint(int size, int pointNum) {
		/* size ������ vector���� ������ index�� pointNum��ŭ �̾Ƽ� ��ȯ
		* �� �� �ε����� ����
		*/
		vector<int> points;
		vector<int> list = GetIndexList(size);
		for (int i = 0; i < pointNum; i++)
		{
			int point = 1 + rand() % (size - 1); // �� ���� ����
			points.push_back(list[point]);
			list.erase(list.begin() + point);
			size--;
		}
		sort(points.begin(), points.end());
		return points;
	}
	void Swap(bool&& a, bool&& b) {
		int temp = a;
		a = b;
		b = temp;
	}
	float GetRandProbability() {
		return rand() / (double)RAND_MAX; // ���� Ȯ�� ���� (0 ~ 1)
	}

	// Init
	//virtual vector<bool> MakeNewSample1() = 0;
	virtual vector<bool> MakeNewSample1() {
		vector<bool> sample = vector<bool>(10000);
		for (int i = 0; i < sample.size(); i++)
		{
			sample[i] = (GetRandProbability() <= 0.5); // i��° item ���� ���� 
		}
		return sample;
	}

	vector<vector<bool>> MakeNewSamples(int sampleNum) {
		/* ���� �ʱ�ȭ �� Population �ϳ��� ����
		* size : sample�� ũ��
		* num : sample�� ����
		*/
		vector<vector<bool>> samples;
		for (int i = 0; i < sampleNum; i++)
		{
			samples.push_back(MakeNewSample1());
		}
		return samples;
	}

	// fitness
	tuple<float, float, float, int, int> FitnessStatistic(vector<vector<bool>>& samples) {
		// ���� fitness�� max, avg, min, maxIdx, minIdx ��ȯ
		vector<float> fitnessList = GetFitnessList(samples);
		float acc = 0;
		float max = 0;
		float min = numeric_limits<float>::max();
		int maxIdx, minIdx;
		for (int i = 0; i < fitnessList.size(); i++)
		{
			float fitness = fitnessList[i];
			acc += fitness;
			if (max < fitness) {
				max = fitness;
				maxIdx = i;
			}
			if (fitness < min) {
				min = fitness;
				minIdx = i;
			}
		}
		float avg = acc / fitnessList.size();
		return make_tuple(max, avg, min, maxIdx, minIdx);
	}
	vector<float> GetAccFitnessRatioList(vector<vector<bool>>& samples) {
		vector<float> fitnessList = GetAccFitnessList(samples);
		for (int i = 0; i < fitnessList.size(); i++)
		{
			fitnessList[i] /= fitnessList[fitnessList.size() - 1];
		}
		return fitnessList;
	}
	vector<float> GetAccFitnessList(vector<vector<bool>>& samples) {
		vector<float> fitnessList = GetFitnessList(samples);
		for (int i = 0; i < fitnessList.size() - 1; i++)
		{
			fitnessList[i + 1] += fitnessList[i];
		}
		return fitnessList;
	}
	vector<float> GetFitnessList(vector<vector<bool>>& samples) {
		vector<float> fitnessList;
		for (int i = 0; i < samples.size(); i++)
		{
			fitnessList.push_back(Fitness(samples[i]));
		}
		return fitnessList;
	}
	virtual float Fitness(vector<bool>& sample) = 0;


	// Print
	void Print(vector<vector<bool>> const& samples) {
		for (int i = 0; i < samples.size(); i++)
		{
			Print(samples[i]);
		}
	}
	void Print(vector<bool> const& sample) {
		string s = "";
		for (int i = 0; i < sample.size(); i++)
		{
			s += (sample[i] ? "1" : "0");
			s += ", ";
		}
		cout << s << endl;
	}
	void Print(vector<float> values) {
		string s = "";
		for (int i = 0; i < values.size(); i++)
		{
			cout << i << ": " << values[i] << endl;
		}
	}

	// Selection
	virtual void Selection(vector<vector<bool>>& samples) = 0;

	void RouletteSelection(vector<vector<bool>>& samples) {
		vector<float> accRatio = GetAccFitnessRatioList(samples); // sample�� ���� ���� ���
		vector<vector<bool>> copy = vector<vector<bool>>(samples);
		samples.clear();

		// Roulette ������� Select
		for (int i = 0; i < copy.size(); i++)
		{
			float r = GetRandProbability();
			for (int i = 0; i < accRatio.size(); i++) {
				if (r <= accRatio[i]) {
					samples.push_back(copy[i]);
					break;
				}
			}
		}
	}
	void TournamentSelection(vector<vector<bool>>& samples, int tau) {
		vector<float> accRatio = GetAccFitnessRatioList(samples); // sample�� ���� ���� ���
		vector<vector<bool>> copy = vector<vector<bool>>(samples);
		samples.clear();

		vector<int> idxList = GetIndexList(copy.size());

		for (int i = 0; i < copy.size(); i++)
		{
			vector<int> participantIdxList = RandomChoose(idxList, tau);
			vector<pair<float, int>> result;
			for (int j = 0; j < participantIdxList.size(); j++)
			{
				int participantIdx = participantIdxList[j];
				vector<bool> sample = copy[participantIdx];
				result.push_back(make_pair(Fitness(sample), participantIdx));
			}
			sort(result.begin(), result.end());
			int winnerIdx = result[result.size() - 1].second;
			samples.push_back(copy[winnerIdx]);
		}
	}


	// Mutation
	void Mutation(vector<bool>& sample, float p = 0.01) {
		/* ���� sample�� ���Ͽ� �� bit�� Ȯ�� p�� ���� Mutation ���ش�.
		 * sample : population ���� �� ����
		 * p = mutation probability
		*/

		for (int i = 0; i < sample.size(); i++)
		{
			float r = GetRandProbability();
			if (r <= p) {
				sample[i] = !sample[i]; // Ȯ���� ���� ������
			}
		}
	}
	void Mutation(vector<vector<bool>>& samples, float p = 0.01) {
		/* Population ��ü�� ���� Mutation ����
		 * samples: population
		 * p: mutation probability
		 */
		for (int i = 0; i < samples.size(); i++)
		{
			Mutation(samples[i], p);
		}
	}

	// CrossOver
	void CrossOver(vector<bool>& s1, vector<bool>& s2, int pointNum) {
		vector<int> points = GetRandomPoint(s1.size(), pointNum);
		if (pointNum % 2 == 1) {
			points.push_back(s1.size());
		}

		for (int i = 0; i < points.size() / 2; i++)
		{
			int from = points[2 * i];
			int to = points[2 * i + 1] - 1;
			for (int idx = from; idx <= to; idx++)
			{
				bool temp = s1[idx];
				s1[idx] = s2[idx];
				s2[idx] = temp;
				//Swap(s1[idx], s2[idx]);
			}
		}
	}
	void CrossOver(vector<vector<bool>>& samples, int pointNum) {
		vector<int> idxList = GetIndexList(samples.size());
		while (0 < idxList.size()) {
			int idx1 = rand() % idxList.size();
			vector<bool>& sample1 = samples[idxList[idx1]];
			idxList.erase(idxList.begin() + idx1);
			int idx2 = rand() % idxList.size();
			vector<bool>& sample2 = samples[idxList[idx2]];
			idxList.erase(idxList.begin() + idx2);
			CrossOver(sample1, sample2, pointNum);
		}
	}

	vector<int> RandomChoose(vector<int> list, int num) {
		vector<int> points;
		for (int i = 0; i < num; i++)
		{
			int point = rand() % list.size();
			points.push_back(list[point]);
			list.erase(list.begin() + point);
		}
		return points;
	}

};

class  KnapsackProblemSearch : public EvolutionSearch {
public:
	KnapsackProblemSearch(
		vector<pair<float, float>> data, int sampleNum, int crossPoint = 1, float mutationProbability = 0.01) : EvolutionSearch(sampleNum, crossPoint, mutationProbability) {
		this->data = data;
	}


protected:


	vector<pair<float, float>> data;
	void PrintData() {
		for (int i = 0; i < data.size(); i++)
		{
			cout << data[i].first << ", " << data[i].second << endl;
		}
	}
};

class RouletteEvolution :public KnapsackProblemSearch {
public:
	RouletteEvolution(
		vector<pair<float, float>> data, int sampleNum, int crossPoint = 1, float mutationProbability = 0.01) : KnapsackProblemSearch(data, sampleNum, crossPoint, mutationProbability) {
		data = LoadData();
	}

protected:
	virtual void Selection(vector<vector<bool>>& samples) {
		RouletteSelection(samples);
	}
	virtual float Fitness(vector<bool>& sample) {
		float weight = 0;
		float fitness = 0;
		for (int i = 0; i < sample.size(); i++)
		{
			if (sample[i])
			{
				weight += data[i].first;
				fitness += data[i].second;
			}
		}
		return (weight <= 280123) ? (fitness) : 0; // �뷮 �ʰ��� 0
	}

};

class TournamentEvolution :public KnapsackProblemSearch {
public:

	TournamentEvolution(vector<pair<float, float>> data, int sampleNum, int crossPoint = 1, float mutationProbability = 0.01, int tau = 5) : KnapsackProblemSearch(data, sampleNum, crossPoint, mutationProbability) {
		this->data = data;
		this->tau = tau;
	}

protected:
	int tau;
	virtual void Selection(vector<vector<bool>>& samples) {
		TournamentSelection(samples, 5);
	}
	virtual float Fitness(vector<bool>& sample) {
		float weight = 0;
		float fitness = 0;
		for (int i = 0; i < sample.size(); i++)
		{
			if (sample[i])
			{
				weight += data[i].first;
				fitness += data[i].second;
			}
		}
		return (weight <= 280123) ? (fitness) : 0; // �뷮 �ʰ��� 0
	}

};

void GridyAlgorithm() {
	vector<pair<float, float>> data = LoadData();

	// weight �� profit�� ���� ������ ����
	vector<pair<float, int>> rank;
	for (int i = 0; i < data.size(); i++)
	{
		rank.push_back(make_pair(data[i].second / data[i].first, i));
	}
	sort(rank.begin(), rank.end());

	// ��ȿ�� ������ ���� �׸����ϰ� ����
	for (int i = 0; i < 100; i++)
	{
		int idx = rank[rank.size() - 1 - i].second;
		cout << "idx: " << idx << ", weight: " << data[idx].first << "profit: " << data[idx].second << endl;
	}

	float weight = 0;
	float profit = 0;
	int i = 0;
	while (weight < 280123) {
		int idx = rank[rank.size() - 1 - i].second;
		weight += data[idx].first;
		profit += data[idx].second;
		i++;
	}
	cout << weight << ", " << profit << endl;
} // baseline

void RouletteEvolutionTest() {
	srand(time(NULL));
	vector<pair<float, float>> data = LoadData();
	RouletteEvolution evolution = RouletteEvolution(data, 100, 3, 0.01);

	evolution.PrintState2();
	for (int i = 0; i < 100; i++)
	{
		evolution.NextGeneration();
		evolution.PrintState2();
	}
}


void TournamentEvolutionTest() {
	srand(time(NULL));
	vector<pair<float, float>> data = LoadData();
	TournamentEvolution evolution = TournamentEvolution(data, 100, 3, 0.01, 5);

	evolution.PrintState2();
	for (int i = 0; i < 100; i++)
	{
		evolution.NextGeneration();
		evolution.PrintState2();
	}
}

int main(void) {
	//GridyAlgorithm();
	//RouletteEvolutionTest();
	TournamentEvolutionTest();
}



