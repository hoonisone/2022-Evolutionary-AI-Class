#include <vector>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <tuple>
#include <random>

using namespace std;

struct GenerationState {
	float maxFitness;
	float minFitness;
	float avgFitness;

	GenerationState(float maxFitness, float minFitness, float avgFitness) {
		this->maxFitness = maxFitness;
		this->minFitness = minFitness;
		this->avgFitness = avgFitness;
	}
};

//vector<pair<float, float>> LoadData() {
//	vector<pair<float, float>> data;
//	string line;
//	ifstream file("Data(0-1Knapsack).txt"); // example.txt ������ ����. ������ ����. 
//	float idx, weight, profit;
//
//	if (file.is_open()) {
//		for (int i = 0; i < 5; i++) {
//			getline(file, line); // 5�� ������
//		}
//		while (getline(file, line)) {
//			file >> idx >> weight >> profit;
//			data.push_back(make_pair(weight, profit));
//		}
//		file.close(); // ������ ������ �ݴ´�. 
//	}
//	else {
//		cout << "Unable to open file";
//	}
//	return data;
//}

class EvolutionSearch {
public:
	int overlapNum;
	int sampleNum;
	float mutationProbability;
	float crossOverProbability;


	EvolutionSearch(int sampleNum, float mutationProbability, float crossOverProbability, int overlapNum) :
		sampleNum(sampleNum),
		mutationProbability(mutationProbability),
		crossOverProbability(crossOverProbability),
		overlapNum(overlapNum) {}

	void Init() {
		population = MakeNewPopulation(sampleNum);
		AddCurState(); ///// ���� fitness ��� �Ұ�
	}

	void NextGeneration() {
		/* ���� ����� ��ȭ
		*/

		//cout << "Selection" << endl;

		vector<vector<bool>> eliteList =  GetTopNIndividuals(population, overlapNum);
		//cout << Fitness(eliteList[0], population);
		//cout << Fitness(eliteList[1], population);

		Selection(population);

		//cout << "SimpleRandomCrossOver" << endl;
		CrossOver(population, crossOverProbability); // cross point = 3

		//cout << "Mutation" << endl;
		Mutation(population, mutationProbability); // mutation probability = 0.01

		for (int i = 0; i < eliteList.size(); i++)
		{
			population[i] = eliteList[i];
		}

		// State ����
		AddCurState();

		generation++;
	}


	vector<vector<bool>> GetTopNIndividuals(vector<vector<bool>> population, int n) {
		vector<vector<bool>> topIndivisuals;
		vector<vector<bool>> sortedPupulation = SortByFitness(population);
		for (int i = 0; i < n; i++)
		{
			topIndivisuals.push_back(sortedPupulation[i]);
		}
		return topIndivisuals;
	}
	vector<vector<bool>> SortByFitness(vector<vector<bool>> population) {
		vector<float> fitnessList = GetFitnessList(population);
		vector<pair<float, int>> idx_fitness;
		for (int i = 0; i < fitnessList.size(); i++)
		{
			idx_fitness.push_back(make_pair(fitnessList[i], i));
		}
		sort(idx_fitness.begin(), idx_fitness.end(), greater<>());
		vector<vector<bool>> sortedPopultion;
		for (int i = 0; i < idx_fitness.size(); i++)
		{
			sortedPopultion.push_back(population[idx_fitness[i].second]);
		}
		return sortedPopultion;
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
		cout << "  * Min: " << get<1>(statistic) << endl;
		cout << "  * Avg: " << get<2>(statistic) << endl;
	}

	void PrintState2() {
		tuple<float, float, float, int, int> statistic = FitnessStatistic(population);
		cout << "Generation[" << generation << "]: " << "Max: " << get<0>(statistic) << ", Min : " << get<1>(statistic) << ", Avg: " << get<2>(statistic) << endl;

	}

	// State
	vector<float> GetMaxFitnessList() {
		vector<float> values;
		for (int i = 0; i < stateList.size(); i++)
		{
			values.push_back(stateList[i].maxFitness);
		}
		return values;
	}
	vector<float> GetMinFitnessList() {
		vector<float> values;
		for (int i = 0; i < stateList.size(); i++)
		{
			values.push_back(stateList[i].minFitness);
		}
		return values;
	}
	vector<float> GetAvgFitnessList() {
		vector<float> values;
		for (int i = 0; i < stateList.size(); i++)
		{
			values.push_back(stateList[i].avgFitness);
		}
		return values;
	}



	void StoreState(string path) {
		ofstream file(path); // example.txt ������ ����. ������ ����. 
		if (file.is_open()) {
			for (size_t i = 0; i < stateList.size(); i++)
			{
				GenerationState state = stateList[i];
				file << state.maxFitness << " " << state.minFitness << " " << state.avgFitness << endl;
			}
			file.close(); // ������ ������ �ݴ´�. 
		}
		else {
			cout << "Unable to open file";
		}
	}

	void StorePopulationAndFitness(string path) {

		ofstream file(path); // example.txt ������ ����. ������ ����. 
		if (file.is_open()) {
			vector<float> maxFitnessList = GetMaxFitnessList();
			for (int i = 0; i < population.size(); i++)
			{
				vector<bool> sample = population[i];
				file << ToString(sample) << "," << Fitness(sample, population) << endl;
			}
			file.close(); // ������ ������ �ݴ´�. 
		}
		else {
			cout << "Unable to open file";
		}
	}

protected:
	int generation; // ������ ���� Ƚ��
	vector<vector<bool>> population; // ���� population

	// State
	vector<GenerationState> stateList; // ���� �� state
	void AddCurState() {
		tuple<float, float, float, int, int> statistic = FitnessStatistic(population);
		float maxFitness = get<0>(statistic);
		float avgFitness = get<1>(statistic);
		float minFitness = get<2>(statistic);

		stateList.push_back(GenerationState(maxFitness, minFitness, avgFitness));
	}


	// Utility
	static vector<int> GetIndexList(int size) {
		vector<int> list;
		for (int i = 0; i < size; i++)
		{
			list.push_back(i);
		}
		return list;
	}
	static vector<int> GetRandomPoint(int size, int pointNum) {
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
	static void Swap(bool&& a, bool&& b) {
		int temp = a;
		a = b;
		b = temp;
	}
	static float GetRandProbability() {
		return rand() / (double)RAND_MAX; // ���� Ȯ�� ���� (0 ~ 1)
	}
	static string ToString(vector<bool> sample) {
		string s = "";
		for (int i = 0; i < sample.size(); i++)
		{
			s += sample[i] ? "1" : "0";
		}
		return s;
	}

	// Init
	//virtual vector<bool> MakeNBitSample() = 0;


	static vector<bool> MakeNBitSample(int size) {
		vector<bool> sample = vector<bool>(size);
		for (int i = 0; i < sample.size(); i++)
		{
			sample[i] = (GetRandProbability() <= 0.5); // i��° item ���� ���� 
		}
		return sample;
	}

	virtual vector<bool> MakeIndividualHandler() = 0;

	vector<vector<bool>> MakeNewPopulation(int sampleNum) {
		/* ���� �ʱ�ȭ �� Population �ϳ��� ����
		* size : sample�� ũ��
		* num : sample�� ����
		*/
		vector<vector<bool>> samples;
		for (int i = 0; i < sampleNum; i++)
		{
			samples.push_back(MakeIndividualHandler());
		}
		return samples;
	}

	// fitness
	tuple<float, float, float, int, int> FitnessStatistic(vector<vector<bool>>& samples) {
		// ���� fitness�� max, min, avg, maxIdx, minIdx ��ȯ
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
		return make_tuple(max, min, avg, maxIdx, minIdx);
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
			fitnessList.push_back(Fitness(samples[i], samples));
		}
		return fitnessList;
	}
	virtual float Fitness(vector<bool>& sample, vector<vector<bool>>& population) = 0;


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
				result.push_back(make_pair(Fitness(sample, samples), participantIdx));
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

	// SimpleRandomCrossOver
	static  void SimpleRandomCrossOver(vector<bool>& s1, vector<bool>& s2, int pointNum) {
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


	template <typename T> vector<pair<T, T>> MakePair(vector<T> list) {
		/* list�� ���ҵ��� �� ���� ���� ����Ʈ�� ��ȯ.*/
		vector<pair<T, T>> pairList; // �� ���� �� ����Ʈ ����

		for (int i = 0; i < list.size() / 2; i++)
		{
			// �������� �ϳ� �̰� ����1
			int idx1 = rand() % list.size(); // �ε��� ����
			T val1 = list[idx1]; // �� ����
			list.erase(list.begin() + idx1); // ����Ʈ���� ����

			// �������� �ϳ� �̰� ����2
			int idx2 = rand() % list.size();
			T val2 = list[idx2];
			list.erase(list.begin() + idx2);

			// �� �߰�
			pairList.push_back(make_pair(val1, val2));
		}
		return pairList;
	}

	void CrossOver(vector<vector<bool>>& samples, float crossOverProbability) {
		vector<int> idxList = GetIndexList(samples.size()); // population�� sample �� idx list ����
		vector<pair<int, int>> idxPairList = MakePair(idxList); // 2 ���� �� ����

		for (int i = 0; i < idxPairList.size(); i++)
		{
			if (GetRandProbability() <= crossOverProbability) { // crossOverProbability�� ���� ����
				// idx ����
				int idx1 = idxPairList[i].first;
				int idx2 = idxPairList[i].second;

				// sample ����
				vector<bool>& sample1 = samples[idx1];
				vector<bool>& sample2 = samples[idx2];

				// crossover ����
				CrossOverHandler(sample1, sample2);
			}
		}
	}

	virtual void CrossOverHandler(vector<bool>& s1, vector<bool>& s2) = 0;

	template <typename T> vector<T> RandomChoose(vector<T> list, int num) {
		/* list���� �������� num���� ���Ҹ� �̾� ����Ʈ�� ��ȯ*/

		vector<T> selectedList; // �� ���õ� ���� ����Ʈ ����

		for (int i = 0; i < num; i++)
		{
			int point = rand() % list.size(); // ������ �ε��� ����
			selectedList.push_back(list[point]); // ���� �߰�
			list.erase(list.begin() + point); // ���� ����
		}
		return selectedList;
	}
};

class FourMaxProblemSearch : public EvolutionSearch {
public:
	FourMaxProblemSearch(int populationSIze, float mutationProbability, float crossOverProbability, int overlapNum) :
		EvolutionSearch(populationSIze, mutationProbability, crossOverProbability, overlapNum) {}

private:
	int individualSize = 50;
	int threshold = 20;
	// EvolutionSearch��(��) ���� ��ӵ�
	vector<bool> MakeIndividualHandler() override
	{
		return MakeNBitSample(50);
	}
	
	virtual void Selection(vector<vector<bool>>& samples) override
	{
		TournamentSelection(samples, 5);
	}

	void CrossOverHandler(vector<bool>& s1, vector<bool>& s2) override
	{
		
		for (int i = 25; i < 50; i++)
		{
			bool temp = s1[i];
			s1[i] = s2[i];
			s2[i] = temp;
		}
	}

	float Fitness(vector<bool>& sample, vector<vector<bool>>& population) override
	{
		vector<bool> left, right;
		left.assign(sample.begin(), sample.begin() + 25);
		right.assign(sample.begin() + 25, sample.begin() + 50);
		float penalty = DistancePenalty(sample, population, threshold);
		//penalty = (penalty == 0) ? (0.1f) : penalty; // zero divide ����
		return (f(left) + f(right))/ (1 + penalty);
	}

	static float DistancePenalty(vector<bool>& sample, vector<vector<bool>> & population, int threshold) {

		float penalty = 0;

		for (int i = 0; i < population.size(); i++)
		{
			float distance = HammingDistance(sample, population[i]);
			
			penalty += (distance < threshold) ? 1 - (distance / threshold) : 0;
		}
		return penalty/population.size();
	}

	static float HammingDistance(vector<bool>& sample1, vector<bool>& sample2) {
		int distance = 0;

		for (int i = 0; i < sample1.size(); i++)
		{
			distance += (sample1[i] != sample2[i]) ? 1 : 0;
		}
		return distance;
	}

	static float u(vector<bool> sample) {
		float fitness = 0;
		for (int i = 0; i < sample.size(); i++)
		{
			fitness += (sample[i]) ? 1 : 0;
		}
		return fitness;
	}
	static float u_(vector<bool> sample) {
		float fitness = 0;
		for (int i = 0; i < sample.size(); i++)
		{
			fitness += (sample[i]) ? 0 : 1;
		}
		return fitness;
	}
	static float f(vector<bool> sample) {
		return max(u(sample), u_(sample));
	}
};

int main(void) {
	int seed = 0;
	int generation = 30; // 50���� ���� ����
	
	srand(seed);
	FourMaxProblemSearch evolution = FourMaxProblemSearch(100, 0.01, 0.5, 20);
	evolution.Init();

	evolution.PrintState2();
	for (int i = 0; i < generation; i++)
	{
		evolution.NextGeneration();
		evolution.PrintState2();
	}
	evolution.StorePopulationAndFitness("fourmax.txt");
}



