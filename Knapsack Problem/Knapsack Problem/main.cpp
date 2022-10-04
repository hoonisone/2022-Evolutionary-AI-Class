#include <vector>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <tuple>
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

vector<pair<float, float>> LoadData() {
	vector<pair<float, float>> data;
	string line;
	ifstream file("Data(0-1Knapsack).txt"); // example.txt 파일을 연다. 없으면 생성. 
	float idx, weight, profit;

	if (file.is_open()) {
		for (int i = 0; i < 5; i++) {
			getline(file, line); // 5줄 버리기
		}
		while (getline(file, line)) {
			file >> idx >> weight >> profit;
			data.push_back(make_pair(weight, profit));
		}
		file.close(); // 열었던 파일을 닫는다. 
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
	float crossOverProbability;
	int sampleNum;

	EvolutionSearch(int sampleNum, int crossPoint = 1, float mutationProbability = 0.01, float crossOverProbability = 0.9) :
		sampleNum(sampleNum), crossPoint(crossPoint),
		mutationProbability(mutationProbability), 
		crossOverProbability(crossOverProbability) {}

	void Init() {
		population = MakeNewSamples(sampleNum);
		AddCurState(); ///// 아직 fitness 계산 불가
	}

	void NextGeneration() {
		/* 다음 세대로 진화
		*/

		//cout << "Selection" << endl;
		Selection(population);

		//cout << "CrossOver" << endl;
		CrossOver(population, crossPoint, crossOverProbability); // cross point = 3

		//cout << "Mutation" << endl;
		Mutation(population, mutationProbability); // mutation probability = 0.01
		
		// State 저장
		AddCurState();

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
		ofstream file(path); // example.txt 파일을 연다. 없으면 생성. 
		if (file.is_open()) {
			for (size_t i = 0; i < stateList.size(); i++)
			{
				GenerationState state = stateList[i];
				file << state.maxFitness << " " << state.minFitness << " " << state.avgFitness << endl;
			}
			file.close(); // 열었던 파일을 닫는다. 
		}
		else {
			cout << "Unable to open file";
		}
	}

	void StorePopulationAndFitness(string path) {
		
		ofstream file(path); // example.txt 파일을 연다. 없으면 생성. 
		if (file.is_open()) {
			vector<float> maxFitnessList = GetMaxFitnessList();
			for (int i = 0; i < population.size(); i++)
			{
				vector<bool> sample = population[i];
				file << ToString(sample) << "," << Fitness(sample) << endl;
			}
			file.close(); // 열었던 파일을 닫는다. 
		}
		else {
			cout << "Unable to open file";
		}
	}

protected:
	int generation; // 유전자 조합 횟수
	vector<vector<bool>> population; // 현재 population

	// State
	vector<GenerationState> stateList; // 세대 별 state
	void AddCurState() {
		tuple<float, float, float, int, int> statistic = FitnessStatistic(population);
		float maxFitness = get<0>(statistic);
		float avgFitness = get<1>(statistic);
		float minFitness = get<2>(statistic);

		stateList.push_back(GenerationState(maxFitness, minFitness, avgFitness));
	}


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
		/* size 길이의 vector에서 랜덤한 index를 pointNum만큼 뽑아서 반환
		* 맨 앞 인덱스는 제외
		*/
		vector<int> points;
		vector<int> list = GetIndexList(size);
		for (int i = 0; i < pointNum; i++)
		{
			int point = 1 + rand() % (size - 1); // 맨 앞은 제외
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
		return rand() / (double)RAND_MAX; // 랜덤 확률 생성 (0 ~ 1)
	}
	string ToString(vector<bool> sample) {
		string s = "";
		for (int i = 0; i < sample.size(); i++)
		{
			s += sample[i] ? "1" : "0";
		}
		return s;
	}

	// Init
	//virtual vector<bool> MakeNewSample1() = 0;
	virtual vector<bool> MakeNewSample1() {
		vector<bool> sample = vector<bool>(10000);
		for (int i = 0; i < sample.size(); i++)
		{
			sample[i] = (GetRandProbability() <= 0.5); // i번째 item 포함 여부 
		}
		return sample;
	}

	vector<vector<bool>> MakeNewSamples(int sampleNum) {
		/* 랜덤 초기화 된 Population 하나를 생성
		* size : sample의 크기
		* num : sample의 개수
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
		// 현재 fitness의 max, min, avg, maxIdx, minIdx 반환
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
		vector<float> accRatio = GetAccFitnessRatioList(samples); // sample별 누적 비율 계산
		vector<vector<bool>> copy = vector<vector<bool>>(samples);
		samples.clear();

		// Roulette 방식으로 Select
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
		vector<float> accRatio = GetAccFitnessRatioList(samples); // sample별 누적 비율 계산
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
		/* 단일 sample에 대하여 각 bit를 확률 p에 따라 Mutation 해준다.
		 * sample : population 내의 한 원소
		 * p = mutation probability
		*/

		for (int i = 0; i < sample.size(); i++)
		{
			float r = GetRandProbability();
			if (r <= p) {
				sample[i] = !sample[i]; // 확률에 따라 뒤집기
			}
		}
	}
	void Mutation(vector<vector<bool>>& samples, float p = 0.01) {
		/* Population 전체에 대해 Mutation 적용
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

	template <typename T> vector<pair<T, T>> MakePair(vector<T> list) {
		/* list의 원소들을 두 개씩 묶어 리스트로 반환.*/
		vector<pair<T, T>> pairList; // 빈 원소 쌍 리스트 생성

		for (int i = 0; i < list.size()/2; i++)
		{
			// 무작위로 하나 뽑고 제거1
			int idx1 = rand() % list.size(); // 인덱스 지정
			T val1 = list[idx1]; // 값 지정
			list.erase(list.begin() + idx1); // 리스트에서 제거

			// 무작위로 하나 뽑고 제거2
			int idx2 = rand() % list.size();
			T val2 = list[idx2];
			list.erase(list.begin() + idx2);
			
			// 쌍 추가
			pairList.push_back(make_pair(val1, val2));
		}
		return pairList;
	}

	void CrossOver(vector<vector<bool>>& samples, int pointNum, float crossOverProbability) {
		vector<int> idxList = GetIndexList(samples.size()); // population의 sample 별 idx list 생성
		vector<pair<int, int>> idxPairList = MakePair(idxList); // 2 개씩 쌍 생성

		for (int i = 0; i < idxPairList.size(); i++)
		{
			if (GetRandProbability() <= crossOverProbability) { // crossOverProbability에 따라 수행
				// idx 지정
				int idx1 = idxPairList[i].first;
				int idx2 = idxPairList[i].second;

				// sample 지정
				vector<bool>& sample1 = samples[idx1];
				vector<bool>& sample2 = samples[idx2];

				// crossover 수행
				CrossOver(sample1, sample2, pointNum);
			}
		}
	}

	template <typename T> vector<T> RandomChoose(vector<T> list, int num) {
		/* list에서 무작위로 num개의 원소를 뽑아 리스트로 반환*/

		vector<T> selectedList; // 빈 선택된 원소 리스트 생성
		
		for (int i = 0; i < num; i++)
		{
			int point = rand() % list.size(); // 무작위 인덱스 지정
			selectedList.push_back(list[point]); // 원소 추가
			list.erase(list.begin() + point); // 원소 삭제
		}
		return selectedList;
	}
};

class  KnapsackProblemSearch : public EvolutionSearch {
public:
	KnapsackProblemSearch(
		vector<pair<float, float>> data, int sampleNum, int crossPoint = 1, float mutationProbability = 0.01, float crossOverProbability = 0.9) : 
			data(data), EvolutionSearch(sampleNum, crossPoint, mutationProbability, crossOverProbability) {
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
		vector<pair<float, float>> data, int sampleNum, int crossPoint = 1, float mutationProbability = 0.01, float crossOverProbability = 0.9) : KnapsackProblemSearch(data, sampleNum, crossPoint, mutationProbability, crossOverProbability) {
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
		return (weight <= 280123) ? (fitness) : 0; // 용량 초과시 0
	}

};

class TournamentEvolution :public KnapsackProblemSearch {
public:

	TournamentEvolution(vector<pair<float, float>> data, int sampleNum, int crossPoint = 1, float mutationProbability = 0.01, float crossOverProbability = 0.9, int tau = 5) : KnapsackProblemSearch(data, sampleNum, crossPoint, mutationProbability, crossOverProbability) {
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
		return (weight <= 280123) ? (fitness) : 0; // 용량 초과시 0
	}

};

void GridyAlgorithm() {
	/* Gridy한 방식으로 weight당 profit 효율이 가장 높은 것 부터 차례대로 담는다.
	* 
	*/
	vector<pair<float, float>> data = LoadData();

	// weight 당 profit이 높은 순으로 정렬
	vector<pair<float, int>> rank;
	for (int i = 0; i < data.size(); i++)
	{
		rank.push_back(make_pair(data[i].second / data[i].first, i));
	}
	sort(rank.begin(), rank.end());

	// 고효율 아이템 부터 그리디하게 선택
	for (int i = 0; i < 100; i++)
	{
		int idx = rank[rank.size() - 1 - i].second;
		cout << "idx: " << idx << ", weight: " << data[idx].first << "profit: " << data[idx].second << endl;
	}

	float weight = 0;
	float profit = 0;
	int i = 0;
	while (weight < 280123) { // threshold를 살짝 넘게 되는 건 무시..
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
	RouletteEvolution evolution = RouletteEvolution(data, 100, 3, 0.1, 0.9);
	evolution.Init();

	evolution.PrintState2();
	for (int i = 0; i < 100; i++)
	{
		evolution.NextGeneration();
		evolution.PrintState2();
	}

	evolution.StoreState("roulette history.txt");
	evolution.StorePopulationAndFitness("roulette population.txt");
}


void TournamentEvolutionTest() {
	srand(time(NULL));
	vector<pair<float, float>> data = LoadData();
	TournamentEvolution evolution = TournamentEvolution(data, 100, 2, 0.01, 0.9, 2);
	evolution.Init();

	evolution.PrintState2();
	for (int i = 0; i < 100; i++)
	{
		evolution.NextGeneration();
		evolution.PrintState2();
	}

	evolution.StoreState("tournament history.txt");
	evolution.StorePopulationAndFitness("tournament population.txt");
}

int main(void) {
	//GridyAlgorithm(); // baseline
	
	RouletteEvolutionTest(); // Roulette

	//TournamentEvolutionTest(); // Tournament
}



