// Writes a small GigaLearn-format checkpoint with LibTorch, as GigaLearn itself saves one: each network
// an nn::Sequential of Linear, LayerNorm (optional) and activation layers saved with torch::save, its Adam
// state with optimizer.save, and RUNNING_STATS.json. Also writes reference policy probabilities computed here,
// in C++, with GigaLearn's masking rule, for the import's parity tests.
//
// Build (Linux, PyTorch's own LibTorch): tools/gigalearn_fixture/build.sh
// Run: make_fixture <out folder> <obs size> <shared|none> <relu|leaky_relu|tanh|sigmoid> <norm|nonorm> <temperature>
#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void add_activation(torch::nn::Sequential& seq, const std::string& kind) {
	if (kind == "relu") seq->push_back(torch::nn::ReLU());
	else if (kind == "leaky_relu") seq->push_back(torch::nn::LeakyReLU());
	else if (kind == "tanh") seq->push_back(torch::nn::Tanh());
	else if (kind == "sigmoid") seq->push_back(torch::nn::Sigmoid());
	else throw std::runtime_error("unknown activation " + kind);
}

static torch::nn::Sequential network(int inputs, std::vector<int> hidden, int outputs, const std::string& act, bool norm) {
	torch::nn::Sequential seq;
	int last = inputs;
	for (int width : hidden) {
		seq->push_back(torch::nn::Linear(last, width));
		if (norm) seq->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({(int64_t)width})));
		add_activation(seq, act);
		last = width;
	}
	if (outputs > 0) seq->push_back(torch::nn::Linear(last, outputs));
	return seq;
}

static void save(torch::nn::Sequential& seq, torch::optim::Adam& optim, const fs::path& dir, const std::string& name) {
	std::ofstream out(dir / (name + ".lt"), std::ios::binary);
	torch::save(seq, out);
	torch::serialize::OutputArchive archive;
	optim.save(archive);
	archive.save_to((dir / (name + "_OPTIM.lt")).string());
}

int main(int argc, char** argv) {
	if (argc != 7) {
		std::cerr << "usage: make_fixture <out> <obs size> <shared|none> <activation> <norm|nonorm> <temperature>\n";
		return 2;
	}
	fs::path root = argv[1];
	int obs = std::stoi(argv[2]);
	bool shared = std::string(argv[3]) == "shared";
	std::string act = argv[4];
	bool norm = std::string(argv[5]) == "norm";
	float temperature = std::stof(argv[6]);
	const int actions = 90;
	torch::manual_seed(7);

	fs::path dir = root / "checkpoints" / "123456";
	fs::create_directories(dir);
	int features = obs;
	torch::nn::Sequential head;
	if (shared) {
		head = network(obs, {32}, 0, act, norm);
		features = 32;
	}
	auto policy = network(features, {32, 24}, actions, act, norm);
	auto critic = network(features, {24}, 1, act, norm);

	auto policy_probs = [&](torch::Tensor x, torch::Tensor mask) {
		if (shared) x = head->forward(x);
		auto logits = policy->forward(x) / temperature;
		return torch::softmax(logits + -1e10f * mask.logical_not(), -1).clamp(1e-11f, 1);
	};
	auto value = [&](torch::Tensor x) {
		if (shared) x = head->forward(x);
		return critic->forward(x).flatten();
	};

	// A few Adam steps, so the saved optimizers hold moments and step counts.
	std::vector<torch::Tensor> all;
	for (auto& p : policy->parameters()) all.push_back(p);
	torch::optim::Adam policy_optim(policy->parameters(), 1e-3);
	torch::optim::Adam critic_optim(critic->parameters(), 1e-3);
	torch::optim::Adam head_optim(shared ? head->parameters() : std::vector<torch::Tensor>{}, 1e-3);
	for (int step = 0; step < 3; step++) {
		auto x = torch::randn({64, obs});
		auto mask = torch::rand({64, actions}) > 0.3;
		auto loss = -policy_probs(x, mask).log().mean() + value(x).pow(2).mean();
		policy_optim.zero_grad(); critic_optim.zero_grad(); head_optim.zero_grad();
		loss.backward();
		policy_optim.step(); critic_optim.step();
		if (shared) head_optim.step();
	}
	if (shared) save(head, head_optim, dir, "SHARED_HEAD");
	save(policy, policy_optim, dir, "POLICY");
	save(critic, critic_optim, dir, "CRITIC");
	{
		std::ofstream stats(dir / "RUNNING_STATS.json");
		stats << std::setprecision(17)
			<< "{\n    \"return_stat\": {\"count\": 4500, \"mean\": 0.25, \"var\": 9000.0},\n"
			<< "    \"run_id\": \"fixture\",\n    \"total_iterations\": 42,\n    \"total_timesteps\": 123456\n}\n";
	}

	// Reference outputs on fixed inputs: obs, mask, probabilities, values.
	torch::NoGradGuard guard;
	auto x = torch::randn({16, obs}) * 2;
	auto mask = torch::rand({16, actions}) > 0.4;
	mask.index_put_({torch::indexing::Slice(), 0}, true);
	auto probs = policy_probs(x, mask);
	auto values = value(x);
	std::ofstream ref(root / "reference.txt");
	ref << std::setprecision(9);
	auto write = [&](const char* name, torch::Tensor t) {
		t = t.to(torch::kFloat).contiguous().flatten();
		ref << name << ' ' << t.numel();
		for (int64_t i = 0; i < t.numel(); i++) ref << ' ' << t[i].item<float>();
		ref << '\n';
	};
	write("obs", x);
	write("mask", mask);
	write("probs", probs);
	write("values", values);
	std::cout << "wrote " << root << "\n";
	return 0;
}
