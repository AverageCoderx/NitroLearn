// A small GigaLearn project for NitroLearn's import tests: every built-in reward form, a custom one,
// commented-out lines, and settings written the ways people write them.
#include <GigaLearnCPP/Learner.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include "MyRewards.h"

using namespace GGL;
using namespace RLGC;

EnvCreateResult MakeEnv(int index) {
	std::vector<WeightedReward> rewards = {
		{ new AirReward(), 0.3f },
		{ new FaceBallReward(), 0.5f },
		{ new VelocityPlayerToBallReward(), 3.f },
		{ new StrongTouchReward(25, 90), 40 },
		{ new ZeroSumReward(new VelocityBallToGoalReward(), 1), 2.5f },
		{ new VelocityBallToGoalReward(true), -0.5f },
		{ new PickupBoostReward(), 8.f },
		{ new SaveBoostReward(0.75f), 0.1f },
		{ new ZeroSumReward(new BumpReward(), 0.5f, 0.8f), 15 },
		{ new ZeroSumReward(new DemoReward(), 0.5f), 60 },
		{ new BumpedPenalty(), 4 },
		{ new DemoedPenalty(), 10 },
		{ new TouchAccelReward(), 12 },
		{ new VelocityReward(), 0.05f },
		{ new SpeedReward(), 0.02f },
		{ new WavedashReward(), 3 },
		{ new TouchBallReward(), 1 },
		// { new ShotReward(), 20 },
		{ new SaveReward(), 30 },
		{ new MyAerialReward(1500.f), 2 },
		{ new KickoffProximityReward(), 5 },
		{ new SpeedTowardsOpponentReward(1.5f), 2 },
		{ new GoalReward(-0.8f), 200 }
	};

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(12),
		new GoalScoreCondition()
	};

	int playersPerTeam = 1;
	auto arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < playersPerTeam; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	result.obsBuilder = new AdvancedObs();
	result.stateSetter = new CombinedState({ { new KickoffState(), 0.7f }, { new RandomState(true, true, false), 0.3f } });
	result.terminalConditions = terminalConditions;
	result.rewards = rewards;
	result.arena = arena;
	return result;
}

int main(int argc, char* argv[]) {
	RocketSim::Init("collision_meshes");
	LearnerConfig cfg = {};
	cfg.deviceType = LearnerDeviceType::GPU_CUDA;
	cfg.tickSkip = 8;
	cfg.actionDelay = cfg.tickSkip - 1;
	cfg.numGames = 512;
	cfg.randomSeed = 77;

	int tsPerItr = 100'000;
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = tsPerItr / 2;
	cfg.ppo.miniBatchSize = 25'000;
	cfg.ppo.epochs = 2;
	cfg.ppo.entropyScale = 0.03f;
	cfg.ppo.gaeGamma = 0.995;
	cfg.ppo.policyLR = 2e-4;
	cfg.ppo.criticLR = 3e-4;
	cfg.ppo.policyTemperature = 1;

	cfg.ppo.sharedHead.layerSizes = { 32 };
	cfg.ppo.policy.layerSizes = { 32, 24 };
	cfg.ppo.critic.layerSizes = { 24 };

	auto activation = ModelActivationType::RELU;
	cfg.ppo.policy.activationType = activation;
	cfg.ppo.critic.activationType = activation;
	cfg.ppo.sharedHead.activationType = activation;
	bool addLayerNorm = true;
	cfg.ppo.policy.addLayerNorm = addLayerNorm;
	cfg.ppo.critic.addLayerNorm = addLayerNorm;
	cfg.ppo.sharedHead.addLayerNorm = addLayerNorm;
	cfg.savePolicyVersions = true;
	cfg.trainAgainstOldVersions = true;

	Learner* learner = new Learner(MakeEnv, cfg);
	learner->Start();
	return EXIT_SUCCESS;
}
