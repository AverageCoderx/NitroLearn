#pragma once
#include <RLGymCPP/Rewards/Reward.h>

namespace RLGC {
	// A reward of the project's own: the import lists it for porting by hand.
	class MyAerialReward : public Reward {
	public:
		float minHeight;
		MyAerialReward(float minHeight) : minHeight(minHeight) {}
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			return player.pos.z > minHeight && !player.isOnGround;
		}
	};
}
