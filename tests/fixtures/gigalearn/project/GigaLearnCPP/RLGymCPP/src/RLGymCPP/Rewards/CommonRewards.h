#pragma once
// Stands in for a project's copy of GigaLearn's CommonRewards.h with a reward added to it (the import tests).
#include "Reward.h"

namespace RLGC {
	class AddedInGigaLearnReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			return player.boost > 50;
		}
	};
	class KickoffProximityReward : public Reward { //zerosum for 2 agents (1v1), good for 1v1, dont try other modes
	public:
		KickoffProximityReward() : Reward() {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (state.ball.vel.Length() < 1.f) {
				float playerDistToBall = (player.pos - state.ball.pos).Length();

				std::vector<float> opponentDistances;
				for (auto& p : state.players) {
					if (p.team != player.team) {
						float opponentDistToBall = (p.pos - state.ball.pos).Length();
						opponentDistances.push_back(opponentDistToBall);
					}
				}

				if (opponentDistances.empty()) {
					return 0.f;
				}

				if (playerDistToBall < *std::min_element(opponentDistances.begin(), opponentDistances.end())) {
					return 1.0f;
				}
				else {
					return -1.0f;
				}
			}

			return 0.0f;
		}
	};



	class SpeedTowardsOpponentReward : public Reward {
	public:
		float scale;

		SpeedTowardsOpponentReward(float scale = 1.0f) : scale(scale) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			// find the single opponent
			const Player* opponent = nullptr;
			for (auto& p : state.players) {
				if (p.team != player.team) {
					opponent = &p;
					break;
				}
			}

			if (!opponent)
				return 0.0f;

			// direction vector from player to opponent
			Vec dirToOpponent = (opponent->pos - player.pos).Normalized();

			// normalized player velocity
			Vec normVel = player.vel / CommonValues::CAR_MAX_SPEED;

			// dot product gives how much velocity points toward opponent
			float alignment = dirToOpponent.Dot(normVel);

			// clamp between -1 and 1, scale for tuning reward weight
			return RS_CLAMP(alignment, -1.f, 1.f) * scale;
		}
	};

}
// KickoffProximityReward and SpeedTowardsOpponentReward above are copied from community GigaLearn bots'
// rewards, whose author let them be used in NitroLearn (CREDITS.md, LICENSES/GigaLearn-notice.md).
