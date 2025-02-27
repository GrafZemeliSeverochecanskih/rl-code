from rl_cliff.agents.tabularagent import TabularAgent


class RandomAgent(TabularAgent):
    def update(self, transition: tuple) -> None:
        pass

    def behavior_policy(self, state) -> int:
        return self.env_action_space.sample()

    def target_policy(self, state) -> int:
        return self.behavior_policy(state)
