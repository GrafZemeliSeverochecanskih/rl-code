import numpy as np
from util import create_mdp
from util import create_policy_1, create_policy_2
from model_free_prediction import TDEvaluator
from model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator
from model_free_prediction.monte_carlo_evaluator import MCEvaluator


def main() -> None:
    np.random.seed(42)
    np.set_printoptions(precision=3, floatmode="fixed")
    env = create_mdp()
    policy_1 = create_policy_1()
    policy_2 = create_policy_2()

    n_episodes = int(1e3)
    td_evaluator = TDEvaluator(env, alpha=0.1)
    value_func1 = td_evaluator.evaluate(policy_1, n_episodes)
    value_func2 = td_evaluator.evaluate(policy_2, n_episodes)
    print(f"TD(0) value function for policy 1: {value_func1}")
    print(f"TD(0) value function for policy 2: {value_func2}")

    td_lamb_evaluator = TDLambdaEvaluator(env, alpha=0.1, lambd=0.5)
    value_func1 = td_lamb_evaluator.evaluate(policy_1, n_episodes)
    value_func2 = td_lamb_evaluator.evaluate(policy_2, n_episodes)
    print(f"TD(0.5) value function for policy 1: {value_func1}")
    print(f"TD(0.5) value function for policy 2: {value_func2}")

    mc_evaluator = MCEvaluator(env)
    value_func1 = mc_evaluator.evaluate(policy_1, n_episodes)
    value_func2 = mc_evaluator.evaluate(policy_2, n_episodes)
    print(f"MC value function for policy 1: {value_func1}")
    print(f"MC value function for policy 2: {value_func2}")

if __name__=="__main__":
    main()