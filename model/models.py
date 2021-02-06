# common library
import pandas as pd
import numpy as np
import time
import gym
from tqdm import trange

# RL models from stable-baselines
from stable_baselines import GAIL, SAC
from stable_baselines import ACER
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import TD3

from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
    AdaptiveParamNoiseSpec,
)
from stable_baselines.common.vec_env import DummyVecEnv
from preprocessing.preprocessors import *
from config import config

# customized env
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade

import logging

logger = logging.getLogger(__name__)


def train_A2C(env_train, iteration: int, timesteps=30000) -> A2C:
    """A2C model"""
    logger.info("======A2C Training========")
    model_name = f"A2C_30k_dow_{iteration}"
    start = time.time()
    model = A2C("MlpPolicy", env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (A2C): ", (end - start) / 60, " minutes")
    return model


def train_ACER(env_train, iteration: int, timesteps=25000) -> ACER:

    model_name = f"ACER_25k_dow_{iteration}"

    start = time.time()
    model = ACER("MlpPolicy", env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (A2C): ", (end - start) / 60, " minutes")
    return model


def train_DDPG(env_train, iteration: int, timesteps=10000) -> DDPG:
    """DDPG model"""

    logger.info("======DDPG Training========")
    # add the noise objects for DDPG
    model_name = "DDPG_10k_dow_{}".format(iteration)
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions)
    )

    start = time.time()
    model = DDPG(
        "MlpPolicy", env_train, param_noise=param_noise, action_noise=action_noise
    )
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (DDPG): ", (end - start) / 60, " minutes")
    return model


def train_PPO(env_train, iteration: int, timesteps=100000) -> PPO2:
    """PPO model"""
    logger.info("======PPO Training========")

    model_name = "PPO_100k_dow_{}".format(iteration)
    start = time.time()
    model = PPO2("MlpPolicy", env_train, ent_coef=0.005, nminibatches=8)
    # model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (PPO): ", (end - start) / 60, " minutes")
    return model


def train_GAIL(env_train, model_name, timesteps=1000):
    """GAIL Model"""
    # from stable_baselines.gail import ExportDataset, generate_expert_traj
    start = time.time()
    # generate expert trajectories
    model = SAC("MLpPolicy", env_train, verbose=1)
    generate_expert_traj(model, "expert_model_gail", n_timesteps=100, n_episodes=10)

    # Load dataset
    dataset = ExpertDataset(
        expert_path="expert_model_gail.npz", traj_limitation=10, verbose=1
    )
    model = GAIL("MLpPolicy", env_train, dataset, verbose=1)

    model.learn(total_timesteps=1000)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (PPO): ", (end - start) / 60, " minutes")
    return model


def DRL_prediction(
    df,
    model,
    name,
    last_state,
    iter_num: int,
    unique_trade_date,
    rebalance_window,
    turbulence_threshold,
    initial,
):
    ### make a prediction based on trained model###

    ## trading env
    trade_data = data_split(
        df,
        start=unique_trade_date[iter_num - rebalance_window],
        end=unique_trade_date[iter_num],
    )
    env_trade = DummyVecEnv(
        [
            lambda: StockEnvTrade(
                trade_data,
                turbulence_threshold=turbulence_threshold,
                initial=initial,
                previous_state=last_state,
                model_name=name,
                iteration=iter_num,
            )
        ]
    )
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({"last_state": last_state})
    df_last_state.to_csv(
        "results/last_state_{}_{}.csv".format(name, iter_num), index=False
    )
    return last_state


def DRL_validation(model, test_data, test_env: DummyVecEnv, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv(
        "results/account_value_validation_{}.csv".format(iteration), index_col=0
    )
    df_total_value.columns = ["account_value_train"]
    df_total_value["daily_return"] = df_total_value.pct_change(1)
    sharpe = (
        (4 ** 0.5)
        * df_total_value["daily_return"].mean()
        / df_total_value["daily_return"].std()
    )
    return sharpe


def run_ensemble_strategy(
    df: pd.DataFrame,
    unique_trade_date: pd.DataFrame,
    rebalance_window: int,
    validation_window: int,
) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    print("============Start Ensemble Strategy============")
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_ensemble = []

    # based on the analysis of the in-sample data
    # turbulence_threshold = 140
    insample_turbulence = df[(df.datadate < 20151000) & (df.datadate >= 20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=["datadate"])
    insample_turbulence_threshold = np.quantile(
        insample_turbulence.turbulence.values, 0.90
    )

    start = time.time()
    progress = trange(
        rebalance_window + validation_window, len(unique_trade_date), rebalance_window
    )

    models_to_train = ((train_PPO, "PPO"),)

    for i in progress:
        print("============================================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        end_date_index = df.index[
            df["datadate"]
            == unique_trade_date[i - rebalance_window - validation_window]
        ].to_list()[-1]
        start_date_index = end_date_index - validation_window * 30 + 1

        historical_turbulence = df.iloc[start_date_index : (end_date_index + 1), :]
        # historical_turbulence = df[(df.datadate<unique_trade_date[i - rebalance_window - validation_window]) & (df.datadate>=(unique_trade_date[i - rebalance_window - validation_window - 63]))]

        historical_turbulence = historical_turbulence.drop_duplicates(
            subset=["datadate"]
        )

        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        print("turbulence_threshold: ", turbulence_threshold)

        ############## Environment Setup starts ##############
        ## training env
        train = data_split(
            df,
            start=20090000,
            end=unique_trade_date[i - rebalance_window - validation_window],
        )
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        ## validation env
        validation = data_split(
            df,
            start=unique_trade_date[i - rebalance_window - validation_window],
            end=unique_trade_date[i - rebalance_window],
        )
        env_val = DummyVecEnv(
            [
                lambda: StockEnvValidation(
                    validation, turbulence_threshold=turbulence_threshold, iteration=i
                )
            ]
        )
        obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print(
            "======Model training from: ",
            20090000,
            "to ",
            unique_trade_date[i - rebalance_window - validation_window],
        )
        # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
        # print("==============Model Training===========")

        models = []
        sharpes = []
        for train_func, model_name in models_to_train:
            model_a2c, sharpe_a2c = train_and_eval(
                train_func,
                env_train,
                env_val,
                i,
                obs_val,
                rebalance_window,
                unique_trade_date,
                validation,
                validation_window,
                model_name=model_name,
            )
            models.append(model_a2c)
            sharpes.append(sharpe_a2c)

        idx = sharpes.index(max(sharpes))
        model_ensemble = models[idx]
        ############## Training and Validation ends ##############

        ############## Trading starts ##############
        print(
            "======Trading from: ",
            unique_trade_date[i - rebalance_window],
            "to ",
            unique_trade_date[i],
        )
        # print("Used Model: ", model_ensemble)
        last_state_ensemble = DRL_prediction(
            df=df,
            model=model_ensemble,
            name="ensemble",
            last_state=last_state_ensemble,
            iter_num=i,
            unique_trade_date=unique_trade_date,
            rebalance_window=rebalance_window,
            turbulence_threshold=turbulence_threshold,
            initial=initial,
        )
        # print("============Trading Done============")
        ############## Trading ends ##############

    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")


def train_and_eval(
    train_func,
    env_train,
    env_val,
    i,
    obs_val,
    rebalance_window,
    unique_trade_date,
    validation,
    validation_window,
    model_name: str,
):
    model_ddpg = train_func(env_train, iteration=i)
    logger.info(
        f"======{model_name} Validation from: "
        f"{unique_trade_date[i - rebalance_window - validation_window]} to "
        f"{unique_trade_date[i - rebalance_window]}"
    )
    DRL_validation(
        model=model_ddpg, test_data=validation, test_env=env_val, test_obs=obs_val
    )
    sharpe_ddpg = get_validation_sharpe(i)

    return model_ddpg, sharpe_ddpg
