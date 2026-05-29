#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fonte.py: Script principal de treinamento PPO para DroneEnv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


try:
    import gymnasium as gym
    _GYMNASIUM = True
except Exception:
    import gym
    _GYMNASIUM = False
import numpy as np
import torch
from stable_baselines3 import PPO

# --- Optional: action masking (real masks) ---
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from sb3_contrib.common.maskable.utils import get_action_masks
    _SB3_CONTRIB_OK = True
except Exception:
    MaskablePPO = None
    MaskableEvalCallback = None
    get_action_masks = None
    _SB3_CONTRIB_OK = False

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecCheckNan,
    VecNormalize,
)
from stable_baselines3.common.running_mean_std import RunningMeanStd

# ==============
# Imports locais
# ==============
from drone_env import DroneEnv
from calculaProbLink import calcula_confiabilidade_iterativa


# =========================
# Utilidades gerais
# =========================


def set_global_seeds(seed: int):
    """
    Define seeds globais para reprodutibilidade.
    Se seed < 0, escolhe aleatoriamente.
    """
    if seed < 0:
        seed = random.randint(0, 10_000_000)
        print(f"[INFO] Seed aleatória escolhida: {seed}")
    else:
        print(f"[INFO] Seed fixa usada: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # se for usar GPU:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


def as_bool(x: Any) -> bool:
    """
    Converte argumento de linha de comando em booleano.
    Aceita 0/1, true/false, yes/no, etc.
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ["1", "true", "t", "yes", "y", "sim"]:
        return True
    if s in ["0", "false", "f", "no", "n", "nao", "não"]:
        return False
    raise ValueError(f"Não consigo interpretar '{x}' como booleano.")


def cosine_schedule(start: float, end: float, progress: float) -> float:
    """
    Retorna valor interpolado por cosseno entre start e end,
    dado progress em [0,1].
    """
    if progress <= 0.0:
        return start
    if progress >= 1.0:
        return end
    # interpolação cosseno
    cos_v = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end + (start - end) * cos_v


# =========================
# Configuração de Treino
# =========================

@dataclass
class TrainConfig:
    # infra RL
    seed: int = -1
    timesteps: int = 40_000_000
    n_envs: int = 32
    mask_actions: int = 1  # 1=usa MaskablePPO (sb3-contrib)
    n_steps: int = 6144
    batch_size: int = 12288
    n_epochs: int = 20
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    ent_start: Optional[float] = None
    ent_end: Optional[float] = None
    vf_coef: float = 0.8
    max_grad_norm: float = 0.5
    clip_start: float = 0.2
    clip_end: float = 0.12
    lr_start: float = 1e-4
    lr_end: float = 7e-5
    target_kl: Optional[float] = None

    # arquitetura
    net_width: int = 256
    net_layers: int = 2
    activation: str = "tanh"          # tanh | relu
    policy_arch: Optional[str] = None # JSON ou "512,512,256,128" ou {"pi":[...],"vf":[...]}

    # avaliação e checkpoints
    eval_freq: int = 25_000
    n_eval_episodes: int = 100
    deterministic_eval: bool = True
    checkpoint_freq: int = 1_032_192
    tb_log: str = "./ppo_drone_tensorboard"
    tb_metric_freq: int = 10_000
    best_path: str = "./ppo_best_model"
    eval_log_path: str = "./ppo_eval"
    ckpt_path: str = "./ppo_checkpoints"
    final_path: str = "./ppo_final/model"
    episode_horizon: int = 2000
    use_model: str = "fresh"  # "fresh" | "continue"

    # continuidade / normalização
    reset_ret_rms: int = 0  # 1: reseta ret_rms ao carregar VecNormalize

    # observação extra (zonas)
    obs_zone_map: int = 1       # 1: inclui mapa downsample da zona na observação
    zone_map_scale: int = 4     # fator de downsample (ex: 4 -> 25x25 em grid 100)
    obs_edge_cross: int = 1     # 1: inclui matriz NxN de "link cruza zona"

    # ambiente DroneEnv
    grid_size: int = 100
    num_drones: int = 13
    variable_drones: bool = True
    min_drones: Optional[int] = None
    max_drones: Optional[int] = None
    max_out_connections: int = 10
    square_size_m: float = 20.0
    p_sem_zona: float = 0.0
    zonas_file: Optional[str] = None

    # parametrização de zonas
    zonas_count: Optional[int] = None
    zona_w: Optional[int] = None
    zona_h: Optional[int] = None
    zona_single_fixed: bool = False

    # HPO / Optuna
    hpo_optuna: int = 0
    hpo_trials: int = 60
    hpo_budget: int = 2_000_000
    hpo_pruner: str = "asha"
    hpo_seed: int = 42
    hpo_n_jobs: int = -1
    hpo_k_seeds: int = 1
    hpo_ckpt_dir: str = "./optuna_ckpt"
    hpo_study_name: str = "ppo_drone_hpo"


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser("PPO DroneEnv + Optuna")

    # treino
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--timesteps", type=int, default=40_000_000)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--mask-actions", type=int, default=1, choices=[0, 1],
                   help="1: usa MaskablePPO (sb3-contrib) + action masking real; 0: PPO padrão")
    p.add_argument("--n-steps", type=int, default=6144)
    p.add_argument("--batch-size", type=int, default=24576)
    p.add_argument("--n-epochs", type=int, default=29)
    p.add_argument("--ent-coef", type=float, default=0.015)
    p.add_argument("--ent-start", type=float, default=None)
    p.add_argument("--ent-end", type=float, default=None)
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--target_kl", type=float, default=0.02)

    # schedules
    p.add_argument("--lr-start", type=float, default=1e-4)
    p.add_argument("--lr-end", type=float, default=7e-5)
    p.add_argument("--clip-start", type=float, default=0.20)
    p.add_argument("--clip-end", type=float, default=0.10)

    # rede
    p.add_argument("--net-width", type=int, default=256)
    p.add_argument("--net-layers", type=int, default=3)
    p.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu"])
    p.add_argument("--policy-arch", type=str, default=None)

    # logística de log / checkpoints / avaliação
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--n-eval-episodes", type=int, default=200)
    p.add_argument("--deterministic-eval", type=str, default="true")
    p.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    p.add_argument("--tb-log", type=str, default="./ppo_drone_tensorboard")
    p.add_argument("--tb-metric-freq", type=int, default=10_000)
    p.add_argument("--best-path", type=str, default="./ppo_best_model")
    p.add_argument("--eval-log-path", type=str, default="./ppo_eval")
    p.add_argument("--ckpt-path", type=str, default="./ppo_checkpoints")
    p.add_argument("--final-path", type=str, default="./ppo_final/model")
    p.add_argument("--episode-horizon", type=int, default=2000)
    p.add_argument("--use-model", type=str, default="fresh", choices=["fresh", "continue"])
    p.add_argument("--reset-ret-rms", type=int, default=0, choices=[0, 1],
                   help="1: ao continuar, reseta VecNormalize.ret_rms (normalização de retorno); 0: mantém")

    # observação extra (zonas)
    p.add_argument("--obs-zone-map", type=int, default=1, choices=[0, 1],
                   help="1: inclui mapa (downsample) da zona proibida na observação")
    p.add_argument("--zone-map-scale", type=int, default=4,
                   help="downsample do mapa de zona (ex: 4 -> 25x25 em grid 100)")
    p.add_argument("--obs-edge-cross", type=int, default=1, choices=[0, 1],
                   help="1: inclui feature NxN indicando se um link cruza a zona")


    # ambiente DroneEnv
    p.add_argument("--grid-size", type=int, default=100)
    p.add_argument("--num-drones", type=int, default=13)
    p.add_argument("--max-out-connections", type=int, default=10)
    p.add_argument("--square-size-m", type=float, default=20.0)
    p.add_argument("--p-sem-zona", type=float, default=0.3)
    p.add_argument("--zonas-file", type=str, default=None)

    p.add_argument("--variable-drones", type=str, default="true")
    p.add_argument("--min-drones", type=int, default=9)
    p.add_argument("--max-drones", type=int, default=13)


    # zonas
    p.add_argument("--zonas-count", type=int, default=None)
    p.add_argument("--zona-w", type=int, default=None)
    p.add_argument("--zona-h", type=int, default=None)
    p.add_argument("--zona-single-fixed", type=int, default=0)

    # HPO
    p.add_argument("--hpo-optuna", type=int, default=0)
    p.add_argument("--hpo-trials", type=int, default=30)
    p.add_argument("--hpo-budget", type=int, default=2_000_000)
    p.add_argument("--hpo-pruner", type=str, default="asha", choices=["asha", "median"])
    p.add_argument("--hpo-seed", type=int, default=42)
    p.add_argument("--hpo-n-jobs", type=int, default=-1)
    p.add_argument("--hpo-k-seeds", type=int, default=1)
    p.add_argument("--hpo-ckpt-dir", type=str, default="./optuna_ckpt")
    p.add_argument("--hpo-study-name", type=str, default="ppo_drone_hpo")

    a = p.parse_args()
    return TrainConfig(
        seed=a.seed,
        timesteps=a.timesteps,
        n_envs=a.n_envs,
        mask_actions=a.mask_actions,
        n_steps=a.n_steps,
        batch_size=a.batch_size,
        n_epochs=a.n_epochs,
        ent_coef=a.ent_coef,
        ent_start=a.ent_start,
        ent_end=a.ent_end,
        vf_coef=a.vf_coef,
        gamma=a.gamma,
        gae_lambda=a.gae_lambda,
        target_kl=a.target_kl,
        lr_start=a.lr_start, lr_end=a.lr_end,
        clip_start=a.clip_start, clip_end=a.clip_end,
        net_width=a.net_width, net_layers=a.net_layers,
        activation=a.activation, policy_arch=a.policy_arch,
        eval_freq=a.eval_freq,
        n_eval_episodes=a.n_eval_episodes,
        deterministic_eval=as_bool(a.deterministic_eval),
        checkpoint_freq=a.checkpoint_freq,
        tb_log=a.tb_log, best_path=a.best_path, eval_log_path=a.eval_log_path,
        ckpt_path=a.ckpt_path, final_path=a.final_path,
        episode_horizon=a.episode_horizon,
        tb_metric_freq=a.tb_metric_freq,
        use_model=a.use_model,
        reset_ret_rms=a.reset_ret_rms,
        obs_zone_map=a.obs_zone_map,
        zone_map_scale=a.zone_map_scale,
        obs_edge_cross=a.obs_edge_cross,
        grid_size=a.grid_size, num_drones=a.num_drones,
        variable_drones=as_bool(a.variable_drones),
        min_drones=a.min_drones, max_drones=a.max_drones,
        max_out_connections=a.max_out_connections,
        square_size_m=a.square_size_m, p_sem_zona=a.p_sem_zona,
        zonas_file=a.zonas_file,
        zonas_count=a.zonas_count,
        zona_w=a.zona_w, zona_h=a.zona_h,
        zona_single_fixed=as_bool(a.zona_single_fixed),
        hpo_optuna=a.hpo_optuna, hpo_trials=a.hpo_trials,
        hpo_budget=a.hpo_budget, hpo_pruner=a.hpo_pruner, hpo_seed=a.hpo_seed,
        hpo_n_jobs=a.hpo_n_jobs,
        hpo_k_seeds=a.hpo_k_seeds,
        hpo_ckpt_dir=a.hpo_ckpt_dir,
        hpo_study_name=a.hpo_study_name,
    )


# =========================
# Env factories
# =========================


def _wrap_env(e: gym.Env, horizon: int) -> gym.Env:
    e = SafeEnvWrapper(e)
    e = Monitor(e)
    e = gym.wrappers.TimeLimit(e, max_episode_steps=horizon)
    return e



def _build_drone_env(
    cfg: TrainConfig,
    zonas_proibidas: Optional[List[Tuple[int, int]]],
    eval_mode: bool,
    num_drones_override: Optional[int] = None,
):
    """Cria DroneEnv passando apenas kwargs suportados (compatível entre versões)."""
    kwargs = dict(
        grid_size=cfg.grid_size,
        num_drones=(num_drones_override if num_drones_override is not None else cfg.num_drones),
        max_out_connections=cfg.max_out_connections,
        square_size_m=cfg.square_size_m,
        zonas_proibidas=zonas_proibidas,
        p_sem_zona=cfg.p_sem_zona,
        variable_drones=cfg.variable_drones,
        min_drones=cfg.min_drones,
        max_drones=cfg.max_drones,
        eval_mode=eval_mode,
        zonas_count=cfg.zonas_count,
        zona_size=(cfg.zona_w, cfg.zona_h) if (cfg.zona_w and cfg.zona_h) else None,
        zona_single_fixed=cfg.zona_single_fixed,
        # observações extra (se suportado pelo env)
        obs_zone_map=bool(cfg.obs_zone_map),
        zone_map_scale=int(cfg.zone_map_scale),
        obs_edge_cross=bool(cfg.obs_edge_cross),
    )
    sig = inspect.signature(DroneEnv.__init__)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return DroneEnv(**filtered)


def make_env_factory(
    cfg: TrainConfig,
    zonas_from_file: Optional[List[Tuple[int, int]]],
    rank: int,
):
    """
    Retorna função fábrica para SubprocVecEnv.
    """
    def _f():
        env = _build_drone_env(cfg, zonas_from_file, eval_mode=False)
        env.reset(seed=rank)
        env = _wrap_env(env, cfg.episode_horizon)
        return env

    return _f


def build_train_env(
    cfg: TrainConfig,
    zonas_from_file: Optional[List[Tuple[int, int]]],
    ckpt_vec_path: Optional[str] = None,
) -> VecNormalize:
    """
    Cria o ambiente de treino.

    Se `ckpt_vec_path` for fornecido, carrega o VecNormalize salvo nesse arquivo
    (continuação de treino a partir de checkpoint).
    Caso contrário, cria um VecNormalize novo.
    """
    venv = SubprocVecEnv([make_env_factory(cfg, zonas_from_file, i) for i in range(cfg.n_envs)])
    venv = VecCheckNan(venv, raise_exception=True)

    if ckpt_vec_path is not None and os.path.isfile(ckpt_vec_path):
        print(f"[ENV] Carregando VecNormalize de checkpoint: {ckpt_vec_path}")
        vec = VecNormalize.load(ckpt_vec_path, venv)
        # Garante consistência com o gamma do PPO desta execução
        try:
            vec.gamma = cfg.gamma
        except Exception:
            pass
        if getattr(cfg, 'reset_ret_rms', 0):
            # Reseta apenas a normalização de retorno (ret_rms) para evitar \"herdar\" escalas antigas
            vec.ret_rms = RunningMeanStd(shape=())
            try:
                vec.returns = np.zeros((vec.num_envs,), dtype=np.float32)
            except Exception:
                pass
            print('[ENV] reset_ret_rms=1: VecNormalize.ret_rms foi resetado')
    else:
        if ckpt_vec_path is not None:
            print(f"[ENV] Aviso: ckpt_vec_path {ckpt_vec_path} não encontrado. Criando VecNormalize novo.")
        vec = VecNormalize(
            venv,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=cfg.gamma,
            epsilon=1e-6,
        )

    return vec



def build_eval_env(cfg: TrainConfig, zonas_from_file: Optional[List[Tuple[int, int]]]) -> VecNormalize:
    def _f():
        env = _build_drone_env(cfg, zonas_from_file, eval_mode=False)
        env.reset(seed=cfg.seed + 1)  # <- seed determinístico do eval
        env = _wrap_env(env, cfg.episode_horizon)
        return env

    venv = DummyVecEnv([_f])          # <- primeiro cria
    venv.seed(cfg.seed + 1)           # <- depois seed
    venv = VecCheckNan(venv, raise_exception=True)

    vec = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,             # pode manter True
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=cfg.gamma,              # <- usar o mesmo gamma do PPO
        epsilon=1e-6,
    )

    vec.training = False              # <- não atualizar estatísticas no eval

    return vec




# =========================
# Callbacks customizados
# =========================
class SafeEnvWrapper(gym.Wrapper):
    """
    Wrapper de segurança: trata exceções no step/reset para não travar SubprocVecEnv.
    Garante que nenhuma exceção escape para o processo pai (evitando EOFError).
    """

    def step(self, action):
        try:
            return self.env.step(action)
        except Exception as e:
            print(f"[SafeEnvWrapper] Exceção no step: {e}", flush=True)
            # Tenta recuperar com um reset protegido
            try:
                obs, info = self.env.reset()
            except Exception as e2:
                print(f"[SafeEnvWrapper] Exceção no reset de recuperação: {e2}", flush=True)
                # Como último recurso, devolve uma observação aleatória válida
                obs = self.env.observation_space.sample()
                info = {"error_reset": True}
            # Gymnasium: step retorna (obs, reward, terminated, truncated, info)
            return obs, -1e6, True, False, info

    def reset(self, **kwargs):
        try:
            return self.env.reset(**kwargs)
        except Exception as e:
            print(f"[SafeEnvWrapper] Exceção no reset: {e}", flush=True)
            try:
                obs, info = self.env.reset()
            except Exception as e2:
                print(f"[SafeEnvWrapper] Exceção no reset de recuperação: {e2}", flush=True)
                obs = self.env.observation_space.sample()
                info = {"error_reset": True}
            return obs, info





class EntropyCoefLoggerCallback(BaseCallback):
    """Registra (e opcionalmente agenda) o valor efetivo do ent_coef no TensorBoard.

    Observação: SB3 PPO espera ent_coef como float. Para ter schedule determinístico, este callback
    atualiza model.ent_coef no fim de cada rollout.
    """

    def __init__(self, ent_start=None, ent_end=None, verbose: int = 0):
        super().__init__(verbose)
        self.ent_start = ent_start
        self.ent_end = ent_end

    def _on_step(self) -> bool:
        # obrigatório (BaseCallback é abstrata)
        return True

    def _on_rollout_end(self) -> None:
        try:
            pr = float(getattr(self.model, "_current_progress_remaining", 0.0))  # 1->0 ao longo do treino
            progress = 1.0 - pr  # 0->1
            # Atualiza ent_coef se schedule foi solicitado
            if self.ent_start is not None and self.ent_end is not None:
                val = float(cosine_schedule(self.ent_start, self.ent_end, progress))
                try:
                    self.model.ent_coef = val
                except Exception:
                    pass
            else:
                val = float(getattr(self.model, "ent_coef", 0.0) or 0.0)

            self.logger.record("train/ent_coef_value", float(val))
        except Exception:
            pass


class EnvMetricsCallback(BaseCallback):
    """Callback para logar métricas do info[] do ambiente no TensorBoard.

    Espera chaves como 'reliability', 'max_hops', 'avg_hops', 'throughput',
    'drones_in_zone', 'links_cross_zone', etc., em cada info.
    """

    def __init__(self, metric_prefix: str = "custom", verbose: int = 0):
        super().__init__(verbose)
        self.metric_prefix = metric_prefix

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True

        acc: dict = {}
        for info in infos:
            if not isinstance(info, dict):
                continue
            for k, v in info.items():
                if isinstance(v, (int, float, bool)):
                    acc.setdefault(k, []).append(float(v))

        if not acc:
            return True

        # mapeia alguns nomes para *_mean para ficar mais explícito no TensorBoard
        name_map = {
            "reliability": "reliability_mean",
            "max_hops": "max_hops_mean",
            "avg_hops": "avg_hops_mean",
        }

        for k, vals in acc.items():
            mean_val = float(sum(vals) / max(1, len(vals)))
            tag_name = name_map.get(k, k)
            tag = f"{self.metric_prefix}/{tag_name}"
            self.logger.record(tag, mean_val)
        return True




class TrainingSummaryCallback(BaseCallback):
    """Callback leve e determinístico para registrar progresso no TensorBoard.

    Mantém o custo baixo: só registra a cada `metric_freq` timesteps.
    """

    def __init__(self, metric_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.metric_freq = int(metric_freq)
        self._last_t = 0

    def _on_step(self) -> bool:
        try:
            if (self.num_timesteps - self._last_t) >= self.metric_freq:
                self._last_t = int(self.num_timesteps)
                # alguns sinais úteis
                self.logger.record("train/num_timesteps", float(self.num_timesteps))
                # progress remaining (1->0)
                pr = float(getattr(self.model, "_current_progress_remaining", 0.0))
                self.logger.record("train/progress_remaining", pr)
        except Exception:
            pass
        return True

class SaveBestModelCallback(BaseCallback):
    """
    Callback que salva o melhor modelo com base no resultado do EvalCallback.

    Sempre que o EvalCallback encontrar um novo "melhor modelo"
    (melhor `best_mean_reward`), este callback:

      - garante o salvamento de best_model.zip em `save_path`;
      - salva também:
          * best_vecnormalize.pkl (VecNormalize do ambiente de treino)
          * best_seed.txt        (seed usada no treino)
    """

    def __init__(
        self,
        save_path: str,
        eval_callback: EvalCallback,
        seed: Optional[int] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.eval_callback = eval_callback
        self.seed = seed
        # Último best_mean_reward conhecido (do EvalCallback)
        self._last_best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        """
        Não depende mais do logger.
        Olha diretamente para eval_callback.best_mean_reward.
        Só faz algo quando esse valor melhora.
        """

        # O EvalCallback só terá esse atributo depois da primeira avaliação
        best_from_eval = getattr(self.eval_callback, "best_mean_reward", None)
        if best_from_eval is None:
            return True

        # Se não houve melhora em relação ao último valor conhecido, não faz nada
        if best_from_eval <= self._last_best_mean_reward:
            return True

        # Novo melhor modelo segundo o EvalCallback
        self._last_best_mean_reward = best_from_eval

        os.makedirs(self.save_path, exist_ok=True)

        # 1) Salvar o best_model.zip (não faz mal sobrescrever: é só o melhor)
        model_path = os.path.join(self.save_path, "best_model.zip")
        try:
            self.model.save(model_path)
        except Exception as e:
            if self.verbose > 0:
                print(f"[CB] Erro ao salvar best_model.zip: {e}")
            # Mesmo que dê erro aqui, vamos tentar salvar VecNormalize/seed
            # mas provavelmente o erro é mais grave. Ainda assim, não aborta treino.

        # 2) Salvar VecNormalize associado (best_vecnormalize.pkl)
        vec_path = None
        try:
            # Usa o env direto do modelo, que em 5GML.py é um VecNormalize
            env = self.model.get_env()
            if isinstance(env, VecNormalize):
                vec_path = os.path.join(self.save_path, "best_vecnormalize.pkl")
                env.save(vec_path)
            elif hasattr(env, "venv") and isinstance(env.venv, VecNormalize):
                vec_path = os.path.join(self.save_path, "best_vecnormalize.pkl")
                env.venv.save(vec_path)
        except Exception as e:
            if self.verbose > 0:
                print(f"[CB] Aviso: não foi possível salvar VecNormalize do best_model: {e}")
            vec_path = None

        # 3) Salvar a seed em best_seed.txt (se fornecida)
        seed_path = None
        if self.seed is not None:
            try:
                seed_path = os.path.join(self.save_path, "best_seed.txt")
                with open(seed_path, "w", encoding="utf-8") as f:
                    f.write(str(self.seed) + "\n")
            except Exception as e:
                if self.verbose > 0:
                    print(f"[CB] Aviso: não foi possível salvar best_seed.txt: {e}")
                seed_path = None

        if self.verbose > 0:
            msg = (
                f"[CB] Novo melhor modelo (segundo EvalCallback, "
                f"mean_reward={best_from_eval:.3f}) salvo em {model_path}"
            )
            if vec_path:
                msg += f" | VecNormalize: {vec_path}"
            if seed_path:
                msg += f" | Seed: {seed_path}"
            print(msg)

        return True






class TerminalEvalBestCallback(BaseCallback):
    """Avaliação periódica baseada em métricas TERMINAIS (alinha com o programa de testes).

    Critério de "melhor modelo":
      1) maior success_rate (info['is_success'] ou info['success'])
      2) em empate, maior reliability_terminal_mean
      3) em empate, maior reliability_terminal_p10 (robustez)

    Salva em save_dir:
      - best_model.zip
      - best_vecnormalize.pkl (do ambiente de treino)
      - best_seed.txt (se fornecida)
      - best_metrics.json
      - eval_terminal_metrics.csv (histórico)
    """

    def __init__(
        self,
        train_env: VecNormalize,
        eval_env: VecNormalize,
        eval_freq_calls: int,
        n_eval_episodes: int,
        deterministic: bool,
        save_dir: str,
        log_dir: str,
        seed: Optional[int] = None,
        verbose: int = 1,
        rollback_on_regress: bool = True,
        regress_patience: int = 2,
        regress_best_success_threshold: float = 0.10,
        regress_min_success_drop: float = 0.05,
        regress_min_rel_drop: float = 0.01,
        regress_min_zone_increase: float = 0.5,
        regress_min_left_increase: float = 0.5,
        regress_min_right_ratio_drop: float = 0.05,
        regress_min_dag_drop: float = 0.05,
        rollback_cooldown_evals: int = 1,
        early_stop_on_regress: bool = False,
    ):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env
        self.eval_freq_calls = int(max(1, eval_freq_calls))
        self.n_eval_episodes = int(max(1, n_eval_episodes))
        self.deterministic = bool(deterministic)
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.seed = seed

        self._best_key = (-1.0, -1e9, -1e9, -1.0, -1.0, -1.0, -1.0)  # (success_rate, -zone_viol_mean, -left_links_mean, dag_rate, right_links_ratio_mean, rel_mean, rel_p10)

        # Anti-forgetting (lexicographic guard):
        # If the training model regresses (w.r.t. the best lexicographic key) for
        # `regress_patience` consecutive terminal evaluations, rollback the policy
        # (and VecNormalize stats) to the best snapshot saved so far.
        self.rollback_on_regress = bool(rollback_on_regress)
        self.early_stop_on_regress = bool(early_stop_on_regress)
        self.regress_patience = int(max(1, regress_patience))
        self.regress_best_success_threshold = float(max(0.0, regress_best_success_threshold))
        self.regress_min_success_drop = float(max(0.0, regress_min_success_drop))
        self.regress_min_rel_drop = float(max(0.0, regress_min_rel_drop))
        self.regress_min_zone_increase = float(max(0.0, regress_min_zone_increase))
        self.regress_min_left_increase = float(max(0.0, regress_min_left_increase))
        self.regress_min_right_ratio_drop = float(max(0.0, regress_min_right_ratio_drop))
        self.regress_min_dag_drop = float(max(0.0, regress_min_dag_drop))
        self.rollback_cooldown_evals = int(max(0, rollback_cooldown_evals))
        self._regress_strikes = 0
        self._regress_cooldown = 0

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self._csv_path = os.path.join(self.log_dir, "eval_terminal_metrics.csv")
        if not os.path.isfile(self._csv_path):
            with open(self._csv_path, "w", encoding="utf-8") as f:
                f.write(
                    "timesteps,success_rate,rel_term_mean,rel_term_p10,rel_term_min,rel_term_max,"
                    "zone_viol_mean,left_links_mean,max_hops_mean\n"
                )

    def _sync_vecnormalize(self):
        try:
            if isinstance(self.train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                self.eval_env.obs_rms = self.train_env.obs_rms
                self.eval_env.ret_rms = self.train_env.ret_rms
                try:
                    self.eval_env.norm_reward = bool(getattr(self.train_env, "norm_reward", True))
                except Exception:
                    pass
                try:
                    self.eval_env.gamma = float(getattr(self.train_env, "gamma", self.eval_env.gamma))
                except Exception:
                    pass
        except Exception as e:
            if self.verbose > 0:
                print(f"[TerminalEval] Aviso: falha ao sincronizar VecNormalize: {e}")

    def _run_one_episode(self) -> Dict[str, Any]:
        obs = self.eval_env.reset()
        done = False
        last_info: Dict[str, Any] = {}
        use_masks = (
            _SB3_CONTRIB_OK
            and (MaskablePPO is not None)
            and isinstance(self.model, MaskablePPO)
            and (get_action_masks is not None)
        )
        while not done:
            if use_masks:
                # get_action_masks suporta VecEnv/VecNormalize e busca env.action_masks()
                masks = get_action_masks(self.eval_env)
                action, _ = self.model.predict(obs, deterministic=self.deterministic, action_masks=masks)
            else:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)

            obs, _reward, dones, infos = self.eval_env.step(action)
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            last_info = info0 if isinstance(info0, dict) else {}
            done = bool(dones[0]) if isinstance(dones, (list, tuple, np.ndarray)) else bool(dones)
        return last_info

    def _extract_terminal_metrics(self, info: Dict[str, Any]) -> Tuple[float, bool, int, int, float, int, float]:
        rel_term = float(info.get("reliability_terminal", info.get("reliability", 0.0)) or 0.0)
        success = bool(info.get("is_success", info.get("success", False)))
        if "zone_violations_terminal" in info:
            zone_viol = int(info.get("zone_violations_terminal", 0) or 0)
        else:
            zone_viol = int(info.get("drones_in_zone", 0) or 0) + int(info.get("links_cross_zone", 0) or 0)
        left_links = int(info.get("left_links_terminal", info.get("left_links_count", 0) or 0) or 0)
        right_ratio = float(info.get("right_links_ratio_terminal", info.get("right_links_ratio", 0.0)) or 0.0)
        is_dag = int(info.get("is_dag_terminal", info.get("is_dag", 0) or 0) or 0)
        max_hops = float(info.get("max_hops", 0.0) or 0.0)
        return rel_term, success, zone_viol, left_links, right_ratio, is_dag, max_hops


    def _copy_vecnormalize_stats(self, dst: VecNormalize, src: VecNormalize) -> None:
        """Copy VecNormalize running stats into an existing wrapper (no re-wrapping)."""
        for attr in ("obs_rms", "ret_rms", "clip_obs", "clip_reward", "gamma", "epsilon", "norm_obs", "norm_reward"):
            if hasattr(src, attr):
                try:
                    setattr(dst, attr, getattr(src, attr))
                except Exception:
                    pass
        # Some versions keep per-env returns buffer
        if hasattr(src, "returns") and hasattr(dst, "returns"):
            try:
                dst.returns = np.array(getattr(src, "returns"), copy=True)
            except Exception:
                pass

    def _rollback_to_best(self) -> bool:
        """Rollback current training model (and VecNormalize) to the best snapshot saved."""
        best_model_path = os.path.join(self.save_dir, "best_model.zip")
        if not os.path.isfile(best_model_path):
            return False

        ok = True

        # 1) Rollback policy weights (and optimizer if possible)
        try:
            loaded = self.model.__class__.load(
                best_model_path,
                env=self.model.get_env(),
                device=getattr(self.model, "device", "auto"),
            )
            self.model.policy.load_state_dict(loaded.policy.state_dict())
            if hasattr(self.model.policy, "optimizer") and hasattr(loaded.policy, "optimizer"):
                try:
                    self.model.policy.optimizer.load_state_dict(loaded.policy.optimizer.state_dict())
                except Exception:
                    pass
        except Exception:
            try:
                # Fallback: load parameters directly into the current model
                self.model.set_parameters(best_model_path, exact_match=False)
            except Exception as e:
                ok = False
                if self.verbose > 0:
                    print(f"[TerminalEval] Rollback falhou ao recarregar pesos: {type(e).__name__}: {e}")

        # 2) Rollback VecNormalize stats (train env)
        try:
            best_vn_path = os.path.join(self.save_dir, "best_vecnormalize.pkl")
            if isinstance(self.train_env, VecNormalize) and os.path.isfile(best_vn_path):
                vn_loaded = VecNormalize.load(best_vn_path, self.train_env.venv)
                self._copy_vecnormalize_stats(self.train_env, vn_loaded)
        except Exception as e:
            if self.verbose > 0:
                print(f"[TerminalEval] Aviso: rollback de VecNormalize falhou: {type(e).__name__}: {e}")

        return ok

    def _on_step(self) -> bool:
        if (self.n_calls % self.eval_freq_calls) != 0:
            return True

        self._sync_vecnormalize()

        rel_terms: List[float] = []
        successes = 0
        zone_viol_list: List[int] = []
        left_links_list: List[int] = []
        right_ratio_list: List[float] = []
        dag_list: List[int] = []
        max_hops_list: List[float] = []

        for _ in range(self.n_eval_episodes):
            info = self._run_one_episode()
            rel_term, success, zone_viol, left_links, right_ratio, is_dag, max_hops = self._extract_terminal_metrics(info)
            rel_terms.append(rel_term)
            successes += int(success)
            zone_viol_list.append(zone_viol)
            left_links_list.append(left_links)
            right_ratio_list.append(right_ratio)
            dag_list.append(int(is_dag))
            max_hops_list.append(max_hops)

        rel_terms_arr = np.array(rel_terms, dtype=np.float64)
        success_rate = float(successes) / float(self.n_eval_episodes)

        rel_mean = float(np.mean(rel_terms_arr)) if rel_terms_arr.size else 0.0
        rel_p10 = float(np.percentile(rel_terms_arr, 10)) if rel_terms_arr.size else 0.0
        rel_min = float(np.min(rel_terms_arr)) if rel_terms_arr.size else 0.0
        rel_max = float(np.max(rel_terms_arr)) if rel_terms_arr.size else 0.0

        zone_viol_mean = float(np.mean(zone_viol_list)) if zone_viol_list else 0.0
        left_links_mean = float(np.mean(left_links_list)) if left_links_list else 0.0
        right_ratio_mean = float(np.mean(right_ratio_list)) if right_ratio_list else 0.0
        dag_rate = float(np.mean(dag_list)) if dag_list else 0.0
        max_hops_mean = float(np.mean(max_hops_list)) if max_hops_list else 0.0

        # TensorBoard (TERMINAL metrics only; no contamination from dirty reset)
        self.logger.record("eval_terminal/success_rate", success_rate)
        self.logger.record("eval_terminal/success_ratio", success_rate)  # alias
        self.logger.record("eval_terminal/reliability_terminal_mean", rel_mean)
        self.logger.record("eval_terminal/reliability_terminal_p10", rel_p10)
        self.logger.record("eval_terminal/zone_viol_mean", zone_viol_mean)
        self.logger.record("eval_terminal/violation_zone_mean", zone_viol_mean)  # alias requested
        self.logger.record("eval_terminal/left_links_mean", left_links_mean)
        self.logger.record("eval_terminal/right_links_ratio_mean", right_ratio_mean)
        self.logger.record("eval_terminal/right_links_ratio", right_ratio_mean)  # alias requested
        self.logger.record("eval_terminal/is_dag_rate", dag_rate)
        self.logger.record("eval_terminal/max_hops_mean", max_hops_mean)

        with open(self._csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{int(self.num_timesteps)},{success_rate:.6f},{rel_mean:.12f},{rel_p10:.12f},"
                f"{rel_min:.12f},{rel_max:.12f},{zone_viol_mean:.6f},{left_links_mean:.6f},"
                f"{right_ratio_mean:.6f},{dag_rate:.6f},{max_hops_mean:.6f}\n"
            )

        # Lexicographic key aligned with FINAL success definition:
        # success -> (zone clean) -> (no left-links) -> DAG -> right-links -> reliability
        key = (success_rate, -zone_viol_mean, -left_links_mean, dag_rate, right_ratio_mean, rel_mean, rel_p10)

        # --- Anti-forgetting: regression detection vs best lexicographic key ---
        if self._regress_cooldown > 0:
            self._regress_cooldown -= 1
        else:
            # best key components
            best_s, best_nz, best_nl, best_dag, best_rr, best_rm, _best_p10 = self._best_key
            best_zone = -float(best_nz)
            best_left = -float(best_nl)

            regress = False

            if best_s >= self.regress_best_success_threshold:
                # once we have meaningful success, protect success first, then constraints, then reliability
                if success_rate < (best_s - self.regress_min_success_drop):
                    regress = True
                elif abs(success_rate - best_s) <= 1e-12:
                    if zone_viol_mean > (best_zone + self.regress_min_zone_increase):
                        regress = True
                    elif left_links_mean > (best_left + self.regress_min_left_increase):
                        regress = True
                    elif dag_rate < (best_dag - self.regress_min_dag_drop):
                        regress = True
                    elif right_ratio_mean < (best_rr - self.regress_min_right_ratio_drop):
                        regress = True
                    elif rel_mean < (best_rm - self.regress_min_rel_drop):
                        regress = True
            else:
                # before success appears, protect partial learning (constraints and structure)
                if zone_viol_mean > (best_zone + self.regress_min_zone_increase):
                    regress = True
                elif left_links_mean > (best_left + self.regress_min_left_increase):
                    regress = True
                elif dag_rate < (best_dag - self.regress_min_dag_drop):
                    regress = True
                elif right_ratio_mean < (best_rr - self.regress_min_right_ratio_drop):
                    regress = True
                elif rel_mean < (best_rm - self.regress_min_rel_drop):
                    regress = True

            if regress:
                self._regress_strikes += 1
            else:
                self._regress_strikes = 0

        self.logger.record("eval_terminal/regress_strikes", float(self._regress_strikes))

        if key > self._best_key:
            self._best_key = key
            self._regress_strikes = 0
            self._regress_cooldown = 0

            model_path = os.path.join(self.save_dir, "best_model.zip")
            self.model.save(model_path)

            try:
                if isinstance(self.train_env, VecNormalize):
                    self.train_env.save(os.path.join(self.save_dir, "best_vecnormalize.pkl"))
            except Exception as e:
                if self.verbose > 0:
                    print(f"[TerminalEval] Aviso: falha ao salvar best_vecnormalize.pkl: {e}")

            if self.seed is not None:
                try:
                    with open(os.path.join(self.save_dir, "best_seed.txt"), "w", encoding="utf-8") as sf:
                        sf.write(str(int(self.seed)) + "\n")
                except Exception:
                    pass

            metrics = {
                "timesteps": int(self.num_timesteps),
                "success_rate": success_rate,
                "reliability_terminal_mean": rel_mean,
                "reliability_terminal_p10": rel_p10,
                "reliability_terminal_min": rel_min,
                "reliability_terminal_max": rel_max,
                "zone_viol_mean": zone_viol_mean,
                "left_links_mean": left_links_mean,
                "right_links_ratio_mean": right_ratio_mean,
                "is_dag_rate": dag_rate,
                "max_hops_mean": max_hops_mean,
            }
            try:
                with open(os.path.join(self.save_dir, "best_metrics.json"), "w", encoding="utf-8") as jf:
                    json.dump(metrics, jf, indent=2)
            except Exception:
                pass

            if self.verbose > 0:
                print(
                    "[TerminalEval] Novo melhor modelo salvo | "
                    f"succ={success_rate:.3f} zone_mean={zone_viol_mean:.3f} left_mean={left_links_mean:.3f} "
                    f"dag={dag_rate:.3f} right_ratio={right_ratio_mean:.3f} rel_mean={rel_mean:.6f} rel_p10={rel_p10:.6f} | "
                    f"timesteps={int(self.num_timesteps)}"
                )

        # --- Anti-forgetting: rollback / early-stop on consecutive regression ---
        if self.rollback_on_regress and (self._regress_strikes >= self.regress_patience):
            if self.verbose > 0:
                bs, bnz, bnl, bdag, brr, brm, _ = self._best_key
                print(
                    "[TerminalEval] Regressão detectada "
                    f"({self._regress_strikes}/{self.regress_patience}). "
                    f"Rollback para best_model.zip | best_succ={bs:.3f} best_zone={-bnz:.3f} best_right={brr:.3f} "
                    f"| timesteps={int(self.num_timesteps)}"
                )
            _ok = self._rollback_to_best()
            self._regress_strikes = 0
            self._regress_cooldown = int(self.rollback_cooldown_evals)

            if self.early_stop_on_regress:
                if self.verbose > 0:
                    print("[TerminalEval] Early-stop por regressão (após rollback).")
                return False

        return True
# =========================
# HPO com Optuna
# =========================
def find_latest_checkpoint(ckpt_dir: str, name_prefix: str = "ckpt") -> Tuple[Optional[str], Optional[str], int]:
    """
    Procura, em `ckpt_dir`, o arquivo de checkpoint mais recente gerado pelo CheckpointCallback,
    assumindo o padrão de nomes:
      - {name_prefix}_{N}_steps.zip
      - {name_prefix}_vecnormalize_{N}_steps.pkl

    Retorna:
      (caminho_modelo, caminho_vecnormalize, N_timesteps)
    ou (None, None, 0) se nada for encontrado.
    """
    if not os.path.isdir(ckpt_dir):
        return None, None, 0

    best_steps = -1
    best_model_path = None

    for fname in os.listdir(ckpt_dir):
        # Ex.: ckpt_123456_steps.zip
        if not fname.startswith(name_prefix + "_"):
            continue
        if not fname.endswith("_steps.zip"):
            continue

        parts = fname.split("_")
        # esperado algo como ["ckpt", "123456", "steps.zip"]
        if len(parts) < 3:
            continue

        steps_str = parts[1]
        try:
            steps = int(steps_str)
        except ValueError:
            continue

        if steps > best_steps:
            best_steps = steps
            best_model_path = os.path.join(ckpt_dir, fname)

    if best_model_path is None or best_steps < 0:
        return None, None, 0

    # Tenta achar o VecNormalize correspondente: ckpt_vecnormalize_{N}_steps.pkl
    best_vec_path = None
    target_token = f"vecnormalize_{best_steps}_steps"
    for fname in os.listdir(ckpt_dir):
        if target_token in fname and fname.endswith(".pkl"):
            best_vec_path = os.path.join(ckpt_dir, fname)
            break

    return best_model_path, best_vec_path, best_steps



def _load_zonas(zonas_file: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    if not zonas_file:
        return None
    if not os.path.exists(zonas_file):
        print(f"[WARN] Zonas file {zonas_file} não encontrado.")
        return None
    zonas: List[Tuple[int, int]] = []
    with open(zonas_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            x, y = int(parts[0]), int(parts[1])
            zonas.append((x, y))
    print(f"[INFO] Carregadas {len(zonas)} zonas de {zonas_file}")
    return zonas


def run_optuna(cfg: TrainConfig):
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import SuccessiveHalvingPruner, MedianPruner

    # ------------------------------------------------------------
    # Seleção do algoritmo (PPO vs MaskablePPO) para este HPO
    # ------------------------------------------------------------
    if getattr(cfg, "mask_actions", 0):
        try:
            from sb3_contrib import MaskablePPO as model_cls  # type: ignore
            from sb3_contrib.common.maskable.utils import get_action_masks as _get_action_masks  # type: ignore
            _is_maskable = True
        except Exception as e:
            raise RuntimeError(
                "mask_actions=1, mas sb3-contrib não está disponível. Instale com: pip install sb3-contrib"
            ) from e
    else:
        from stable_baselines3 import PPO as model_cls  # type: ignore
        _is_maskable = False

    def objective(trial: "optuna.trial.Trial") -> float:
        # seed global por trial para reprodutibilidade
        if cfg.hpo_seed >= 0:
            seed_trial = cfg.hpo_seed + trial.number * cfg.hpo_k_seeds
        else:
            seed_trial = random.randint(0, 10_000_000)
        set_global_seeds(seed_trial)
        # hiperparâmetros
        lr = trial.suggest_float("learning_rate", 3e-5, 5e-4, log=True)
        clip = trial.suggest_float("clip_range", 0.10, 0.30)
        ent_coef = trial.suggest_float("ent_coef", 0.001, 0.05, log=True)
        vf_coef = trial.suggest_float("vf_coef", 0.5, 1.5)
        gamma = trial.suggest_float("gamma", 0.97, 0.999)
        gae_lam = trial.suggest_float("gae_lambda", 0.8, 0.99)
        n_epochs = trial.suggest_int("n_epochs", 10, 35)
        target_kl = trial.suggest_float("target_kl", 0.005, 0.05)

        # arquitetura
        net_width = trial.suggest_int("net_width", 128, 512, log=False, step=64)
        net_layers = trial.suggest_int("net_layers", 2, 4)
        activation = trial.suggest_categorical("activation", ["tanh", "relu"])

        # config "derivada"
        env_timesteps = min(cfg.timesteps, cfg.hpo_budget)
        n_steps = cfg.n_steps
        n_envs = cfg.n_envs
        total_batch = n_envs * n_steps
        batch_size = min(cfg.batch_size, total_batch)
        seed_trial = cfg.seed if cfg.seed >= 0 else random.randint(0, 10_000_000)
        print(f"[HPO] Trial {trial.number} com seed base {seed_trial}")

        # Política
        if activation == "tanh":
            act_fn = torch.nn.Tanh
        else:
            act_fn = torch.nn.ReLU

        net_arch = [net_width] * net_layers
        policy_kwargs = dict(
            net_arch=net_arch,
            activation_fn=act_fn,
        )


        # ------------------------------------------------------------
        # Score do HPO: confiabilidade TERMINAL com restrição de zona
        # ------------------------------------------------------------
        # Objetivo: maximizar reliability_terminal_mean EXIGINDO zero violação de zona:
        #   zone_viol = drones_in_zone + links_cross_zone  (métricas do info terminal)
        # Penalizamos fortemente qualquer violação para que o Optuna priorize soluções sem zona.
        ZONE_PENALTY = 1000.0

        def _evaluate_terminal(_model, _eval_env, n_eps: int):
            rel_terms = []
            zone_viol = []
            succ = 0
            for _ in range(int(max(1, n_eps))):
                obs = _eval_env.reset()
                done = False
                last_info = {}
                while not done:
                    if _is_maskable:
                        masks = _get_action_masks(_eval_env)
                        act, _ = _model.predict(obs, deterministic=True, action_masks=masks)
                    else:
                        act, _ = _model.predict(obs, deterministic=True)
                    obs, _rew, dones, infos = _eval_env.step(act)
                    info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
                    last_info = info0 if isinstance(info0, dict) else {}
                    done = bool(dones[0]) if isinstance(dones, (list, tuple, np.ndarray)) else bool(dones)

                rel = float(last_info.get("reliability_terminal", last_info.get("reliability", 0.0)) or 0.0)
                zv = int(last_info.get("drones_in_zone", 0) or 0) + int(last_info.get("links_cross_zone", 0) or 0)
                rel_terms.append(rel)
                zone_viol.append(zv)
                succ += int(bool(last_info.get("is_success", last_info.get("success", False))))

            rel_mean = float(np.mean(rel_terms)) if rel_terms else 0.0
            zone_mean = float(np.mean(zone_viol)) if zone_viol else 0.0
            success_rate = float(succ) / float(max(1, n_eps))
            return rel_mean, zone_mean, success_rate

        # seeds para robustez (variando por trial)
        seeds = [seed_trial + k for k in range(cfg.hpo_k_seeds)]
        scores = []
        rel_means = []
        zone_means = []
        success_rates = []

        from stable_baselines3.common.vec_env import VecNormalize as VN
        from stable_baselines3.common.vec_env import DummyVecEnv as SPE


        for k_idx, seed_k in enumerate(seeds):
            # construção de envs por seed
            def make_env(eval_mode: bool, seed_val: int, num_drones_val: int):
                def _f():
                    env = _build_drone_env(cfg, _load_zonas(cfg.zonas_file), eval_mode=eval_mode, num_drones_override=num_drones_val)
                    env.reset(seed=seed_val)
                    env = _wrap_env(env, cfg.episode_horizon)
                    return env
                return _f

            num_drones_trial = cfg.num_drones
            print(f"[HPO] Trial {trial.number} seed_idx={k_idx}, seed_k={seed_k}, drones={num_drones_trial}")

            train_env = SPE(
                [make_env(False, seed_val=seed_k + i, num_drones_val=num_drones_trial)
                 for i in range(cfg.n_envs)]
            )
            train_env = VecCheckNan(train_env, raise_exception=True)
            train_env = VN(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0, epsilon=1e-6)

            eval_env = SPE(
                [make_env(True, seed_val=seed_k, num_drones_val=num_drones_trial)]
            )

            eval_env = VecCheckNan(eval_env, raise_exception=True)
            eval_env = VN(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, epsilon=1e-6)
            eval_env.obs_rms = train_env.obs_rms

            eval_dir = os.path.join(cfg.hpo_ckpt_dir, f"trial_{trial.number}", f"seed_{seed_k}")
            os.makedirs(eval_dir, exist_ok=True)
            # cfg.eval_freq é dado em TIMESTEPS reais.
            # O EvalCallback, porém, interpreta eval_freq como número de chamadas de callback.
            # Como usamos n_envs ambientes paralelos, cada chamada de callback acumula n_envs timesteps.
            # Aqui convertemos a frequência em timesteps (limitada por um valor ligado ao batch)
            # para a frequência em chamadas de callback.
            eval_freq_steps = max(min(
                batch_size * n_epochs // 2,
                cfg.eval_freq
            ), batch_size)
            eval_freq_calls = max(eval_freq_steps // n_envs, 1)

            # Eval env: não atualizar estatísticas (comparação mais estável)
            eval_env.training = False

            model = model_cls(
                "MlpPolicy",
                train_env,
                learning_rate=lr,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lam,
                clip_range=clip,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=cfg.max_grad_norm,
                target_kl=target_kl,
                policy_kwargs=policy_kwargs,
                verbose=0,
            )
            # Treina
            model.learn(total_timesteps=env_timesteps)

            # Avalia métricas TERMINAIS
            rel_mean, zone_mean, success_rate = _evaluate_terminal(model, eval_env, cfg.n_eval_episodes)

            # Score escalar para o Optuna (maximizar):
            # - confiabilidade terminal
            # - penalidade MUITO forte para violações de zona
            if zone_mean > 0.0:
                score = float(-1e6 - ZONE_PENALTY * zone_mean + rel_mean)
            else:
                score = float(rel_mean)

            scores.append(score)
            rel_means.append(float(rel_mean))
            zone_means.append(float(zone_mean))
            success_rates.append(float(success_rate))

            # report para pruner (ASHA/Median)
            try:
                trial.report(score, step=k_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            except Exception:
                # se optuna não estiver disponível aqui, ignore (não deve acontecer)
                pass

            print(
                f"[HPO] Trial {trial.number} seed_k={seed_k}: "
                f"score={score:.6f} rel_term_mean={rel_mean:.6f} zone_viol_mean={zone_mean:.3f} "
                f"success_rate={success_rate:.3f}"
            )

            train_env.close()
            eval_env.close()
        mean_rel = float(np.mean(rel_means)) if rel_means else 0.0
        mean_zone = float(np.mean(zone_means)) if zone_means else 0.0
        mean_succ = float(np.mean(success_rates)) if success_rates else 0.0

        val = float(np.mean(scores))
        std = float(np.std(scores))
        trial.set_user_attr("mean_score", val)
        trial.set_user_attr("std_score", std)
        trial.set_user_attr("rel_term_mean", mean_rel)
        trial.set_user_attr("zone_viol_mean", mean_zone)
        trial.set_user_attr("success_rate", mean_succ)
        return val

    # study / sampler / pruner
    os.makedirs(cfg.hpo_ckpt_dir, exist_ok=True)
    os.makedirs(cfg.hpo_ckpt_dir, exist_ok=True)

    # Storage: por padrão usa SQLite local; se OPTUNA_STORAGE_URL estiver setado,
    # usa (ex.: PostgreSQL) e evita "database is locked" do SQLite sob concorrência.
    db_path = os.path.join(cfg.hpo_ckpt_dir, "optuna_hpo.db")
    default_storage = f"sqlite:///{db_path}?timeout=60"
    storage = os.environ.get("OPTUNA_STORAGE_URL", default_storage)
    print(f"[HPO] Optuna storage = {storage}")

    # Se estiver usando SQLite, tente reduzir chance de lock (WAL + busy_timeout).
    if storage.startswith("sqlite:///"):
        try:
            import sqlite3
            sqlite_path = storage[len("sqlite:///"):].split("?", 1)[0]
            con = sqlite3.connect(sqlite_path, timeout=60)
            cur = con.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA busy_timeout=60000;")  # 60s
            con.commit()
            con.close()
        except Exception as e:
            print("[HPO] WARN: falha ao configurar WAL/busy_timeout no SQLite:", repr(e))

    sampler = TPESampler(seed=cfg.hpo_seed, multivariate=True)
    if cfg.hpo_pruner == "asha":
        pruner = SuccessiveHalvingPruner()
    elif cfg.hpo_pruner == "median":
        pruner = MedianPruner()
    else:
        pruner = None

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=cfg.hpo_study_name,
        storage=storage,
        load_if_exists=True,
    )

    # paralelismo configurável
    study.optimize(objective, n_trials=cfg.hpo_trials, n_jobs=cfg.hpo_n_jobs, gc_after_trial=True)

    # salvar CSV com attrs
    try:
        os.makedirs(cfg.hpo_ckpt_dir, exist_ok=True)
        df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
        csv_path = os.path.join(cfg.hpo_ckpt_dir, "study_trials.csv")
        df.to_csv(csv_path,
                  index=False, encoding="utf-8")
        print(f"[HPO] Trials salvos em {csv_path}")
    except Exception as e:
        print(f"[HPO] Falha ao salvar dataframe de trials: {e}")

    if study.best_trial is not None:
        print("[HPO] Melhor trial:")
        print(f"  number={study.best_trial.number}")
        print(f"  value={study.best_trial.value}")
        print("  params:")
        for k, v in study.best_trial.params.items():
            print(f"    {k} = {v}")
        ua = study.best_trial.user_attrs
        if "mean_score" in ua:
            print("  mean_score:", ua["mean_score"])
        if "std_score" in ua:
            print("  std_score:", ua["std_score"])
    else:
        print("[HPO] Nenhum trial válido encontrado.")


# =========================
# Treino "normal"
# =========================

def _divisores_uteis(n: int) -> List[int]:
    """
    Retorna divisores "não triviais" de n, ordenados decrescentemente.
    Útil para ajustar batch_size de forma que seja divisor do total_batch.
    """
    divs = []
    for d in range(1, n + 1):
        if n % d == 0:
            divs.append(d)
    divs.sort(reverse=True)
    return divs


def train(cfg: TrainConfig):

    torch.set_num_threads(1)
    if cfg.mask_actions and (not _SB3_CONTRIB_OK):
        raise RuntimeError("mask_actions=1, mas sb3-contrib não está instalado. Instale com: pip install sb3-contrib")
    # Ajuste de seed
    real_seed = set_global_seeds(cfg.seed)
    cfg.seed = real_seed

    # Ajuste de batch_size para ser divisor de (n_envs * n_steps)
    total = cfg.n_envs * cfg.n_steps
    if total % cfg.batch_size != 0:
        divs = _divisores_uteis(total)
        if divs:
            print(f"[WARN] Ajustando batch_size {cfg.batch_size} -> {divs[0]} (divisor de {total})")
            cfg.batch_size = divs[0]

    # logs
    os.makedirs(cfg.tb_log, exist_ok=True)
    os.makedirs(cfg.best_path, exist_ok=True)
    os.makedirs(cfg.eval_log_path, exist_ok=True)
    os.makedirs(cfg.ckpt_path, exist_ok=True)
    logger = configure(cfg.tb_log, ["stdout", "tensorboard"])

    # zonas
    zonas = _load_zonas(cfg.zonas_file)
    # Se for modo CONTINUE, tenta localizar o checkpoint mais recente
    ckpt_model_path: Optional[str] = None
    ckpt_vec_path: Optional[str] = None
    ckpt_steps: int = 0
    resuming_from_ckpt: bool = False

    if cfg.use_model == "continue":
        ckpt_model_path, ckpt_vec_path, ckpt_steps = find_latest_checkpoint(cfg.ckpt_path, name_prefix="ckpt")

        if ckpt_model_path is not None:
            print(
                f"[MAIN] Modo CONTINUE: encontrado checkpoint mais recente em {ckpt_model_path} "
                f"(≈ {ckpt_steps} timesteps)."
            )
            resuming_from_ckpt = True
        else:
            print(
                f"[MAIN] Modo CONTINUE solicitado, mas nenhum checkpoint encontrado em {cfg.ckpt_path}. "
                f"Treinando como FRESH."
            )


    # envs
        # envs
    if resuming_from_ckpt:
        # Continua treino a partir do VecNormalize salvo
        train_env = build_train_env(cfg, zonas, ckpt_vec_path=ckpt_vec_path)
    else:
        # Treino do zero (VecNormalize novo)
        train_env = build_train_env(cfg, zonas, ckpt_vec_path=None)

    eval_env = build_eval_env(cfg, zonas)

    # Se carregamos VecNormalize de checkpoint, sincroniza as estatísticas no eval_env
    if isinstance(train_env, VecNormalize) and isinstance(eval_env, VecNormalize):
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms

    # política
    if cfg.activation == "tanh":
        act_fn = torch.nn.Tanh
    else:
        act_fn = torch.nn.ReLU

    if cfg.policy_arch:
        # pode ser JSON ou string "256,256,128"
        try:
            if cfg.policy_arch.strip().startswith("{"):
                net_arch = json.loads(cfg.policy_arch)
            else:
                parts = [int(x.strip()) for x in cfg.policy_arch.split(",") if x.strip()]
                net_arch = parts
        except Exception as e:
            print(f"[WARN] policy_arch inválido ({cfg.policy_arch}), usando full net_width/layers. Erro: {e}")
            net_arch = [cfg.net_width] * cfg.net_layers
    else:
        net_arch = [cfg.net_width] * cfg.net_layers

    # se net_arch for dict, assumimos que já está no formato {"pi":[...], "vf":[...]}
    if isinstance(net_arch, dict):
        policy_kwargs = dict(
            net_arch=net_arch,
            activation_fn=act_fn,
        )
    else:
        policy_kwargs = dict(
            net_arch=net_arch,
            activation_fn=act_fn,
        )

    # Schedules (learning_rate, clip_range, ent_coef)
    def lr_schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        return cosine_schedule(cfg.lr_start, cfg.lr_end, progress)

    def clip_schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        return cosine_schedule(cfg.clip_start, cfg.clip_end, progress)

    if cfg.ent_start is not None and cfg.ent_end is not None:
        def ent_schedule(progress_remaining: float) -> float:
            progress = 1.0 - progress_remaining
            return cosine_schedule(cfg.ent_start, cfg.ent_end, progress)
        ent_coef = ent_schedule
    else:
        ent_coef = cfg.ent_coef

    # Modelo PPO
    # Modelo PPO
    if resuming_from_ckpt and ckpt_model_path is not None:
        print(f"[MAIN] Carregando modelo PPO a partir do checkpoint: {ckpt_model_path}")
        # device="auto" deixa o SB3 escolher; altere se quiser forçar "cpu"
        try:
            model = (MaskablePPO if cfg.mask_actions else PPO).load(ckpt_model_path, env=train_env, device="auto")
            model.set_logger(logger)
        except Exception as e:
            print(
                "[WARN] Falha ao carregar checkpoint como modelo compatível (provável mismatch de action_space). "
                f"Treinando como FRESH mantendo VecNormalize (se carregado). Erro: {type(e).__name__}: {str(e)[:200]}"
            )
            resuming_from_ckpt = False
            ckpt_model_path = None
            model = (MaskablePPO if cfg.mask_actions else PPO)(
                "MlpPolicy",
                train_env,
                learning_rate=lr_schedule,
                n_steps=cfg.n_steps,
                batch_size=cfg.batch_size,
                n_epochs=cfg.n_epochs,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                clip_range=clip_schedule,
                ent_coef=ent_coef,
                vf_coef=cfg.vf_coef,
                max_grad_norm=cfg.max_grad_norm,
                target_kl=cfg.target_kl,
                policy_kwargs=policy_kwargs,
                verbose=1,
            )
            model.set_logger(logger)
    else:
        print("[MAIN] Nenhum checkpoint válido para CONTINUE. Criando modelo (MaskablePPO/PPO) do zero.")
        model = (MaskablePPO if cfg.mask_actions else PPO)(
            "MlpPolicy",
            train_env,
            learning_rate=lr_schedule,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=clip_schedule,
            ent_coef=ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            target_kl=cfg.target_kl,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
        model.set_logger(logger)
    # Avaliação periódica baseada em métricas TERMINAIS (alinha com o programa de testes)
    # cfg.eval_freq é dado em TIMESTEPS reais; este callback usa número de chamadas.
    # Como usamos cfg.n_envs ambientes paralelos, cada chamada do callback corresponde a cfg.n_envs timesteps.
    eval_freq_calls = max(cfg.eval_freq // cfg.n_envs, 1)

    terminal_eval_cb = TerminalEvalBestCallback(
        train_env=train_env,
        eval_env=eval_env,
        eval_freq_calls=eval_freq_calls,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=cfg.deterministic_eval,
        save_dir=cfg.best_path,
        log_dir=cfg.eval_log_path,
        seed=cfg.seed,
        verbose=1,
    )

    # Checkpoint periódico:
    # cfg.checkpoint_freq é em timesteps; como usamos n_envs paralelos,
    # SB3 recomenda ajustar: save_freq = max(freq // n_envs, 1)
    ckpt_save_freq = max(cfg.checkpoint_freq // cfg.n_envs, 1)
    checkpoint_cb = CheckpointCallback(
        save_freq=ckpt_save_freq,
        save_path=cfg.ckpt_path,
        name_prefix="ckpt",
        save_replay_buffer=False,
        save_vecnormalize=True,  # salva também o VecNormalize em ckpt_vecnormalize_XXXX_steps.pkl
        verbose=1,
    )

    summary_cb = TrainingSummaryCallback(metric_freq=cfg.tb_metric_freq, verbose=1)
    env_metrics_cb = EnvMetricsCallback(metric_prefix="custom", verbose=0)
    entcoef_cb = EntropyCoefLoggerCallback(ent_start=cfg.ent_start, ent_end=cfg.ent_end, verbose=0)

    # Todos os callbacks juntos
    comb_cb = CallbackList([terminal_eval_cb, checkpoint_cb, summary_cb, env_metrics_cb, entcoef_cb])






    # ------------------------------------------------------------------
    # Treinamento
    # ------------------------------------------------------------------
    print(f"[INFO] Iniciando treinamento. Timesteps desta rodada: {cfg.timesteps}")

    already_trained = int(getattr(model, "num_timesteps", 0))

    if resuming_from_ckpt:
        # Modo CONTINUE real: retomando um modelo que já tem timesteps
        print(
            f"[MAIN] CONTINUE a partir de checkpoint: modelo já tem {already_trained} timesteps. "
            f"Vamos adicionar mais {cfg.timesteps} timesteps "
            f"(alvo interno ≈ {already_trained + cfg.timesteps})."
        )
        reset_flag = False       # NÃO zera o contador interno
        total_timesteps = cfg.timesteps  # timesteps ADICIONAIS
    else:
        # Treino do zero (FRESH)
        print(
            f"[MAIN] Modo FRESH: resetando num_timesteps e treinando por "
            f"{cfg.timesteps} timesteps a partir do zero."
        )
        reset_flag = True
        total_timesteps = cfg.timesteps


    # Sempre chama learn, mesmo que already_trained > cfg.timesteps
    model.learn(
        total_timesteps=total_timesteps,
        callback=comb_cb,
        reset_num_timesteps=reset_flag,
    )

    # Salvar modelo final
    os.makedirs(os.path.dirname(cfg.final_path), exist_ok=True)
    model.save(cfg.final_path)
    print(f"[INFO] Modelo final salvo em {cfg.final_path}")

    # Salvar VecNormalize
    norm_path = cfg.final_path + "_vecnormalize.pkl"
    train_env.save(norm_path)
    print(f"[INFO] VecNormalize salvo em {norm_path}")

    train_env.close()
    eval_env.close()


# =========================
# Main
# =========================

def main():
    cfg = parse_args()

    # Se hpo_optuna > 0, executa HPO
    if cfg.hpo_optuna > 0:
        print("[MAIN] Rodando HPO com Optuna...")
        run_optuna(cfg)
        return

    print("[MAIN] Rodando treino padrão PPO...")
    train(cfg)


if __name__ == "__main__":
    main()
