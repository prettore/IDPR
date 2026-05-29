# -*- coding: utf-8 -*-
"""
DroneEnv completo (PPO):
- Zonas parametrizáveis (count, w/h, single_fixed, p_sem_zona)
- Movimento macro+micro com níveis {0,1,2,4,8,16} células (step_code 0..5) com 5 direções (parado, +x, -x, +y, -y)
- Atualização de links com validações (max_out, sem self-loop, evita left-link opcional, evita cruzar zona)
- Cache de matriz de distâncias
- Recompensa com confiabilidade/latência/throughput/hops/estrutura
- Termos de espalhamento: NN-spacing, repulsão curta, uniformidade de ocupação (bins)
"""

import math
import zlib
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYMNASIUM = True
except Exception:
    import gym
    from gym import spaces

    _GYMNASIUM = False
from collections import deque
import torch
from scipy.sparse.csgraph import floyd_warshall

try:
    from calculaProbLink_otimizado import calcula_confiabilidade_iterativa, prob_link_literature, RadioParams
except Exception:
    try:
        from calculaProbLinkOtimizado import calcula_confiabilidade_iterativa, prob_link_literature, RadioParams
    except Exception:
        from calculaProbLink import calcula_confiabilidade_iterativa, prob_link_literature, RadioParams

# ---- Safe RadioParams construction (handles dataclass signatures) ----
_RADIO_DEFAULTS = dict(
    pt_dbm=19.54242509439325,
    gt_db=2.0,
    gr_db=2.0,
    freq_hz=2.4e9,
    n=2.3,
    d0_m=1.0,
    pmin_dbm=-83.0,
    sigma_db=4.0,
)

def _make_radio_params():
    """Create RadioParams robustly across different implementations."""
    try:
        return RadioParams()
    except Exception:
        pass
    try:
        return RadioParams(**_RADIO_DEFAULTS)
    except Exception:
        pass
    # last resort: allocate object and set attrs
    try:
        obj = RadioParams.__new__(RadioParams)
        for k, v in _RADIO_DEFAULTS.items():
            try:
                setattr(obj, k, v)
            except Exception:
                pass
        return obj
    except Exception:
        return None

def _torch_to_numpy_cpu_view(t: "torch.Tensor") -> np.ndarray:
    """Converte tensor torch -> numpy com o menor overhead possível.

    - CPU: retorna um *view* via .detach().numpy() (sem cópia).
    - Não-CPU: faz .detach().cpu().numpy() (cópia inevitável).

    Observação: este projeto roda em CPU por padrão (DEVICE=cpu), então
    isso elimina o custo do .cpu() e reduz overhead em hot-paths.
    """
    if not isinstance(t, torch.Tensor):
        return np.asarray(t)
    if t.device.type == "cpu":
        return t.detach().numpy()
    return t.detach().cpu().numpy()

# =========================
# Constantes/Parâmetros
# =========================

DEVICE = torch.device("cpu")

MAX_DRONES = 15
GRID_SIZE_DEFAULT = 100
SQUARE_SIZE_M_DEFAULT = 20.0

# Movimento: código discreto (0..3) -> passo em células {1,2,3,4}.
# Observação: dir_code==0 significa 'não mover' (o passo é ignorado).
STEP_CELLS_CHOICES = [1, 2, 3, 4, 8, 16]
STEP_CELLS_N = len(STEP_CELLS_CHOICES)

# Metas/limiares
RELIABILITY_TARGET = 0.99999
RELIABILITY_BASE = 0.4
# Gate (quase duro) para liberar termos secundários:
# - 0.0 em RELIABILITY_UNLOCK_START
# - 1.0 em RELIABILITY_UNLOCK_END (>=0.9999 libera tudo)
RELIABILITY_UNLOCK_START = 0.0
RELIABILITY_UNLOCK_END = 0.99999
THROUGHPUT_ORIGEM_MBPS = 100.0
THROUGHPUT_MAX_MBPS = 9600.0
HOPS_TARGET = 10
# ----------------------------
# Use exclusivamente FIT_WEIGHTS_DEFAULT para ajustar pesos do fitness.

# Latência “física” simplificada
MAX_LATENCY_MS = 100.0
LATENCY_PER_HOP_MS = 1.0
SPEED_M_PER_MS = 0.2  # 0.2 m/ms = 200 m/s (apenas para custo)

# =========================
# Pesos EFETIVOS do fitness (o que realmente entra no fit)
# Edite aqui (ou sobrescreva via DroneEnv(fit_weights={...}))
#
# ATENÇÃO: vários pesos no topo do arquivo são históricos/legados.
# O fitness (fit) usa estes pesos abaixo.
# =========================
FIT_WEIGHTS_DEFAULT = {
    # Confiabilidade principal: rel_score = REL_MAIN*(2*rel_log - 1)
    # onde rel_log = clamp((log10(reliability)+9)/9, 0..1).
    # Assim: rel_score ∈ [-REL_MAIN, +REL_MAIN]
    "REL_MAIN": 20_000.0,
    "REL_STEP": 0.0,  # 0 => confiabilidade grande só no TERMINAL (sem shaping denso no step)
    "TERM_REL": 20_000.0,  # bônus/punição terminal baseado na confiabilidade final (escala grande p/ refletir teste)
    "TERM_SUCCESS_BONUS": 3_000.0,  # bônus terminal ao atingir RELIABILITY_TARGET (0.99999) com zona limpa
    "TERM_ZONE_PEN": 10_000.0,  # penalidade terminal por violação de zona (por drone dentro + por link cruzando)

    # Penalização hard se NÃO existir caminho 0->N-1
    "PATH_MISSING": 3_000.0,

    # Penalização hard por violações de zona (multiplica total_violacoes = drones_in_zone + links_cross_zone)
    "ZONE_HARD": 3_000.0,

    # Zona (penalizações proporcionais por ocorrência)
    # (mantém o efetivo anterior: 20*100=2000 e 20*100.5=2010)
    "ZONE_DRONE_COEF": 2000.0,  # por drone dentro da zona
    "ZONE_LINK_COEF": 2010.0,  # por link que cruza a zona

    # Qualidade média de conexão (coeficiente pequeno)
    # "CONN_QUALITY_COEF": 0.0,
    "CONN_QUALITY_COEF": 400.0,

    # Hops (com gate): hops_score = HOPS_WEIGHT*hops_gate*hop_diff
    "HOPS_WEIGHT": 50.0,

    # Layout do caminho (quando path_exists)
    "LAYOUT_ALIGN_W": 200.0,
    "LAYOUT_STRAIGHT_W": 160.0,
    "LAYOUT_REDUND_W": 450.0,

    # Mistura do espalhamento: layout_terms = LAYOUT_MIX*(nn_reward + uniform_reward) + layout_score
    "LAYOUT_MIX": 2500.0,

    # Uniformidade 1D: nº de bins ao longo do segmento 0->N-1
    "UNIFORM1D_BINS": 280.0,
    # "UNIFORM1D_BINS": 0.0,

    # Puxa drones intermediários para perto da reta (drone0 -> droneN-1)
    # line_align_term = LINE_ALIGN_W * gate_zone * (1 - mean_perp_dist_norm)
    # onde mean_perp_dist_norm ∈ [0,1]
    # "LINE_ALIGN_W": 0.0,
    "LINE_ALIGN_W": 380.0,

    # Penalização por nós intermediários isolados no caminho (por nó)
    "ISOLATED_NODE_PENALTY": 4000.0,

    # Penalização por nós (exceto o 0) que NÃO são alcançáveis a partir do drone 0 (direta ou indiretamente)
    "UNREACHABLE_FROM_FIRST_PENALTY": 0.0,  # (LEGACY) penaliza por contagem; prefira REACH_FROM_FIRST_W
    "REACH_FROM_FIRST_W": 6500.0,  # penaliza (1 - reach_from_first_frac) ∈ [0,1]

    # Termos calculados no ambiente, mas DESABILITADOS no fit por padrão (peso 0.0).
    # Para ativar, defina valores > 0 via fit_weights.
    "LEFT_LINK_COEF": 1000.0,  # penalidade por link "para trás" (x_dst < x_src)

    # Penalidade forte se houver ciclo (loop) no grafo de links
    "CYCLE_PENALTY": 10_000.0,
    "THROUGHPUT_PENALTY_MAX": 500.0,  # penalização máxima quando throughput fica muito perto do máximo (ratio >= 0.85)
    "DISJOINT_PATH_BONUS": 6000.0,  # bônus por caminho disjunto adicional (k-1)
    "DISJOINT_PARTIAL_FACTOR": 0.5,  # fração do bônus para rotas paralelas PARCIAIS (>=2)
    "DISJOINT_PARTIAL_MIN_PROGRESS": 0.5,
    # progresso mínimo (0..1) ao longo do segmento 0->N-1 para contar um alvo parcial
    "DISJOINT_MIN_P": 0.5,  # prob mínima do enlace para contar no bônus de caminhos disjuntos (0 desativa)
    "DIRECT_0N1_MIN_P": 0.7,  # prob mínima para permitir link direto 0<->N-1 (0 desativa)

    "ZONE_FIX_LINK_BONUS": 600.0,  # bônus por link que DEIXA de cruzar zona (delta por step)
    "ZONE_FIX_DRONE_BONUS": 600.0, # bônus por drone que SAI da zona (delta por step)

    "ZONE_CLEAN_ONCE_BONUS": 2500.0, # bônus 1x quando zera violações de zona no episódio
    "INVALID_ACTION_PENALTY": 10.0,  # penalidade por step quando ação de link/move é inválida
    # ---- Bottleneck shaping (melhora o pior enlace no caminho 0->N-1) ----
    "BOTTLENECK_W": 3000.0,  # peso do termo de gargalo (escala ~ milhar)
    "BOTTLENECK_TARGET_P": 0.70,  # alvo de prob mínima no caminho (>= isto dá bônus)
    # ---- Disjoint gate (ativa bônus de rotas paralelas cedo, baseado em conectividade útil) ----
    "DISJOINT_GATE_RFF_START": 0.85,  # começa a ativar quando reach_from_first_frac >= isto
    "DISJOINT_GATE_RF_START": 0.85,  # começa a ativar quando reach_frac >= isto
}


class DroneEnv(gym.Env):
    """
    Ambiente de drones para PPO (Gymnasium). Observação composta por:
      - posições normalizadas (x,y) para cada drone,
      - matriz de adjacência achatada,
      - máscara de drones ativos (aqui sempre 1.0 para N drones).
    Ações (MultiDiscrete):
      [ move_idx (0..MAX_DRONES-1), step_code (0..3)->{1,2,3,4}, dir_code (0..4), toggle_op (0..(1+2*E-1)) ]
"""

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            grid_size=GRID_SIZE_DEFAULT,
            num_drones=13,
            max_out_connections=10,
            square_size_m=SQUARE_SIZE_M_DEFAULT,
            zonas_proibidas=None,
            p_sem_zona=0.0,
            variable_drones=True,
            min_drones=None,
            max_drones=None,
            eval_mode=False,
            # zonas parametrizáveis
            zonas_count=None,
            zona_size=None,  # (w, h)
            zona_single_fixed=False,
            # pesos do fit (efetivos) / logging detalhado do fit
            fit_weights=None,
            log_fit_terms=False,
            # ---- Observação extra (opcional) ----
            obs_zone_map=0,          # acrescenta um mapa (downsample) de zona proibida ao obs
            zone_map_scale=4,        # fator de downsample (ex.: 4 => 100x100 -> 25x25)
            obs_edge_cross=0,        # acrescenta matriz NxN indicando se a aresta (i,j) cruza zona
            # ---- Hard masks ----
            hard_mask_zone_edges=1,  # bloqueia ADD de arestas que cruzam zona (REMOVE permanece permitido)
            hard_mask_left_links=1,  # bloqueia ADD de links à esquerda (x_dst < x_src)
            hard_mask_cycles=1,      # bloqueia ADD de links que criariam ciclos (sem loops)
            hard_block_move_cross_zone=1,  # HARD: reverte movimento se AUMENTAR links que cruzam zona
            # ---- Currículo: "dirty reset" (começa sujo e termina 100% sujo) ----
            dirty_reset_prob_start=0.20,
            dirty_reset_prob_end=1.00,
            dirty_force_drone_in_zone_prob_start=0.20,
            dirty_force_drone_in_zone_prob_end=1.00,
            dirty_anneal_episodes=300,
    ):
        super().__init__()

        # ---------- parâmetros básicos ----------
        self.grid_size = int(grid_size)
        self.square_size_m = float(square_size_m)
        self._radio_params = _make_radio_params()  # usado para filtrar enlaces no bônus de caminhos disjuntos

        # Cache local de p(link) por distância (mesma semântica; evita recomputar erf/log10 repetidamente)
        self._prob_cache = {}
        self.max_out_connections = int(max_out_connections)

        self.variable_drones = bool(variable_drones)
        self.min_drones = int(min_drones) if min_drones is not None else None
        self.max_drones = int(max_drones) if max_drones is not None else None
        self.num_drones = int(num_drones)

        # Define max_drones efetivo usado para dimensionar a observation_space
        if self.max_drones is None:
            self.max_drones = self.num_drones
        # Garante limites [1, MAX_DRONES]
        self.max_drones = max(1, min(int(self.max_drones), MAX_DRONES))
        # Garante que num_drones não ultrapasse max_drones
        if self.num_drones > self.max_drones:
            self.num_drones = self.max_drones

        self.eval_mode = bool(eval_mode)

        # zonas
        self.zonas_proibidas_param = zonas_proibidas
        self.zonas_proibidas = set()
        # cache: grade booleana para consulta rápida de zona proibida (x,y)
        self._forbidden_grid = None
        self._zone_sig = 0  # deterministic signature of forbidden grid
        self.p_sem_zona = float(p_sem_zona)
        self.zonas_count = int(zonas_count) if zonas_count is not None else None
        self.zona_size = tuple(zona_size) if (zona_size and zona_size[0] and zona_size[1]) else None
        self.zona_single_fixed = bool(zona_single_fixed)

        # ---- Observação extra / hard masks ----
        self.obs_zone_map = bool(obs_zone_map)
        self.zone_map_scale = max(1, int(zone_map_scale))
        self.obs_edge_cross = bool(obs_edge_cross)
        self.hard_mask_zone_edges = bool(hard_mask_zone_edges)
        self.hard_mask_left_links = bool(hard_mask_left_links)
        self.hard_mask_cycles = bool(hard_mask_cycles)
        self.hard_block_move_cross_zone = bool(hard_block_move_cross_zone)

        # ---- Currículo: dirty reset ----
        self.dirty_reset_prob_start = float(dirty_reset_prob_start)
        self.dirty_reset_prob_end = float(dirty_reset_prob_end)
        self.dirty_force_drone_in_zone_prob_start = float(dirty_force_drone_in_zone_prob_start)
        self.dirty_force_drone_in_zone_prob_end = float(dirty_force_drone_in_zone_prob_end)
        self.dirty_anneal_episodes = max(1, int(dirty_anneal_episodes))
        self._reset_count = 0
        self._dirty_episode = False
        self._force_drone_in_zone = False

        # cache: zone_map (downsample) e edge_cross (matriz NxN)
        self._zone_map_down = None
        self._zone_map_down_sig = None
        self._edge_cross_mat = None
        self._edge_cross_n = None
        self._edge_cross_zone_sig = None


        # métricas correntes (para logging externo)
        self.current_reliability = 0.0
        self.current_latency_penalty = 0.0
        self.current_throughput = 0.0
        # ---------- espaços ----------
        # ação: 2 dimensões (move 1 drone por step + operação de link "achatada")
        # [move_op, toggle_op]
        #
        # move_op:
        #   0 -> não mover
        #   1 + move_idx*(STEP_CELLS_N*4) + step_code*4 + (dir_code-1)
        #     onde move_idx in [0..max_drones-1], step_code in [0..STEP_CELLS_N-1], dir_code in {1..4}
        #
        # toggle_op:
        #   0              -> não altera links
        #   1..E           -> ADD (src,dst) para cada par dirigido src!=dst (ordem fixa em self._edge_pairs)
        #   E+1..2E        -> REMOVE (src,dst)
        #
        # Isso permite action masking REAL (sem combinações inválidas src/dst).
        self._edge_pairs = [(i, j) for i in range(self.max_drones) for j in range(self.max_drones) if i != j]
        self._edge_pair_to_idx = {p: k for k, p in enumerate(self._edge_pairs)}
        self._E = len(self._edge_pairs)
        self._toggle_ops_n = 1 + 2 * self._E

        # Vetores prontos para indexação vetorizada no action_masks()
        # (evita loop Python por par (src,dst) a cada passo)
        self._edge_src = np.fromiter((p[0] for p in self._edge_pairs), dtype=np.int16, count=self._E)
        self._edge_dst = np.fromiter((p[1] for p in self._edge_pairs), dtype=np.int16, count=self._E)

        self._move_ops_n = int(1 + int(self.max_drones) * int(STEP_CELLS_N) * 4)
        self.action_space = spaces.MultiDiscrete(
            np.array([self._move_ops_n, self._toggle_ops_n], dtype=np.int64)
        )

        # cache simples para action_masks (evita recomputar quando estado não muda)
        self._mask_version = 0
        self._mask_cache_version = -1
        self._mask_cache = None
        # observação: base FIXA com padding até max_drones.
        # Partes:
        #   - pos (max_drones x 2) normalizado em [-1, 1]
        #   - adj (max_drones x max_drones) em {0,1}
        #   - mask (max_drones) em {0,1} indicando drones ativos
        # Extras opcionais (via args):
        #   - zone_map (downsample do grid de zonas proibidas)
        #   - edge_cross (matriz NxN: 1 se o segmento (i,j) cruza zona)
        obs_dim_pos = self.max_drones * 2
        obs_dim_adj = self.max_drones * self.max_drones
        obs_dim_mask = self.max_drones

        # zone-map (downsample)
        if self.obs_zone_map:
            self._zone_map_ds = int(math.ceil(float(self.grid_size) / float(self.zone_map_scale)))
            obs_dim_zone = int(self._zone_map_ds * self._zone_map_ds)
        else:
            self._zone_map_ds = 0
            obs_dim_zone = 0

        # edge-cross (NxN)
        obs_dim_edge_cross = int(self.max_drones * self.max_drones) if self.obs_edge_cross else 0

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(obs_dim_pos + obs_dim_adj + obs_dim_mask + obs_dim_zone + obs_dim_edge_cross,),
            dtype=np.float32
        )

        # ---------- estado ----------
        self._rng = np.random.RandomState()
        self._dist_matrix = None
        self._dist_pos_sig = None
        self._bonus_sucesso_dado = False
        # ---------- pesos efetivos do fitness ----------
        self.fit_w = dict(FIT_WEIGHTS_DEFAULT)
        if fit_weights:
            # sobrescrita parcial, ex: {"REL_MAIN": 800.0}
            self.fit_w.update({k: float(v) for k, v in fit_weights.items()})
        # REL_STEP: por padrão é 10% de REL_MAIN (reduz penalidade/recompensa densa por confiabilidade no step)
        if self.fit_w.get("REL_STEP", None) is None:
            self.fit_w["REL_STEP"] = 0.1 * float(self.fit_w.get("REL_MAIN", 0.0))

        self.log_fit_terms = bool(log_fit_terms)

        self._reset_zonas()
        self._init_state()

        self._last_reward = 0.0
        # Tracking para shaping de zona (delta de violações e bônus 1x quando limpa)
        self._prev_links_cross_zone = None
        self._prev_drones_in_zone = None
        self._zone_clean_once = False

        self._last_reward = 0.0
        self._episode_steps = 0
        self._max_episode_steps = 2000

        self._success_streak = 0  # passos consecutivos em sucesso
        self._success_need = 1

    # =========================
    # Zonas proibidas
    # =========================
    def _reset_zonas(self):
        """(Re)gera as zonas proibidas e reconstrói a grade booleana de consulta rápida.

        Importante: mantém a semântica original:
        - Se zonas_proibidas_param foi fornecido: usa exatamente esse conjunto.
        - Caso contrário: amostra proceduralmente (p_sem_zona, count, tamanho, etc.).
        """
        self.zonas_proibidas.clear()

        # Caso 1: zonas fixas fornecidas externamente
        if self.zonas_proibidas_param is not None:
            for (x, y) in self.zonas_proibidas_param:
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    self.zonas_proibidas.add((x, y))
            self._rebuild_forbidden_grid()
            return

        # Caso 2: chance de não ter zona (modo procedural)
        if self.p_sem_zona > 0.0 and self._rng.rand() < self.p_sem_zona:
            self._rebuild_forbidden_grid()
            return

        # contagem e tamanho
        count = self.zonas_count if self.zonas_count is not None else self._rng.randint(0, 3)
        if count <= 0:
            self._rebuild_forbidden_grid()
            return

        if self.zona_size:
            w, h = map(int, self.zona_size)
        else:
            w, h = max(2, self.grid_size // 12), max(2, self.grid_size // 12)

        if self.zona_single_fixed:
            # única zona central fixa
            cx, cy = self.grid_size // 2, self.grid_size // 2
            x0 = max(0, cx - w // 2)
            y0 = max(0, cy - h // 2)
            for xx in range(x0, min(self.grid_size, x0 + w)):
                for yy in range(y0, min(self.grid_size, y0 + h)):
                    self.zonas_proibidas.add((xx, yy))
            self._rebuild_forbidden_grid()
            return

        # múltiplas zonas aleatórias
        for _ in range(count):
            x0 = self._rng.randint(0, max(1, self.grid_size - w))
            y0 = self._rng.randint(0, max(1, self.grid_size - h))
            for xx in range(x0, min(self.grid_size, x0 + w)):
                for yy in range(y0, min(self.grid_size, y0 + h)):
                    self.zonas_proibidas.add((xx, yy))

        self._rebuild_forbidden_grid()

    def _rebuild_forbidden_grid(self):
        """Reconstrói uma máscara booleana [grid_size x grid_size] com True nas células proibidas.

        - Mantém a representação original em set (self.zonas_proibidas)
        - Apenas acelera consultas repetidas no hot-path (step/fit)
        """
        if not self.zonas_proibidas:
            self._forbidden_grid = None
            self._zone_sig = 0
            # mantém shapes de observação consistentes mesmo sem zona
            self._rebuild_zone_map_cache()
            self._invalidate_edge_cross_cache()
            return
        g = np.zeros((self.grid_size, self.grid_size), dtype=np.bool_)
        for (x, y) in self.zonas_proibidas:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                g[int(x), int(y)] = True
        self._forbidden_grid = g
        self._zone_sig = int(zlib.crc32(g.tobytes()))
        # atualiza caches dependentes de zona
        self._rebuild_zone_map_cache()
        self._invalidate_edge_cross_cache()


    # =========================
    # Caches auxiliares: zone_map / edge_cross
    # =========================

    def _rebuild_zone_map_cache(self):
        """Recalcula o mapa (downsample) da zona proibida usado em obs_zone_map."""
        if not getattr(self, "obs_zone_map", False):
            self._zone_map_down = None
            self._zone_map_down_sig = None
            return

        ds = int(getattr(self, "_zone_map_ds", 0) or 0)
        if ds <= 0:
            self._zone_map_down = np.zeros((0,), dtype=np.float32)
            self._zone_map_down_sig = None
            return

        # Se não existe zona neste episódio, ainda assim retornamos um vetor
        # de tamanho FIXO (ds*ds) cheio de zeros para manter obs consistente.
        if self._forbidden_grid is None:
            self._zone_map_down = np.zeros((ds * ds,), dtype=np.float32)
            self._zone_map_down_sig = (int(getattr(self, "_zone_sig", 0)) ^ (int(self.zone_map_scale) << 16) ^ (ds << 1)) & 0xFFFFFFFF
            return

        scale = max(1, int(self.zone_map_scale))
        g = self._forbidden_grid.astype(bool, copy=False)  # (grid,grid) em [x,y]

        # padding para múltiplo de scale
        target = ds * scale
        pad_x = max(0, target - g.shape[0])
        pad_y = max(0, target - g.shape[1])
        if pad_x or pad_y:
            g = np.pad(g, ((0, pad_x), (0, pad_y)), mode="constant", constant_values=False)

        # reshape -> (ds, scale, ds, scale) e faz OR por bloco
        try:
            g4 = g.reshape(ds, scale, ds, scale)
            down = g4.any(axis=(1, 3)).astype(np.float32, copy=False)
        except Exception:
            # fallback seguro
            down = np.zeros((ds, ds), dtype=np.float32)
            for ix in range(ds):
                for iy in range(ds):
                    x0 = ix * scale
                    y0 = iy * scale
                    block = g[x0:x0 + scale, y0:y0 + scale]
                    down[ix, iy] = 1.0 if np.any(block) else 0.0

        self._zone_map_down = down.reshape(-1).astype(np.float32, copy=False)
        # assinatura: zona + scale
        self._zone_map_down_sig = (int(getattr(self, "_zone_sig", 0)) ^ (scale << 16) ^ (ds << 1)) & 0xFFFFFFFF

    def _invalidate_edge_cross_cache(self):
        """Invalida caches que dependem de zona/posições (edge_cross)."""
        self._edge_cross_mat = None
        self._edge_cross_n = None
        self._edge_cross_zone_sig = None

    def _ensure_edge_cross_mat(self):
        """Garante self._edge_cross_mat (NxN) para o número atual de drones.

        edge_cross_mat[i,j] = 1 se o segmento entre i e j cruza zona proibida.
        """
        if not (getattr(self, "obs_edge_cross", False) or getattr(self, "hard_mask_zone_edges", False)):
            return None

        n = int(self.num_drones)
        if n <= 1:
            self._edge_cross_mat = np.zeros((n, n), dtype=np.uint8)
            self._edge_cross_n = n
            self._edge_cross_zone_sig = int(getattr(self, "_zone_sig", 0))
            return self._edge_cross_mat

        if not getattr(self, "zonas_proibidas", None):
            self._edge_cross_mat = np.zeros((n, n), dtype=np.uint8)
            self._edge_cross_n = n
            self._edge_cross_zone_sig = int(getattr(self, "_zone_sig", 0))
            return self._edge_cross_mat

        zone_sig = int(getattr(self, "_zone_sig", 0))
        if (self._edge_cross_mat is None) or (self._edge_cross_n != n) or (self._edge_cross_zone_sig != zone_sig):
            mat = np.zeros((n, n), dtype=np.uint8)
            pos_np = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2]).astype(np.float32, copy=False)

            # calcula apenas triângulo superior e espelha (simétrico)
            for i in range(n):
                xi = int(round(float(pos_np[i, 0])))
                yi = int(round(float(pos_np[i, 1])))
                p1 = (xi, yi)
                for j in range(i + 1, n):
                    xj = int(round(float(pos_np[j, 0])))
                    yj = int(round(float(pos_np[j, 1])))
                    p2 = (xj, yj)
                    cross = self._linha_passa_por_zona(p1, p2, (i, j))
                    v = 1 if cross else 0
                    mat[i, j] = v
                    mat[j, i] = v

            self._edge_cross_mat = mat
            self._edge_cross_n = n
            self._edge_cross_zone_sig = zone_sig

        return self._edge_cross_mat

    def _update_edge_cross_after_move(self, moved_idx: int):
        """Atualiza incrementalmente a matriz edge_cross quando 1 drone move."""
        if not (getattr(self, "obs_edge_cross", False) or getattr(self, "hard_mask_zone_edges", False)):
            return

        n = int(self.num_drones)
        if n <= 1:
            self._edge_cross_mat = np.zeros((n, n), dtype=np.uint8)
            self._edge_cross_n = n
            self._edge_cross_zone_sig = int(getattr(self, "_zone_sig", 0))
            return

        if not getattr(self, "zonas_proibidas", None):
            self._edge_cross_mat = np.zeros((n, n), dtype=np.uint8)
            self._edge_cross_n = n
            self._edge_cross_zone_sig = int(getattr(self, "_zone_sig", 0))
            return

        moved_idx = int(moved_idx)
        if moved_idx < 0 or moved_idx >= n:
            return

        # se cache inválido (n mudou ou zona mudou), recalcula tudo
        zone_sig = int(getattr(self, "_zone_sig", 0))
        if (self._edge_cross_mat is None) or (self._edge_cross_n != n) or (self._edge_cross_zone_sig != zone_sig):
            self._ensure_edge_cross_mat()
            return

        pos_np = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2]).astype(np.float32, copy=False)
        xi = int(round(float(pos_np[moved_idx, 0])))
        yi = int(round(float(pos_np[moved_idx, 1])))
        p1 = (xi, yi)

        for j in range(n):
            if j == moved_idx:
                self._edge_cross_mat[moved_idx, j] = 0
                continue
            xj = int(round(float(pos_np[j, 0])))
            yj = int(round(float(pos_np[j, 1])))
            p2 = (xj, yj)
            cross = self._linha_passa_por_zona(p1, p2, (moved_idx, j))
            v = 1 if cross else 0
            self._edge_cross_mat[moved_idx, j] = v
            self._edge_cross_mat[j, moved_idx] = v




    # =========================
    # Estado inicial

    # =========================
    # Estado inicial
    # =========================
    def _init_state(self):
        """
        Inicializa o estado:
        - Drone 0 na coluna 0 (x=0), linha central (y≈grid/2) — âncora (estilo drone_env2).
        - Último drone na última coluna (x=grid-1), linha central — âncora (estilo drone_env2).
        - Drones intermediários em posições aleatórias (podem cair na zona proibida).
        - Matriz de adjacência inicial com:
            * em pelo menos 90% dos resets, existência de rota 0 -> ... -> N-1;
            * pelo menos um link incidente (entrada ou saída) em cada nó.
        A rota preferencialmente é "limpa", sem links à esquerda nem cruzando zona proibida.
        """
        center_y = self.grid_size // 2
        last_col = self.grid_size - 1

        dirty_episode = bool(getattr(self, "_dirty_episode", False))

        pos = np.zeros((self.num_drones, 2), dtype=np.float32)

        def _empurra_para_fora_da_zona(x: int, y: int) -> (int, int):
            """
            Se (x,y) estiver em zona proibida, tenta deslocar horizontalmente
            para direita e depois para esquerda até sair.
            Mantém a linha (y).
            """
            if (x, y) not in self.zonas_proibidas:
                return x, y
            # tenta para a direita
            nx = x
            while (nx, y) in self.zonas_proibidas and nx < self.grid_size - 1:
                nx += 1
            if (nx, y) not in self.zonas_proibidas:
                return nx, y
            # tenta para a esquerda
            nx = x
            while (nx, y) in self.zonas_proibidas and nx > 0:
                nx -= 1
            return nx, y

        # Drone 0: coluna esquerda (x=0), linha central (y=center_y) — âncora (estilo drone_env2)
        pos[0, 0] = 0.0
        pos[0, 1] = float(center_y)

        # Drone N-1: coluna direita (x=grid-1), linha central — âncora (estilo drone_env2)
        if self.num_drones > 1:
            pos[self.num_drones - 1, 0] = float(last_col)
            pos[self.num_drones - 1, 1] = float(center_y)

        # Drones intermediários: posições aleatórias (podem cair em zona proibida) e sem colisão
        for i in range(1, self.num_drones - 1):
            for _ in range(1000):
                rx = int(self._rng.randint(0, self.grid_size))
                ry = int(self._rng.randint(0, self.grid_size))
                # evita colisão com drones já posicionados
                collision = False
                for j in range(i):
                    if int(pos[j, 0]) == rx and int(pos[j, 1]) == ry:
                        collision = True
                        break
                if collision:
                    continue
                # em episódios "limpos", evita posicionar drones na zona proibida
                if (not dirty_episode) and ((rx, ry) in self.zonas_proibidas):
                    continue
                pos[i, 0] = float(rx)
                pos[i, 1] = float(ry)
                break
            else:
                # fallback (improvável): coloca na posição do drone 0
                pos[i, 0] = pos[0, 0]
                pos[i, 1] = pos[0, 1]


        # Força (currículo) pelo menos 1 drone intermediário dentro da zona proibida nos episódios "sujos"
        if dirty_episode and getattr(self, "_force_drone_in_zone", False) and self.zonas_proibidas and self.num_drones > 2:
            # escolhe um drone intermediário aleatório
            k = int(self._rng.randint(1, self.num_drones - 1))
            # tenta escolher uma célula dentro da zona que não colida
            zone_cells = list(self.zonas_proibidas)
            self._rng.shuffle(zone_cells)
            chosen = None
            for (zx, zy) in zone_cells:
                # evita colisão com outros drones
                ok = True
                for j in range(self.num_drones):
                    if j == k:
                        continue
                    if int(round(float(pos[j, 0]))) == int(zx) and int(round(float(pos[j, 1]))) == int(zy):
                        ok = False
                        break
                if ok:
                    chosen = (zx, zy)
                    break
            if chosen is None and zone_cells:
                chosen = zone_cells[0]
            if chosen is not None:
                pos[k, 0] = float(chosen[0])
                pos[k, 1] = float(chosen[1])


        # Salva posições no tensor
        self.drone_positions = torch.tensor(pos, dtype=torch.float32, device=DEVICE)

        # ---------- Construção da matriz de adjacência ----------
        n = self.num_drones
        self.adjacency_matrix = torch.zeros(
            (n, n),
            dtype=torch.float32,
            device=DEVICE,
        )

        if n > 1:
            pos_np = _torch_to_numpy_cpu_view(self.drone_positions)

            def _cell_in_zone(idx: int) -> bool:
                xi = int(round(pos_np[idx, 0]))
                yi = int(round(pos_np[idx, 1]))
                return (xi, yi) in self.zonas_proibidas

            # Edges "limpos" (sem left-link e sem cruzar zona) usados para montar a rota principal
            def _edge_ok_path(i: int, j: int) -> bool:
                if i == j:
                    return False
                # extremos fora da zona para a rota principal
                if _cell_in_zone(i) or _cell_in_zone(j):
                    return False
                # evita links à esquerda
                if pos_np[j, 0] < pos_np[i, 0]:
                    return False
                # Evita, por padrão, o link direto entre o primeiro e o último drone quando ele é fisicamente ruim.
                # Isso reduz a incidência de 0->N-1 “artificial” no reset (especialmente quando x0==xN-1).
                p_min_direct = float(self.fit_w.get("DIRECT_0N1_MIN_P", 0.0))
                if p_min_direct > 0.0 and n > 2:
                    if (i == 0 and j == n - 1) or (i == n - 1 and j == 0):
                        d_m = float(np.linalg.norm((pos_np[j] - pos_np[i]) * self.square_size_m))
                        p_dir = float(self._prob_link(d_m))
                        if p_dir < p_min_direct:
                            return False
                # evita cruzar zona proibida
                p1 = (int(round(pos_np[i, 0])), int(round(pos_np[i, 1])))
                p2 = (int(round(pos_np[j, 0])), int(round(pos_np[j, 1])))
                if self._linha_passa_por_zona(p1, p2, (i, j)):
                    return False
                return True

            # Pré-computa arestas válidas para a rota
            valid_edges = [[False] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if _edge_ok_path(i, j):
                        valid_edges[i][j] = True

            path_nodes = None
            # Em ~95% dos resets, tenta construir rota LIMPA 0->...->N-1 (sem cruzar zona)
            if self._rng.rand() < 0.95:
                prev = [-1] * n
                q = deque()
                q.append(0)
                prev[0] = 0
                while q:
                    u = q.popleft()
                    if u == n - 1:
                        break
                    for v in range(n):
                        if valid_edges[u][v] and prev[v] == -1:
                            prev[v] = u
                            q.append(v)

                if prev[n - 1] != -1:
                    # Reconstrói caminho 0 -> ... -> N-1
                    rev_path = [n - 1]
                    while rev_path[-1] != 0:
                        rev_path.append(prev[rev_path[-1]])
                    path_nodes = list(reversed(rev_path))

            # Fallback: se não achou rota "limpa", constrói uma rota monotônica em X (estilo drone_env2)
            if path_nodes is None:

                def _nearest_right_neighbor(i: int) -> int:
                    xi = float(pos_np[i, 0])
                    cand = [j for j in range(n) if float(pos_np[j, 0]) > xi and j != 0]
                    if not cand:
                        return n - 1

                    # Se houver zona proibida, tenta preferir um próximo salto que NÃO cruza a zona.
                    if self.zonas_proibidas and dirty_episode:
                        cand_ok = [j for j in cand if _edge_ok_path(i, j)]
                        if cand_ok:
                            cand = cand_ok

                    dists = [float(np.linalg.norm(pos_np[j] - pos_np[i])) for j in cand]
                    return cand[int(np.argmin(dists))]

                cur = 0
                visited = {cur}
                path_nodes = [cur]
                while cur != n - 1:
                    nxt = _nearest_right_neighbor(cur)
                    if nxt in visited:
                        nxt = n - 1
                    path_nodes.append(nxt)
                    visited.add(nxt)
                    cur = nxt

            # Cria links da rota principal
            for a, b in zip(path_nodes[:-1], path_nodes[1:]):
                if a != b:
                    # respeita max_out_connections na origem
                    out_deg = int((self.adjacency_matrix[a, :] > 0).sum().item())
                    if out_deg < self.max_out_connections:
                        self.adjacency_matrix[a, b] = 1.0

            # ---------- Semear links redundantes no reset ----------
            # Estratégia (para aprender a REMOVER links proibidos):
            # 1) Mantém o backbone 0->...->N-1 gerado acima (preferencialmente limpo).
            # 2) Adiciona POUCOS links extras "bons" (não cruzam zona, quando houver zona).
            # 3) Adiciona ALGUNS links extras "ruins" que cruzam a zona (quando houver zona),
            #    para o agente aprender a removê-los, sem depender do backbone.

            adj_np = _torch_to_numpy_cpu_view(self.adjacency_matrix)
            pos_np = _torch_to_numpy_cpu_view(self.drone_positions)

            def _reachable(adj: np.ndarray, src: int, dst: int) -> bool:
                if src == dst:
                    return True
                visited = {src}
                stack = [src]
                while stack:
                    u = stack.pop()
                    nbrs = np.where(adj[u] > 0.0)[0]
                    for v in nbrs:
                        v = int(v)
                        if v == dst:
                            return True
                        if v not in visited:
                            visited.add(v)
                            stack.append(v)
                return False

            def _can_add_edge(src: int, dst: int, require_no_zone: bool) -> bool:
                if src == dst:
                    return False
                if adj_np[src, dst] > 0.0:
                    return False
                # respeita max_out_connections
                out_deg = int(np.sum(adj_np[src, :] > 0.0))
                if out_deg >= int(self.max_out_connections):
                    return False
                # evita links para trás em X
                if float(pos_np[dst, 0]) < float(pos_np[src, 0]):
                    return False
                # Evita, por padrão, semear link direto 0<->N-1 quando ele é fisicamente ruim.
                p_min_direct = float(self.fit_w.get("DIRECT_0N1_MIN_P", 0.0))
                if p_min_direct > 0.0 and n > 2:
                    if (src == 0 and dst == n - 1) or (src == n - 1 and dst == 0):
                        d_m = float(np.linalg.norm((pos_np[dst] - pos_np[src]) * self.square_size_m))
                        p_dir = float(self._prob_link(d_m))
                        if p_dir < p_min_direct:
                            return False
                # evita ciclos
                if _reachable(adj_np, dst, src):
                    return False
                # opcionalmente exige não cruzar zona
                if require_no_zone and self.zonas_proibidas:
                    p1 = (int(round(pos_np[src, 0])), int(round(pos_np[src, 1])))
                    p2 = (int(round(pos_np[dst, 0])), int(round(pos_np[dst, 1])))
                    if self._linha_passa_por_zona(p1, p2, (src, dst)):
                        return False
                return True

            # --- (2) Links extras "bons" ---
            # Limita a quantidade para não criar dezenas de violações/ruído no início.
            # max_good_edges = max(1, n // 2)
            max_good_edges = max(1, n)
            added_good = 0
            for _ in range(max_good_edges * 10):
                if added_good >= max_good_edges:
                    break
                src = int(self._rng.randint(0, n - 1))
                # candidatos à direita (mantém aleatoriedade, mas escolhe os mais curtos para evitar links gigantes)
                cand = [j for j in range(n) if j != src and float(pos_np[j, 0]) >= float(pos_np[src, 0])]
                self._rng.shuffle(cand)
                # amostra um subconjunto e escolhe os mais próximos primeiro (reduz incidência de link direto 0->N-1)
                cand = cand[: min(16, len(cand))]
                cand.sort(key=lambda j: ((float(pos_np[src, 0]) - float(pos_np[j, 0])) * self.square_size_m) ** 2 +
                                        ((float(pos_np[src, 1]) - float(pos_np[j, 1])) * self.square_size_m) ** 2)
                for dst in cand:
                    dst = int(dst)
                    if _can_add_edge(src, dst, require_no_zone=True):
                        adj_np[src, dst] = 1.0
                        added_good += 1
                        break

            # --- (3) Links extras "ruins" (cruzam zona) ---
            added_bad = 0
            if self.zonas_proibidas and dirty_episode:
                bad_candidates = []
                for src in range(n - 1):
                    for dst in range(n):
                        if src == dst:
                            continue
                        if not _can_add_edge(int(src), int(dst), require_no_zone=False):
                            continue
                        p1 = (int(round(pos_np[src, 0])), int(round(pos_np[src, 1])))
                        p2 = (int(round(pos_np[dst, 0])), int(round(pos_np[dst, 1])))
                        if self._linha_passa_por_zona(p1, p2, (int(src), int(dst))):
                            bad_candidates.append((int(src), int(dst)))

                self._rng.shuffle(bad_candidates)
                # poucos links ruins (suficiente para aprender a remover)
                k_bad = min(len(bad_candidates), max(1, n // 4))
                for src, dst in bad_candidates[:k_bad]:
                    # tenta não estourar out-degree
                    if int(np.sum(adj_np[src, :] > 0.0)) >= int(self.max_out_connections):
                        continue
                    adj_np[src, dst] = 1.0
                    added_bad += 1

            self.adjacency_matrix = torch.tensor(adj_np, dtype=torch.float32, device=DEVICE)

        self._bonus_sucesso_dado = False
        self._dist_matrix = None
        self._dist_pos_sig = None

        # action_masks cache inválido após reset
        if hasattr(self, "_mask_version"):
            self._mask_version += 1
            self._mask_cache = None
            self._mask_cache_version = -1

    # =========================
    # Helpers
    # =========================
    def _calc_distance_matrix(self):
        # cache: evita recomputar O(N²) sem mudanças de posição
        n = int(self.num_drones)
        pos = np.ascontiguousarray(_torch_to_numpy_cpu_view(self.drone_positions[:n, :2]), dtype=np.float32)
        # assinatura robusta para evitar colisões (soma de coords pode colidir)
        sig = (int(zlib.crc32(pos.tobytes())), int(pos.shape[0]))
        if self._dist_pos_sig == sig and self._dist_matrix is not None:
            return self._dist_matrix

        # versão vetorizada (mesma semântica, muito mais rápida que loop Python)
        px = (pos[:, 0] * float(self.square_size_m)).astype(np.float32, copy=False)
        py = (pos[:, 1] * float(self.square_size_m)).astype(np.float32, copy=False)
        dx = px[:, None] - px[None, :]
        dy = py[:, None] - py[None, :]
        d = np.sqrt(dx * dx + dy * dy).astype(np.float32, copy=False)

        self._dist_matrix = d
        self._dist_pos_sig = sig
        return d

    def _prob_link(self, d_m: float) -> float:
        """P(link) com cache por distância.

        NÃO altera semântica: apenas evita recomputar prob_link_literature
        repetidamente para as mesmas distâncias (hot-path em fitness/máscara).
        """
        d = float(d_m)
        if d <= 0.0:
            d = 1e-6
        pc = self._prob_cache.get(d)
        if pc is None:
            pc = float(prob_link_literature(d, self._radio_params))
            self._prob_cache[d] = pc
            # proteção simples contra crescimento patológico
            if len(self._prob_cache) > 200000:
                self._prob_cache.clear()
        return float(pc)

    def _has_path(self, src, dst, adj, out_neighbors=None):
        """BFS para verificar existência de caminho src->dst.

        out_neighbors (opcional): lista de vizinhos de saída por nó.
        Se fornecida, evita varrer range(n) (mais rápido) sem mudar a lógica.
        """
        n = int(adj.shape[0])
        if src < 0 or dst < 0 or src >= n or dst >= n:
            return False
        if src == dst:
            return True

        visited = np.zeros(n, dtype=bool)
        queue = deque([int(src)])
        visited[int(src)] = True
        if out_neighbors is None:
            while queue:
                u = queue.popleft()
                if u == dst:
                    return True
                for v in range(n):
                    if adj[u, v] > 0 and not visited[v]:
                        visited[v] = True
                        queue.append(v)
            return False

        # versão usando lista de adjacência
        while queue:
            u = queue.popleft()
            if u == dst:
                return True
            for v in out_neighbors[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return False


    def _get_any_path_nodes(self, src: int, dst: int, adj_np, out_neighbors=None):
        """Retorna uma rota (lista de nós) src->dst via BFS em arestas dirigidas.
        Se não houver caminho, retorna lista vazia.

        out_neighbors (opcional): lista de vizinhos de saída por nó.
        """
        n = int(adj_np.shape[0])
        if src < 0 or dst < 0 or src >= n or dst >= n:
            return []
        if src == dst:
            return [int(src)]

        parent = [-1] * n
        q = deque([int(src)])
        parent[int(src)] = int(src)

        if out_neighbors is None:
            while q:
                u = q.popleft()
                if u == dst:
                    break
                for v in range(n):
                    if adj_np[u, v] > 0.0 and parent[v] == -1:
                        parent[v] = u
                        q.append(v)
        else:
            while q:
                u = q.popleft()
                if u == dst:
                    break
                for v in out_neighbors[u]:
                    if parent[v] == -1:
                        parent[v] = u
                        q.append(v)

        if parent[int(dst)] == -1:
            return []
        path = [int(dst)]
        cur = int(dst)
        while cur != int(src):
            cur = parent[cur]
            if cur == -1:
                return []
            path.append(cur)
        path.reverse()
        return path


    def _floyd_warshall(self, dist):
        return floyd_warshall(dist, directed=True)

    def _linha_passa_por_zona(self, p1, p2, link=None):
        """Verifica se o segmento (p1->p2) passa por alguma célula proibida.

        Mantém a mesma semântica original (amostragem discreta com round),
        mas faz a verificação de forma vetorizada e, quando disponível,
        usando uma grade booleana (self._forbidden_grid) para acelerar.
        """
        if not self.zonas_proibidas:
            return False
        x1, y1 = p1
        x2, y2 = p2
        steps = int(max(abs(x2 - x1), abs(y2 - y1)) + 1)
        if steps <= 1:
            return False

        # amostragem idêntica (round/banker's rounding)
        t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        xs = np.rint(x1 + t * (x2 - x1)).astype(np.int32, copy=False)
        ys = np.rint(y1 + t * (y2 - y1)).astype(np.int32, copy=False)

        # proteção por limites
        xs = np.clip(xs, 0, self.grid_size - 1)
        ys = np.clip(ys, 0, self.grid_size - 1)

        if self._forbidden_grid is not None:
            return bool(self._forbidden_grid[xs, ys].any())

        # fallback (caso raro): usa o set original
        for x, y in zip(xs.tolist(), ys.tolist()):
            if (int(x), int(y)) in self.zonas_proibidas:
                return True
        return False


    # =========================
    # Gym API

    # =========================
    # Gym API
    # =========================
    def _filter_adj_by_disjoint_quality(self, adj_bin: np.ndarray, dist_m: np.ndarray, p_min: float) -> np.ndarray:
        """Filtra arestas para o bônus de caminhos disjuntos.
        Remove arestas com prob_link(d) < p_min. p_min<=0 desativa.

        Otimização (sem mudar semântica): percorre apenas as arestas existentes (O(E))
        e usa cache local de probabilidade.
        """
        if p_min <= 0.0:
            return adj_bin

        out = adj_bin.copy()
        ei, ej = np.nonzero(out > 0.0)
        if ei.size == 0:
            return out

        # percorre apenas as arestas existentes
        for i, j in zip(ei.tolist(), ej.tolist()):
            d = float(dist_m[int(i), int(j)])
            if d <= 0.0:
                continue
            if self._prob_link(d) < float(p_min):
                out[int(i), int(j)] = 0.0
        return out

    def _max_edge_disjoint_paths_cap(self, cap: np.ndarray, src: int, dst: int) -> int:
        """Max-flow (capacidade unitária) para contar caminhos edge-disjoint.

        Substitui Edmonds-Karp por Dinic (mesma resposta lógica, bem mais rápido em grafos mais densos).
        `cap` NÃO precisa ser preservado; por compatibilidade, tratamos como entrada somente.
        """
        n = int(cap.shape[0])
        src = int(src)
        dst = int(dst)
        if n <= 1 or src == dst:
            return 0

        # Dinic (grafos pequenos: implementação Python simples)
        class _Edge:
            __slots__ = ("to", "rev", "cap")
            def __init__(self, to: int, rev: int, cap: int):
                self.to = to
                self.rev = rev
                self.cap = cap

        g = [[] for _ in range(n)]

        def add_edge(fr: int, to: int, c: int):
            g[fr].append(_Edge(to, len(g[to]), c))
            g[to].append(_Edge(fr, len(g[fr]) - 1, 0))

        # monta grafo a partir da matriz de capacidades
        ei, ej = np.nonzero(cap > 0)
        for u, v in zip(ei.tolist(), ej.tolist()):
            add_edge(int(u), int(v), int(cap[int(u), int(v)]))

        flow = 0
        INF = 10 ** 9

        while True:
            level = [-1] * n
            q = deque([src])
            level[src] = 0
            while q:
                v = q.popleft()
                for e in g[v]:
                    if e.cap > 0 and level[e.to] < 0:
                        level[e.to] = level[v] + 1
                        q.append(e.to)
            if level[dst] < 0:
                break

            it = [0] * n

            def dfs(v: int, f: int) -> int:
                if v == dst:
                    return f
                i = it[v]
                while i < len(g[v]):
                    e = g[v][i]
                    if e.cap > 0 and level[e.to] == level[v] + 1:
                        ret = dfs(e.to, f if f < e.cap else e.cap)
                        if ret:
                            e.cap -= ret
                            g[e.to][e.rev].cap += ret
                            return ret
                    i += 1
                    it[v] = i
                return 0

            while True:
                pushed = dfs(src, INF)
                if not pushed:
                    break
                flow += pushed

        return int(flow)

    def _max_edge_disjoint_paths_no_zone_links(
        self,
        adj_np: np.ndarray,
        pos_np: np.ndarray,
        src: int,
        dst: int,
        pos_round: np.ndarray = None,
        zone_edge_cache: dict = None,
    ) -> int:
        """Conta caminhos edge-disjoint SOMENTE se NÃO houver nenhum link que cruze zona proibida.
        Regra pedida: se existir ao menos 1 link cruzando zona, retorna 0.

        Otimizações (sem mudar semântica):
          - usa `pos_round` pré-computado quando fornecido
          - usa cache local por endpoints (x1,y1,x2,y2) para evitar recomputar _linha_passa_por_zona
        """
        n = int(adj_np.shape[0])
        if n <= 1:
            return 0
        src = int(src)
        dst = int(dst)
        if src == dst:
            return 0

        # Se não há zonas, calcula direto
        if not self.zonas_proibidas:
            cap = (adj_np > 0.0).astype(np.int32)
            return self._max_edge_disjoint_paths_cap(cap, src, dst)

        if pos_round is None:
            pos_round = np.rint(pos_np[:, :2]).astype(np.int32, copy=False)
        if zone_edge_cache is None:
            zone_edge_cache = {}

        def edge_crosses(i: int, j: int) -> bool:
            x1 = int(pos_round[i, 0]); y1 = int(pos_round[i, 1])
            x2 = int(pos_round[j, 0]); y2 = int(pos_round[j, 1])
            k = (x1, y1, x2, y2)
            v = zone_edge_cache.get(k)
            if v is None:
                v = bool(self._linha_passa_por_zona((x1, y1), (x2, y2), (i, j)))
                zone_edge_cache[k] = v
            return v

        # Se existir QUALQUER link que cruze zona, desqualifica
        ei, ej = np.nonzero(adj_np > 0.0)
        for i, j in zip(ei.tolist(), ej.tolist()):
            if edge_crosses(int(i), int(j)):
                return 0

        cap = (adj_np > 0.0).astype(np.int32)
        return self._max_edge_disjoint_paths_cap(cap, src, dst)

    def _max_edge_disjoint_partial_paths(self, adj_np: np.ndarray, pos_np: np.ndarray, src: int, dst: int,
                                         min_progress: float = 0.5) -> int:
        """Conta o número máximo de caminhos edge-disjoint *parciais* saindo de src em direção a dst.

        Ideia:
          - Define um conjunto de "alvos parciais" = nós alcançáveis a partir de src que já avançaram
            ao menos `min_progress` (0..1) ao longo do segmento src->dst (projeção no vetor).
          - Conecta cada alvo parcial a um super-sink com capacidade 1 e calcula max-flow com capacidades unitárias.
          - O valor do fluxo é o número de rotas paralelas (edge-disjoint) que progridem em direção ao destino,
            mesmo que não cheguem ao dst (rotas parciais).

        Observação:
          - Não filtra por zonas proibidas aqui; as penalidades de zona já tratam isso separadamente.
        """
        n = int(adj_np.shape[0])
        if n <= 1:
            return 0
        src = int(src)
        dst = int(dst)
        if src == dst:
            return 0

        # vetor src->dst
        p0 = np.array([float(pos_np[src, 0]), float(pos_np[src, 1])], dtype=np.float64)
        pL = np.array([float(pos_np[dst, 0]), float(pos_np[dst, 1])], dtype=np.float64)
        v = pL - p0
        den = float(v[0] * v[0] + v[1] * v[1])
        if den <= 1e-12:
            return 0

        cap0 = (adj_np > 0.0).astype(np.int32)

        # alcançáveis a partir de src
        reachable = [False] * n
        reachable[src] = True
        q = deque([src])
        while q:
            u = q.popleft()
            for vv in range(n):
                if cap0[u, vv] > 0 and not reachable[vv]:
                    reachable[vv] = True
                    q.append(vv)

        # coleta alvos parciais por projeção no segmento src->dst
        min_progress = float(max(0.0, min(1.0, min_progress)))
        targets = []
        for i in range(n):
            if i == src:
                continue
            if not reachable[i]:
                continue
            pi = np.array([float(pos_np[i, 0]), float(pos_np[i, 1])], dtype=np.float64)
            t = float(((pi - p0) @ v) / den)  # projeção normalizada
            if t >= min_progress:
                targets.append(i)

        if not targets:
            return 0

        # super-sink
        sink = n
        cap = np.zeros((n + 1, n + 1), dtype=np.int32)
        cap[:n, :n] = cap0
        for i in targets:
            cap[int(i), sink] = 1

        return int(self._max_edge_disjoint_paths_cap(cap, src, sink))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            # Seed o RNG interno do ambiente
            self._rng.seed(int(seed))

        if self.variable_drones and self.min_drones and self.max_drones:
            # Sorteia num_drones dentro de [min_drones, max_drones]
            self.num_drones = int(
                self._rng.randint(int(self.min_drones), int(self.max_drones) + 1)
            )
        else:
            # Garante que num_drones não ultrapasse max_drones (capacidade)
            if self.num_drones > self.max_drones:
                self.num_drones = self.max_drones

        self._reset_zonas()

        # -------------------------
        # Currículo: dirty reset
        # -------------------------
        # Ideia:
        #  - no começo, só parte dos episódios nasce "sujo" (com drones/arestas em zona)
        #  - ao final, 100% dos episódios ficam sujos (e sempre há pelo menos 1 drone na zona),
        #    forçando o agente a aprender a "limpar" removendo links e realocando drones.
        self._reset_count += 1
        if self.zonas_proibidas:
            prog = min(1.0, float(self._reset_count) / float(self.dirty_anneal_episodes))
            p_dirty = self.dirty_reset_prob_start + (self.dirty_reset_prob_end - self.dirty_reset_prob_start) * prog
            p_force = self.dirty_force_drone_in_zone_prob_start + (
                self.dirty_force_drone_in_zone_prob_end - self.dirty_force_drone_in_zone_prob_start
            ) * prog

            self._dirty_episode = (float(self._rng.rand()) < float(p_dirty))
            # força drone dentro da zona somente quando "sujo"
            self._force_drone_in_zone = bool(self._dirty_episode) and (float(self._rng.rand()) < float(p_force))
        else:
            self._dirty_episode = False
            self._force_drone_in_zone = False

        # Recalcula estado inicial de acordo com o currículo
        self._init_state()
        # reset tracking de zona
        self._prev_links_cross_zone = None
        self._prev_drones_in_zone = None
        self._zone_clean_once = False
        self._episode_steps = 0
        # Reset de sucesso (evita "carregar" streak/bonus entre episódios)
        self._success_streak = 0
        self._bonus_sucesso_dado = False
        obs = self._get_obs()
        if _GYMNASIUM:
            return obs, {"reset": True}
        return obs

    def step(self, action):
        """Executa 1 passo do ambiente.

        Ajuste (mask): aplica toggle_op (ADD/REMOVE) ANTES do movimento.
        Assim, o action-mask calculado no estado pré-ação continua coerente
        com a validação do link, evitando muitos `invalid_link_action` artificiais
        quando a ação inclui mover + togglar link no mesmo step.

        action: [move_op, toggle_op]  (ou formatos legados 4D/6D; ver decodificação abaixo)
        """

        # ---------- decodifica ação ----------
        # Novo formato (reestruturado): action = [move_op, toggle_op]
        # move_op codifica (move_idx, step_code, dir_code) em um único inteiro:
        #   0 = não mover
        #   1 + move_idx*(STEP_CELLS_N*4) + step_code*4 + (dir_code-1), com dir_code in {1..4}
        #
        # Compatibilidade:
        # - formato antigo 4D: [move_idx, step_code, dir_code, toggle_op]
        # - formato antigo 6D: [move_idx, step_code, dir_code, toggle_flag, src, dst]

        move_idx = 0
        step_code = 0
        dir_code = 0
        toggle_op = 0

        if len(action) >= 6:
            # [move_idx, step_code, dir_code, toggle_flag, src, dst]
            move_idx = int(action[0])
            step_code = int(action[1])
            dir_code = int(action[2])

            old_toggle_flag = int(action[3])
            old_src = int(action[4])
            old_dst = int(action[5])
            if old_toggle_flag == 1 and old_src != old_dst:
                edge_idx = self._edge_pair_to_idx.get((old_src, old_dst), None)
                if edge_idx is not None:
                    # no formato antigo era "toggle" (se existe remove, senão adiciona)
                    if (old_src < self.num_drones and old_dst < self.num_drones and
                            self.adjacency_matrix[old_src, old_dst].item() > 0.5):
                        toggle_op = 1 + self._E + edge_idx  # REMOVE
                    else:
                        toggle_op = 1 + edge_idx  # ADD

        elif len(action) == 4:
            # [move_idx, step_code, dir_code, toggle_op]
            move_idx = int(action[0])
            step_code = int(action[1])
            dir_code = int(action[2])
            toggle_op = int(action[3])

        else:
            # novo [move_op, toggle_op]
            move_op = int(action[0]) if len(action) > 0 else 0
            toggle_op = int(action[1]) if len(action) > 1 else 0

            if move_op <= 0:
                move_idx = 0
                step_code = 0
                dir_code = 0
            else:
                move_op = max(0, min(int(move_op), int(self._move_ops_n) - 1))
                k = int(move_op) - 1
                per_drone = int(STEP_CELLS_N) * 4
                move_idx = int(k // per_drone)
                r = int(k % per_drone)
                step_code = int(r // 4)
                dir_code = int(1 + (r % 4))

        step_code = max(0, min(int(STEP_CELLS_N) - 1, int(step_code)))
        step_cells = int(STEP_CELLS_CHOICES[step_code])

        dx, dy = 0, 0
        if dir_code == 1:
            dx = +1
        elif dir_code == 2:
            dx = -1
        elif dir_code == 3:
            dy = +1
        elif dir_code == 4:
            dy = -1
        else:
            # dir_code==0 ou inválido => não mover
            step_cells = 0

        # ---------- 1) aplica toggle ANTES do movimento ----------
        invalid_link_action = False
        link_changed = False

        if toggle_op != 0:
            op = int(toggle_op) - 1
            is_remove = (op >= self._E)
            edge_idx = (op - self._E) if is_remove else op

            if 0 <= edge_idx < self._E:
                src, dst = self._edge_pairs[edge_idx]

                # drones fora do range / inativos
                if src >= self.num_drones or dst >= self.num_drones:
                    invalid_link_action = True
                elif src == dst:
                    invalid_link_action = True
                else:
                    is_on = (self.adjacency_matrix[src, dst].item() > 0.5)

                    if is_remove:
                        # Remoção: só é válida se a aresta existe.
                        if is_on:
                            self.adjacency_matrix[src, dst] = 0.0
                            link_changed = True
                        else:
                            invalid_link_action = True
                    else:
                        # Adição: só é válida se a aresta NÃO existe e respeita regras.
                        if is_on:
                            invalid_link_action = True
                        else:
                            # 1) limite de grau de saída
                            out_deg_src = int(torch.sum(self.adjacency_matrix[src] > 0.5).item())
                            if out_deg_src >= self.max_out_connections:
                                invalid_link_action = True

                            # 2) evita criar link direto 0<->N-1 quando é fisicamente ruim (p abaixo do limiar).
                            if (not invalid_link_action):
                                p_min_direct = float(self.fit_w.get("DIRECT_0N1_MIN_P", 0.0))
                                if p_min_direct > 0.0 and self.num_drones > 2:
                                    if (src == 0 and dst == self.num_drones - 1) or (
                                            src == self.num_drones - 1 and dst == 0):
                                        dx_m = (float(self.drone_positions[src, 0].item()) - float(
                                            self.drone_positions[dst, 0].item())) * self.square_size_m
                                        dy_m = (float(self.drone_positions[src, 1].item()) - float(
                                            self.drone_positions[dst, 1].item())) * self.square_size_m
                                        d_m = float(math.hypot(dx_m, dy_m))
                                        p_dir = float(self._prob_link(d_m))
                                        if p_dir < p_min_direct:
                                            invalid_link_action = True

                            # 3) Hard-mask: se habilitado, bloqueia ADD de arestas que cruzam zona proibida.
                            if (not invalid_link_action) and self.hard_mask_zone_edges and self.zonas_proibidas:
                                p1 = (int(round(float(self.drone_positions[src, 0].item()))), int(round(float(self.drone_positions[src, 1].item()))))
                                p2 = (int(round(float(self.drone_positions[dst, 0].item()))), int(round(float(self.drone_positions[dst, 1].item()))))
                                if self._linha_passa_por_zona(p1, p2, (src, dst)):
                                    invalid_link_action = True

                            # 3b) Bloqueia criação de ciclos: se já existe caminho dst -> src,
                            # então adicionar src->dst fecha um ciclo dirigido.
                            if not invalid_link_action:
                                n = int(self.num_drones)
                                if self._has_path(dst, src, self.adjacency_matrix[:n, :n]):
                                    invalid_link_action = True

                            # 4) (mask-relax) NÃO bloqueia por left-link. Essa restrição é aprendida via penalidades.
                            if not invalid_link_action:
                                self.adjacency_matrix[src, dst] = 1.0
                                link_changed = True
            else:
                invalid_link_action = True

        if link_changed:
            self._mask_version += 1

                # ---------- 2) movimento (1 drone por step) ----------
        invalid_move_count = 0
        moved = False
        move_reverted = False

        # IMPORTANTE: drones 0 e (num_drones-1) são fixos (âncoras).
        # Apenas UM drone intermediário (move_idx) pode se mover por step.
        if dir_code != 0 and step_cells > 0:
            if move_idx <= 0 or move_idx >= self.num_drones - 1:
                invalid_move_count += 1
            else:
                i = int(move_idx)
                x0 = int(self.drone_positions[i, 0].item())
                y0 = int(self.drone_positions[i, 1].item())

                # Snapshot de arestas incidentes ANTES do movimento (para hard-block pós-movimento).
                outs = np.array([], dtype=np.int64)
                ins = np.array([], dtype=np.int64)
                prev_left_out = prev_left_in = None
                prev_cross_out = prev_cross_in = None

                n = int(self.num_drones)
                if n > 1:
                    adj_np = (_torch_to_numpy_cpu_view(self.adjacency_matrix[:n, :n]) > 0.5)
                    outs = np.nonzero(adj_np[i, :])[0].astype(np.int64, copy=False)
                    ins = np.nonzero(adj_np[:, i])[0].astype(np.int64, copy=False)

                    pos_np = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2]).astype(np.float32, copy=False)
                    xi = float(pos_np[i, 0])

                    if outs.size:
                        prev_left_out = (pos_np[outs, 0] < xi)
                    if ins.size:
                        prev_left_in = (xi < pos_np[ins, 0])

                    if getattr(self, "zonas_proibidas", None):
                        # edge_cross é mantido em cache; garante que exista para o n atual
                        self._ensure_edge_cross_mat()
                        if self._edge_cross_mat is not None:
                            cross_np = self._edge_cross_mat[:n, :n]
                            if outs.size:
                                prev_cross_out = (cross_np[i, outs] > 0)
                            if ins.size:
                                prev_cross_in = (cross_np[ins, i] > 0)

                # HARD inválido: se o drone está FORA da zona e o movimento (em qualquer micro-passo)
                # tentaria ENTRAR na zona proibida, então a ação é inválida e NÃO move.
                # Observação: se o drone já estiver dentro (dirty episode), permitimos mover para SAIR.
                x, y = x0, y0
                invalid_zone_move = False
                for _ in range(step_cells):
                    nx = max(0, min(self.grid_size - 1, x + dx))
                    ny = max(0, min(self.grid_size - 1, y + dy))

                    cur_in_zone = (x, y) in self.zonas_proibidas
                    nxt_in_zone = (nx, ny) in self.zonas_proibidas

                    if (not cur_in_zone) and nxt_in_zone:
                        invalid_zone_move = True
                        break

                    x, y = nx, ny

                if invalid_zone_move:
                    invalid_move_count += 1
                    x, y = x0, y0

                # aplica nova posição
                if x != x0 or y != y0:
                    self.drone_positions[i, 0] = float(x)
                    self.drone_positions[i, 1] = float(y)
                    moved = True

                if moved:
                    self._dist_matrix = None
                    self._dist_pos_sig = None

                    # Atualiza edge_cross (depende de posição)
                    if self.obs_edge_cross or self.hard_mask_zone_edges:
                        self._update_edge_cross_after_move(i)

                    # HARD pós-movimento: NÃO permitir que o movimento crie NOVOS left-links incidentes
                    # e NÃO permitir que crie NOVOS cruzamentos de zona incidentes.
                    hard_bad = False
                    if n > 1:
                        pos_np2 = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2]).astype(np.float32, copy=False)
                        xi2 = float(pos_np2[i, 0])

                        if outs.size and prev_left_out is not None:
                            new_left_out = (pos_np2[outs, 0] < xi2)
                            if np.any(new_left_out & (~prev_left_out)):
                                hard_bad = True

                        if (not hard_bad) and ins.size and prev_left_in is not None:
                            new_left_in = (xi2 < pos_np2[ins, 0])
                            if np.any(new_left_in & (~prev_left_in)):
                                hard_bad = True

                        if (not hard_bad) and getattr(self, "zonas_proibidas", None) and self._edge_cross_mat is not None:
                            cross_np2 = self._edge_cross_mat[:n, :n]

                            if outs.size and prev_cross_out is not None:
                                new_cross_out = (cross_np2[i, outs] > 0)
                                if np.any(new_cross_out & (~prev_cross_out)):
                                    hard_bad = True

                            if (not hard_bad) and ins.size and prev_cross_in is not None:
                                new_cross_in = (cross_np2[ins, i] > 0)
                                if np.any(new_cross_in & (~prev_cross_in)):
                                    hard_bad = True

                    if hard_bad:
                        invalid_move_count += 1
                        # Reverte posição
                        self.drone_positions[i, 0] = float(x0)
                        self.drone_positions[i, 1] = float(y0)

                        # Atualiza edge_cross de volta para a posição anterior
                        if self.obs_edge_cross or self.hard_mask_zone_edges:
                            self._update_edge_cross_after_move(i)

                        move_reverted = True
                        moved = False

                    # Invalida máscaras (ADD/REMOVE e também a própria máscara de movimento)
                    # porque dependem de posição (left-link / cruzamento de zona).
                    if moved:
                        self._mask_version += 1
                        self._mask_cache = None
                        self._mask_cache_version = -1

        # ---------- fit/reward ----------

        reward, info = self._calculate_fit()

        # Penaliza ações inválidas para ajudar a política a aprender mais rápido
        if invalid_move_count > 0 or invalid_link_action:
            reward -= float(self.fit_w.get("INVALID_ACTION_PENALTY", 0.0)) * (float(self.num_drones) / float(max(1, self.max_drones)))

        # ----- critério de sucesso -----
        success_now = bool(
            info.get("path_exists", False)
            and self.current_reliability >= RELIABILITY_TARGET
            and info.get("drones_in_zone", 0) == 0
            and info.get("links_cross_zone", 0) == 0
            and info.get("left_links_count", 0) == 0
            and info.get("is_dag", 1) == 1
        )

        if success_now:
            self._success_streak += 1
        else:
            self._success_streak = 0

        success_reached = self._success_streak >= self._success_need
        if success_reached and not self._bonus_sucesso_dado:
            reward += 500.0
            self._bonus_sucesso_dado = True

        # ----- término de episódio -----
        self._last_reward = float(reward)
        self._episode_steps += 1

        done = False
        if self._episode_steps >= self._max_episode_steps or success_reached:
            done = True

        truncation = (self._episode_steps >= self._max_episode_steps)
        obs = self._get_obs()
        info["reliability"] = float(self.current_reliability)
        info["invalid_action"] = bool(invalid_move_count > 0 or invalid_link_action)
        info["invalid_action_penalty_scale"] = float(self.num_drones) / float(max(1, self.max_drones))
        info["invalid_move_count"] = int(invalid_move_count)
        info["success"] = bool(success_reached)
        if success_reached:
            info["success_at_step"] = int(self._episode_steps)

        # Terminal shaping: alinha o treino com a métrica do TESTE (confiabilidade do estado FINAL do episódio).
        if done:
            info["is_success"] = bool(success_reached)
            # Recalcula confiabilidade no estado FINAL para evitar qualquer divergência (cache/stale)
            try:
                n = int(self.num_drones)
                adj_t = _torch_to_numpy_cpu_view(self.adjacency_matrix)
                pos_t = _torch_to_numpy_cpu_view(self.drone_positions)
                adj_np = (adj_t[:n, :n] > 0).astype(np.float32)
                pos_np = np.asarray(pos_t[:n, :2], dtype=np.float32)

                # Snapshot terminal: evita divergências no pós-processamento quando VecEnv faz auto-reset.
                # (O teste TesteMultiDrone.txt dá preferência a estes campos.)
                info["terminal_adj01"] = adj_np
                info["terminal_pos_cells"] = pos_np
                info["terminal_num_drones"] = int(n)
                info["terminal_square_size_m"] = float(self.square_size_m)
                path_final = bool(self._has_path(0, n - 1, adj_np)) if n > 1 else False
                if path_final:
                    self.current_reliability = float(
                        calcula_confiabilidade_iterativa(adj_np, pos_np, self.square_size_m, params=self._radio_params)
                    )
                else:
                    self.current_reliability = 0.0
            except Exception:
                pass

            info["reliability_terminal"] = float(self.current_reliability)
            info["done_reason"] = ("success" if success_reached else ("timeout" if truncation else "done"))

            viol = int(info.get("drones_in_zone", 0) + info.get("links_cross_zone", 0))
            info["zone_violations_terminal"] = int(viol)

            # métricas terminais (estado FINAL, sem "sujeira" inicial)
            info["left_links_terminal"] = int(info.get("left_links_count", 0))
            info["right_links_ratio_terminal"] = float(info.get("right_links_ratio", 0.0))
            info["success_terminal"] = int(1 if success_reached else 0)


            # Terminal bonus/penalty baseado em confiabilidade final
            term_rel_w = float(self.fit_w.get("TERM_REL", 0.0))
            if term_rel_w != 0.0:
                # Usa a MESMA métrica de _calculate_fit(): -log10(1-rel) mapeada entre REL_BASE e TARGET
                eps = 1e-12
                rel_clipped = max(eps, min(1.0 - eps, float(self.current_reliability)))
                fail = max(eps, 1.0 - rel_clipped)
                fail_b = max(eps, 1.0 - float(RELIABILITY_BASE))
                fail_t = max(eps, 1.0 - float(RELIABILITY_TARGET))
                a = -math.log10(fail_b)
                b = -math.log10(fail_t)
                cur = -math.log10(fail)
                if b <= a + 1e-12:
                    rel_term = 1.0 if rel_clipped >= float(RELIABILITY_TARGET) else 0.0
                else:
                    rel_term = (cur - a) / (b - a)
                rel_term = max(0.0, min(1.0, float(rel_term)))
                term_rel_score = term_rel_w * (2.0 * rel_term - 1.0)
                reward += term_rel_score
                info["term_rel_term"] = float(rel_term)
                info["term_rel_score"] = float(term_rel_score)

            # bônus de sucesso terminal (apenas se zona limpa)
            if (self.current_reliability >= RELIABILITY_TARGET and viol == 0 and
                    int(info.get("left_links_count", 0)) == 0 and int(info.get("is_dag", 1)) == 1):
                reward += float(self.fit_w.get("TERM_SUCCESS_BONUS", 0.0))

            # penalidade terminal por violação de zona
            if viol > 0:
                reward -= float(self.fit_w.get("TERM_ZONE_PEN", 0.0))

            self._last_reward = float(reward)

        # Return compatível Gymnasium vs Gym
        if _GYMNASIUM:
            return obs, float(reward), bool(done), bool(truncation), info
        return obs, float(reward), bool(done or truncation), info
    def _get_obs(self):
        """Observação com padding até max_drones e máscara de drones ativos.
        - As primeiras self.num_drones entradas representam drones ativos.
        - Os demais índices (até max_drones) ficam zerados e mask=0.
        """
        num = int(self.num_drones)
        pos = _torch_to_numpy_cpu_view(self.drone_positions[:num, :2])  # (num_drones, 2)
        pos_norm = (pos / max(1.0, (self.grid_size - 1))) * 2.0 - 1.0

        adj = _torch_to_numpy_cpu_view(self.adjacency_matrix[:num, :num])  # (num_drones, num_drones)

        max_n = int(self.max_drones)

        pos_full = np.zeros((max_n, 2), dtype=np.float32)
        adj_full = np.zeros((max_n, max_n), dtype=np.float32)
        mask = np.zeros((max_n,), dtype=np.float32)

        pos_full[:num, :] = pos_norm
        adj_full[:num, :num] = adj
        mask[:num] = 1.0

        obs_parts = [pos_full.reshape(-1), adj_full.reshape(-1), mask]

                # (opcional) mapa downsample da zona proibida
        if self.obs_zone_map:
            # Garante cache atualizado e, principalmente, tamanho FIXO
            # mesmo quando não há zona (p_sem_zona>0).
            expected = int(self._zone_map_ds * self._zone_map_ds) if int(getattr(self, '_zone_map_ds', 0) or 0) > 0 else 0
            sig = (int(getattr(self, '_zone_sig', 0)) ^ (int(self.zone_map_scale) << 16) ^ (int(self._zone_map_ds) << 1)) & 0xFFFFFFFF
            if (self._zone_map_down is None) or (self._zone_map_down_sig != sig) or (expected and (self._zone_map_down.shape[0] != expected)):
                self._rebuild_zone_map_cache()
            if expected > 0 and (self._zone_map_down is None or self._zone_map_down.shape[0] != expected):
                # fallback ultra-seguro: tamanho fixo
                self._zone_map_down = np.zeros((expected,), dtype=np.float32)
                self._zone_map_down_sig = sig
            obs_parts.append(self._zone_map_down if self._zone_map_down is not None else np.zeros((expected,), dtype=np.float32))

        # (opcional) matriz NxN indicando cruzamento de zona por aresta (i,j)
        if self.obs_edge_cross:
            cross_full = np.zeros((max_n, max_n), dtype=np.float32)
            cross_mat = self._ensure_edge_cross_mat()
            if cross_mat is not None and num > 0:
                # cross_mat é (num,num)
                cross_full[:num, :num] = cross_mat.astype(np.float32, copy=False)
            obs_parts.append(cross_full.reshape(-1))

        return np.concatenate(obs_parts, axis=0).astype(np.float32, copy=False)

    # =========================
    # Fitness / Recompensa
    # =========================

    def _emax_scales(self, n=None):
        """Compute normalization scales for variable number of active drones.

        Emax(n) = n * max_out_connections (upper bound on directed edges).
        Scales are defined so that when n == self.max_drones the scale is ~1.0.
        """
        if n is None:
            n = int(self.num_drones)
        else:
            n = int(n)
        n_ref = int(self.max_drones)
        max_out = int(self.max_out_connections)

        emax_n = max(1, n * max_out)
        emax_ref = max(1, n_ref * max_out)

        link_scale = float(emax_ref) / float(emax_n)              # scale for link-count terms
        drone_scale = float(n_ref) / float(max(1, n))             # scale for drone-count terms
        drone_scale_wo0 = float(max(1, n_ref - 1)) / float(max(1, n - 1))  # excludes drone 0
        relay_scale = float(max(1, n_ref - 2)) / float(max(1, n - 2))      # excludes anchors 0 and n-1

        return {
            "Emax_n": int(emax_n),
            "Emax_ref": int(emax_ref),
            "link_scale": float(link_scale),
            "drone_scale": float(drone_scale),
            "drone_scale_wo0": float(drone_scale_wo0),
            "relay_scale": float(relay_scale),
        }


    def _calculate_fit(self):
        """
        Calcula a aptidão (fitness) do estado atual.

        Prioridade de objetivos (em ordem):
        1. Existir caminho 0 -> N-1.
        2. Alta confiabilidade 0->N-1 (RELIABILITY_TARGET).
        3. Nenhum drone nem link em zonas proibidas.
        4. Throughput abaixo do teto máximo (sem encostar no limite).
        5. Número de hops razoável (apenas depois dos objetivos acima estarem bons).
        6. Layout "bonito" (drones entre o primeiro e o último) com peso MUITO pequeno.
        """
        w = self.fit_w  # pesos efetivos do fit (centralizados)

        # Conversões básicas (sempre considerando SOMENTE drones ativos: 0..num_drones-1)
        n = int(self.num_drones)
        scales = self._emax_scales(n)
        Emax_n = int(scales["Emax_n"])
        Emax_ref = int(scales["Emax_ref"])
        link_scale = float(scales["link_scale"])
        drone_scale = float(scales["drone_scale"])
        drone_scale_wo0 = float(scales["drone_scale_wo0"])
        relay_scale = float(scales["relay_scale"])
        pos_np = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2])
        adj_np = (_torch_to_numpy_cpu_view(self.adjacency_matrix[:n, :n]) > 0.5).astype(np.float32)
        # Pré-computa arestas e listas de adjacência (hot-path): evita varreduras O(n²) repetidas
        ei, ej = np.nonzero(adj_np > 0.0)
        out_neighbors = [[] for _ in range(n)]
        in_neighbors = [[] for _ in range(n)]
        if ei.size > 0:
            for i, j in zip(ei.tolist(), ej.tolist()):
                out_neighbors[int(i)].append(int(j))
                in_neighbors[int(j)].append(int(i))
        out_deg = np.bincount(ei.astype(np.int32, copy=False), minlength=n) if ei.size > 0 else np.zeros(n, dtype=np.int32)
        in_deg = np.bincount(ej.astype(np.int32, copy=False), minlength=n) if ej.size > 0 else np.zeros(n, dtype=np.int32)
        pos_round = np.rint(pos_np[:, :2]).astype(np.int32, copy=False)
        dist_full = self._calc_distance_matrix()

        # recorta (dist_full já é n x n)
        dist_m = dist_full[:n, :n]
        # path_exists precisa ser definido antes de ser usado (escopo local em Python)
        path_exists = bool(self._has_path(0, n - 1, adj_np, out_neighbors=out_neighbors)) if n > 1 else False
        # -------------------------
        # Sem loops: detecta ciclos no grafo (DAG esperado)
        # -------------------------
        is_dag = True
        if n > 1 and ei.size > 0:
            indeg_tmp = in_deg.astype(np.int32, copy=True)
            stack = [i for i in range(n) if indeg_tmp[i] == 0]
            seen = 0
            while stack:
                u = stack.pop()
                seen += 1
                for v in out_neighbors[u]:
                    indeg_tmp[v] -= 1
                    if indeg_tmp[v] == 0:
                        stack.append(v)
            is_dag = (seen == n)

        cycle_penalty = 0.0
        if not is_dag:
            cycle_penalty = -float(w.get("CYCLE_PENALTY", 0.0)) * float(link_scale)

        # -------------------------
        # Bottleneck shaping: melhora o pior enlace no caminho 0->(n-1)
        # - encontra um caminho (BFS) e calcula p_min ao longo do caminho.
        # - recompensa se p_min ultrapassa BOTTLENECK_TARGET_P; penaliza se abaixo.
        # -------------------------
        bottleneck_min_p = 0.0
        bottleneck_max_dist_m = 0.0
        bottleneck_term = 0.0
        if path_exists:
            path_nodes_b = self._get_any_path_nodes(0, n - 1, adj_np, out_neighbors=out_neighbors)
            if len(path_nodes_b) >= 2:
                p_min = 1.0
                d_max = 0.0
                for a, b in zip(path_nodes_b[:-1], path_nodes_b[1:]):
                    d = float(dist_m[a, b])
                    d_max = max(d_max, d)
                    p = float(self._prob_link(d))
                    p_min = min(p_min, p)
                bottleneck_min_p = float(p_min)
                bottleneck_max_dist_m = float(d_max)
                bt_w = float(w.get("BOTTLENECK_W", 0.0))
                bt_t = float(w.get("BOTTLENECK_TARGET_P", 0.7))
                if bt_w != 0.0:
                    bottleneck_term = bt_w * (float(bottleneck_min_p) - bt_t)
                    bottleneck_term = max(-bt_w, min(bt_w, bottleneck_term))

        # -------------------------
        # Qualidade de conexão (links mais curtos são melhores) - peso pequeno
        # connection_quality ∈ [0, 1]
        # -------------------------
        num_links = int(ei.size)
        # Emax-based normalization can overweight link penalties when graphs are sparse (few links).
        # To keep training stable with variable n, soften link scaling by link density (num_links / Emax_n).
        link_density = float(num_links) / float(max(1, Emax_n)) if num_links > 0 else 0.0
        link_scale_eff = float(link_scale) * float(np.sqrt(max(1e-6, link_density)))
        if num_links > 0:
            mean_dist = float(dist_m[ei, ej].mean())
            norm_mean_dist = mean_dist / max(1.0, self.grid_size * self.square_size_m)
            connection_quality = 1.0 / (1.0 + norm_mean_dist)
        else:
            connection_quality = 0.0

        conn_quality_term = w["CONN_QUALITY_COEF"] * float(connection_quality)

        # -------------------------
        # Alcançabilidade até o último drone (peso=1 no fit)
        # reach_last_penalty = -(1 - reach_frac) ∈ [-1, 0]
        # -------------------------
        reach_count = 0
        reach_frac = 0.0
        if n > 1:
            # Uma única BFS no grafo reverso a partir do último drone
            can_reach_last = np.zeros(n, dtype=bool)
            dst = n - 1
            can_reach_last[dst] = True
            qrev = deque([dst])
            while qrev:
                u = qrev.popleft()
                for p in in_neighbors[u]:
                    if not can_reach_last[p]:
                        can_reach_last[p] = True
                        qrev.append(p)
            # conta apenas src em [0..n-2] (mesma semântica do código antigo)
            reach_count = int(np.sum(can_reach_last[: max(0, n - 1)]))
            reach_frac = reach_count / max(1, n - 1)
        reach_last_penalty = -(1.0 - float(reach_frac))

        # -------------------------
        # Alcançabilidade A PARTIR do primeiro drone (0) – penaliza componentes desconectados do 0
        # reach_from_first_frac = fração de nós (1..N-1) alcançáveis via arestas dirigidas a partir do 0
        # unreachable_from_first_penalty = -UNREACHABLE_FROM_FIRST_PENALTY * (#nós não alcançáveis)
        # -------------------------
        reachable = np.zeros(n, dtype=bool)
        reachable[0] = True
        q0 = deque([0])
        while q0:
            u = q0.popleft()
            # segue arestas de saída u -> v (via lista de adjacência)
            for v in out_neighbors[u]:
                if not reachable[v]:
                    reachable[v] = True
                    q0.append(v)

        reachable_count = int(np.sum(reachable[1:]))  # exclui o próprio 0
        reach_from_first_frac = reachable_count / max(1, n - 1)
        unreachable_from_first_count = int((n - 1) - reachable_count)
        unreachable_from_first_penalty = -float(w.get("UNREACHABLE_FROM_FIRST_PENALTY", 0.0)) * float(unreachable_from_first_count) * float(drone_scale_wo0)
        reach_from_first_penalty = -float(w.get("REACH_FROM_FIRST_W", 0.0)) * float(1.0 - reach_from_first_frac)

        # -------------------------
        # Estrutura de relays (apenas penalizações; peso=1 no fit)
        # - ISOLATED_NODE_PENALTY entra dentro de invalid_relay_penalty
        # - demais termos usam penalizações internas (0.2 * diff etc.)
        # -------------------------
        invalid_relay_penalty = 0.0
        blocked_relay_penalty = 0.0
        imbalance_penalty = 0.0
        dead_end_penalty = 0.0

        for node in range(1, n - 1):
            in_deg_i = int(in_deg[node])
            out_deg_i = int(out_deg[node])

            # nó completamente isolado (não atua como relay)
            if in_deg_i == 0 and out_deg_i == 0:
                invalid_relay_penalty -= float(w["ISOLATED_NODE_PENALTY"])
                continue

            # Nós com apenas entrada ou apenas saída (estruturas "terminais" indesejadas)
            if in_deg_i == 0 and out_deg_i > 0:
                blocked_relay_penalty -= 0.2

            if in_deg_i > 0 and out_deg_i == 0 and node != n - 1:
                blocked_relay_penalty -= 0.2
                dead_end_penalty -= 0.2

            # Desequilíbrio entre entrada e saída
            if (in_deg_i + out_deg_i) > 0:
                diff = abs(in_deg_i - out_deg_i)
                imbalance_penalty -= 0.2 * diff

        # -------------------------
        # Normaliza penalizações estruturais por nº de relays (n variável)
        invalid_relay_penalty *= float(relay_scale)
        blocked_relay_penalty *= float(relay_scale)
        imbalance_penalty *= float(relay_scale)
        dead_end_penalty *= float(relay_scale)

        # Links "para trás" (x_dst < x_src) – calculado mas NÃO entra no fit atual
        # -------------------------
        if num_links > 0:
            left_links_count = int(np.sum(pos_np[ej, 0] < pos_np[ei, 0]))
            right_links_count = int(num_links - left_links_count)
            right_links_ratio = float(right_links_count) / float(max(1, num_links))
        else:
            left_links_count = 0
            right_links_count = 0
            right_links_ratio = 0.0
        left_links_penalty = 0.0  # gated: applied only when path 0->(n-1) exists

        # -------------------------
        # Hops e latência: caminho mais longo 0->N-1
        # latency_penalty ≤ 0, latency_reward ≥ 0
        # -------------------------
        path_exists = False
        max_hops = 0
        avg_hops = 0.0
        hops_score = 0.0
        latency_penalty = 0.0
        latency_reward = 0.0

        longest_path_dist = 0.0

        if num_links > 0 and n > 1:
            visited = np.zeros(n, dtype=bool)
            dst = n - 1
            state = {"path_exists": False, "max_hops": 0, "longest_path_dist": 0.0}

            def dfs(u, hops, dist_acc):
                if u == dst:
                    state["path_exists"] = True
                    if hops > state["max_hops"]:
                        state["max_hops"] = hops
                        state["longest_path_dist"] = dist_acc
                    return
                visited[u] = True
                for v in out_neighbors[u]:
                    if not visited[v]:
                        dfs(v, hops + 1, dist_acc + float(dist_m[u, v]))
                visited[u] = False

            dfs(0, 0, 0.0)

            path_exists = bool(state["path_exists"])
            max_hops = int(state["max_hops"])
            longest_path_dist = float(state["longest_path_dist"])

            if path_exists and max_hops > 0:
                avg_hops = float(max_hops)
                hops = max_hops
                latency_ms = (
                        hops * LATENCY_PER_HOP_MS
                        + longest_path_dist / max(1e-6, SPEED_M_PER_MS)
                )
                if latency_ms > MAX_LATENCY_MS:
                    latency_penalty -= 0.005 * (latency_ms - MAX_LATENCY_MS) / MAX_LATENCY_MS
                else:
                    latency_reward += 0.005 * (MAX_LATENCY_MS - latency_ms) / MAX_LATENCY_MS
            else:
                max_hops -= 15.0


        # Gate left-link penalty: allow exploration until a 0->(n-1) path exists
        if path_exists and left_links_count > 0:
            left_links_penalty = -float(w["LEFT_LINK_COEF"]) * float(left_links_count) * float(link_scale_eff)
            left_links_gate = 1
        else:
            left_links_penalty = 0.0
            left_links_gate = 0

        self.current_latency_penalty = float(latency_penalty)

        # -------------------------
        # Throughput (simplificado) – calculado mas NÃO entra no fit atual
        # Normaliza por Emax(n): mantém escala aproximada quando n varia (9..13)
        # -------------------------
        throughput = 0.0
        num_links_eff = int(max(1, int(round(float(num_links) * float(link_scale_eff))))) if num_links > 0 else 0
        if path_exists:
            throughput = THROUGHPUT_ORIGEM_MBPS
            throughput = min(throughput * max(1, num_links_eff), THROUGHPUT_MAX_MBPS)
        self.current_throughput = float(throughput)

        throughput_penalty = 0.0
        if throughput > 0.0:
            ratio_tp = throughput / THROUGHPUT_MAX_MBPS
            if ratio_tp >= 0.85:
                # penalidade progressiva a partir de 85% do teto, chegando ao máximo em 100%
                excess = (ratio_tp - 0.85) / 0.15
                excess = max(0.0, min(1.0, float(excess)))
                throughput_penalty -= float(w["THROUGHPUT_PENALTY_MAX"]) * (excess ** 2)

        # -------------------------
        # Confiabilidade 0->N-1 (cálculo físico)
        # -------------------------
        if path_exists:
            try:
                reliability = float(
                    calcula_confiabilidade_iterativa(adj_np, pos_np, self.square_size_m, params=self._radio_params)
                )
            except Exception:
                reliability = 0.0
        else:
            reliability = 0.0

        self.current_reliability = float(reliability)

        # -------------------------
        # Termos de espalhamento (layout)
        # nn_reward, uniform_reward ~ [0,1] quando reliability>0
        # (no seu código atual, se reliability==0: nn_reward recebe -100)
        # -------------------------
        nn_reward = 0.0
        uniform_reward = 0.0
        nn_mean_dist = 0.0

        if reliability > 0.0:
            if n > 1:
                # NN-spacing
                nn_dists = []
                for i in range(n):
                    dists_i = dist_m[i, :n].copy()
                    dists_i[i] = np.inf
                    nn = float(np.min(dists_i))
                    if np.isfinite(nn):
                        nn_dists.append(nn)
                if nn_dists:
                    nn_mean_dist = float(np.mean(nn_dists))
                    max_possible = math.sqrt(2.0) * self.grid_size * self.square_size_m
                    nn_norm = min(1.0, nn_mean_dist / max(1.0, max_possible))
                    nn_reward = nn_norm

                # Uniformidade 1D ao longo do segmento (drone0 -> droneN-1)
                # Ajuda a espalhar os drones do início ao fim (evita “bola”), sem incentivar “preencher bordas” 2D.
                p0 = pos_np[0, :2]
                p1 = pos_np[n - 1, :2]
                v = p1 - p0
                vv = float(v[0] * v[0] + v[1] * v[1])
                bins = int(w.get("UNIFORM1D_BINS", 8))

                if vv > 1e-9 and n > 2 and bins > 0:
                    counts = np.zeros((bins,), dtype=np.float32)

                    # usa apenas drones intermediários (não inclui âncoras 0 e N-1)
                    for i in range(1, n - 1):
                        p = pos_np[i, :2]
                        t = float(((p - p0) @ v) / vv)
                        t = max(0.0, min(1.0, t))
                        b = min(bins - 1, int(t * bins))
                        counts[b] += 1.0

                    # opcional: ignora bins cujo ponto central esteja em zona proibida
                    valid = np.ones((bins,), dtype=bool)
                    if self.zonas_proibidas:
                        for b in range(bins):
                            tmid = (b + 0.5) / bins
                            pmid = p0 + tmid * v
                            cx = int(round(pmid[0]))
                            cy = int(round(pmid[1]))
                            if (cx, cy) in self.zonas_proibidas:
                                valid[b] = False

                    valid_counts = counts[valid]
                    if valid_counts.size > 0:
                        mean_count = float(valid_counts.mean())
                        var = float(((valid_counts - mean_count) ** 2).mean())
                        denom = max(1.0, float((n - 2) ** 2))  # ordem de grandeza do nº de intermediários
                        var_norm = var / denom
                        uniform_score = 1.0 / (1.0 + var_norm)
                        uniform_reward = float(uniform_score)
        else:
            nn_reward -= 100.0

        # -------------------------
        # Penalização por drones/links em zona proibida (proporcional)
        # -------------------------
        drones_in_zone = 0
        links_cross_zone = 0
        zone_fix_bonus = 0.0
        zone_clean_once_bonus = 0.0
        # cache local por endpoints (x1,y1,x2,y2) para acelerar checagens de cruzamento de zona
        zone_edge_cache = {}
        if self.zonas_proibidas:
            # conta drones na zona via grade booleana quando disponível
            if self._forbidden_grid is not None:
                drones_in_zone = int(self._forbidden_grid[pos_round[:n, 0], pos_round[:n, 1]].sum())
            else:
                for i in range(n):
                    if (int(pos_round[i, 0]), int(pos_round[i, 1])) in self.zonas_proibidas:
                        drones_in_zone += 1

            # conta links que cruzam zona iterando apenas arestas existentes
            if num_links > 0:
                for i, j in zip(ei.tolist(), ej.tolist()):
                    x1 = int(pos_round[i, 0]); y1 = int(pos_round[i, 1])
                    x2 = int(pos_round[j, 0]); y2 = int(pos_round[j, 1])
                    k = (x1, y1, x2, y2)
                    v = zone_edge_cache.get(k)
                    if v is None:
                        v = bool(self._linha_passa_por_zona((x1, y1), (x2, y2), (int(i), int(j))))
                        zone_edge_cache[k] = v
                    if v:
                        links_cross_zone += 1

        zone_penalty = 0.0
        # zone_penalty = -ZONE_DRONE_COEF*drones_in_zone - ZONE_LINK_COEF*links_cross_zone
        if drones_in_zone > 0:
            zone_penalty -= float(w["ZONE_DRONE_COEF"]) * float(drones_in_zone) * float(drone_scale)
        if links_cross_zone > 0:
            zone_penalty -= float(w["ZONE_LINK_COEF"]) * float(links_cross_zone) * float(link_scale_eff)

        # -------------------------
        # Shaping de ZONA: recompensa por REMOVER violações (delta) + bônus 1x quando limpa
        # -------------------------
        zone_fix_bonus = 0.0
        prev_links = self._prev_links_cross_zone
        prev_drones = self._prev_drones_in_zone
        if prev_links is None:
            prev_links = links_cross_zone
        if prev_drones is None:
            prev_drones = drones_in_zone

        d_links = int(prev_links) - int(links_cross_zone)
        d_drones = int(prev_drones) - int(drones_in_zone)
        if d_links > 0:
            zone_fix_bonus += float(w.get("ZONE_FIX_LINK_BONUS", 0.0)) * float(d_links) * float(link_scale_eff)
        if d_drones > 0:
            zone_fix_bonus += float(w.get("ZONE_FIX_DRONE_BONUS", 0.0)) * float(d_drones) * float(drone_scale)

        zone_clean_once_bonus = 0.0
        if (drones_in_zone == 0 and links_cross_zone == 0):
            if not getattr(self, "_zone_clean_once", False):
                zone_clean_once_bonus = float(w.get("ZONE_CLEAN_ONCE_BONUS", 0.0))
                self._zone_clean_once = True
        else:
            self._zone_clean_once = False

        self._prev_links_cross_zone = int(links_cross_zone)
        self._prev_drones_in_zone = int(drones_in_zone)
        # -------------------------
        # Alinhamento à reta (drone0 -> droneN-1): puxa drones intermediários para perto da linha
        # line_align_score ∈ [0,1] (1 = bem alinhado)
        # line_align_term = LINE_ALIGN_W * gate_zone * line_align_score
        # -------------------------
        line_align_score = 0.0
        line_align_term = 0.0
        LINE_ALIGN_W = float(w.get("LINE_ALIGN_W", 0.0))
        if LINE_ALIGN_W > 0.0 and n > 2:
            p0_line = pos_np[0, :2]
            p1_line = pos_np[n - 1, :2]
            v_line = p1_line - p0_line
            vv_line = float(v_line[0] * v_line[0] + v_line[1] * v_line[1])
            if vv_line > 1e-9:
                dperp = []
                for i in range(1, n - 1):
                    p = pos_np[i, :2]
                    t = float(((p - p0_line) @ v_line) / vv_line)
                    t = max(0.0, min(1.0, t))
                    proj = p0_line + t * v_line
                    dp = float(np.linalg.norm(p - proj))  # em "células"
                    dperp.append(dp)

                if dperp:
                    mean_dp = float(np.mean(dperp))
                    # normaliza por ~meia largura do grid (ordem de grandeza)
                    mean_dp_norm = min(1.0, mean_dp / max(1.0, self.grid_size * 0.5))
                    line_align_score = 1.0 - mean_dp_norm
                    # não compete com a penalidade enorme de zona: só bonifica quando não há violações
                    gate_zone = 1.0 if (drones_in_zone == 0 and links_cross_zone == 0) else 0.0
                    line_align_term = LINE_ALIGN_W * gate_zone * float(line_align_score)

        # -------------------------
        # Recompensa principal de confiabilidade (log-scale em (1-rel))
        # - Usa -log10(1-rel) para dar muita sensibilidade perto de 1.0
        # - Faz o agente 'sentir' melhora a partir de RELIABILITY_BASE
        # - rel_term=1.0 em RELIABILITY_TARGET
        # -------------------------
        eps = 1e-12
        rel_clipped = max(eps, min(1.0 - eps, float(reliability)))
        fail = max(eps, 1.0 - rel_clipped)
        fail_b = max(eps, 1.0 - float(RELIABILITY_BASE))
        fail_t = max(eps, 1.0 - float(RELIABILITY_TARGET))

        a = -math.log10(fail_b)
        b = -math.log10(fail_t)
        cur = -math.log10(fail)
        if b <= a + 1e-12:
            rel_term = 1.0 if rel_clipped >= float(RELIABILITY_TARGET) else 0.0
        else:
            rel_term = (cur - a) / (b - a)
        rel_term = max(0.0, min(1.0, float(rel_term)))

        # rel_score (no step) usa REL_STEP (padrão 0 => confiabilidade grande só no terminal)
        rel_step_w = float(w.get("REL_STEP", 0.0))
        rel_score = rel_step_w * (2.0 * rel_term - 1.0)

        # Para gates/relatórios
        rel_frac = float(rel_term)

        # Gate suave (quase duro) para termos secundários
        if reliability >= float(RELIABILITY_UNLOCK_END):
            rel_unlock_gate = 1.0
        elif reliability <= float(RELIABILITY_UNLOCK_START):
            rel_unlock_gate = 0.0
        else:
            rel_unlock_gate = (float(reliability) - float(RELIABILITY_UNLOCK_START)) / max(1e-12, (
                        float(RELIABILITY_UNLOCK_END) - float(RELIABILITY_UNLOCK_START)))
            rel_unlock_gate = max(0.0, min(1.0, rel_unlock_gate))

        # Penalização por falta de caminho 0->N-1
        path_penalty = 0.0
        if not path_exists:
            path_penalty -= float(w["PATH_MISSING"])

        # Penalização "hard" proporcional às violações de zona
        # GATE: quando NÃO existe caminho 0->N-1, NÃO aplicamos a penalidade hard
        # (deixa o agente explorar links/posições para criar conectividade e só depois "limpar" a zona)
        # gate ~0 até reach_frac~0.6, sobe suave e chega em 1 perto de 0.9
        zone_hard_gate = np.clip((reach_frac - 0.6) / (0.9 - 0.6), 0.0, 1.0)

        zone_hard_penalty = 0.0
        if zone_hard_gate > 0.0 and (drones_in_zone > 0 or links_cross_zone > 0):
            total_violacoes_zona = float(drones_in_zone) * float(drone_scale) + float(links_cross_zone) * float(link_scale_eff)
            zone_hard_penalty -= float(w["ZONE_HARD"]) * total_violacoes_zona * float(zone_hard_gate)

        # -------------------------
        # Hops: só contam DEPOIS dos objetivos principais
        # -------------------------
        comp_rel = rel_frac
        comp_path = 1.0 if path_exists else 0.0
        comp_zone = 1.0 if (drones_in_zone == 0 and links_cross_zone == 0) else 0.0
        comp_left = 1.0 if (left_links_count == 0) else 0.0
        comp_dag = 1.0 if is_dag else 0.0

        # Gate de hops: só otimiza hops depois que objetivos primários estão bons
        primary_ok = (comp_rel + comp_path + comp_zone + comp_left + comp_dag) / 5.0
        hops_gate = primary_ok ** 2

        # Curto-circuito: usamos o menor nº de hops (menor caminho em hops) para não punir redundância.
        # max_hops continua sendo logado/medido separadamente.
        min_hops = 0
        if path_exists:
            parent_min = [-1] * n
            q_min = deque([0])
            parent_min[0] = 0
            while q_min:
                u = q_min.popleft()
                if u == n - 1:
                    break
                for v in range(n):
                    if adj_np[u, v] > 0.0 and parent_min[v] == -1:
                        parent_min[v] = u
                        q_min.append(v)
            if parent_min[n - 1] != -1:
                cur = n - 1
                while cur != 0:
                    cur = parent_min[cur]
                    min_hops += 1

        if path_exists and min_hops > 0:
            # Otimiza diretamente pelo MENOR nº de hops (1 é o melhor possível)
            hop_norm = (float(min_hops) - 1.0) / max(1.0, float(n - 1))
            hop_norm = max(0.0, min(1.0, hop_norm))
            hop_term = 1.0 - 2.0 * hop_norm  # ∈ [-1, +1]
            hops_score = float(w["HOPS_WEIGHT"]) * float(hops_gate) * float(hop_term)
        # -------------------------
        # Layout ao longo do caminho 0->N-1 – bônus MUITO pequeno
        # layout_score usa pesos w["LAYOUT_*"]
        # -------------------------
        layout_score = 0.0
        layout_gate = primary_ok ** 3

        if path_exists:
            parent = [-1] * n
            q = deque([0])
            parent[0] = 0
            while q:
                u = q.popleft()
                if u == n - 1:
                    break
                for v in range(n):
                    if adj_np[u, v] > 0.0 and parent[v] == -1:
                        parent[v] = u
                        q.append(v)

            if parent[n - 1] != -1:
                path_nodes = []
                cur = n - 1
                while True:
                    path_nodes.append(cur)
                    if cur == 0:
                        break
                    cur = parent[cur]
                path_nodes = path_nodes[::-1]

                center_x = self.grid_size / 2.0
                x_coords = pos_np[path_nodes, 0]
                y_coords = pos_np[path_nodes, 1]

                dx_center = np.abs(x_coords - center_x)
                max_dx = self.grid_size / 2.0
                align_score = 1.0 - np.clip(dx_center.mean() / max_dx, 0.0, 1.0)

                if len(path_nodes) >= 2:
                    y0 = y_coords[0]
                    yN = y_coords[-1]
                    t = np.linspace(0.0, 1.0, len(path_nodes))
                    y_line = y0 + t * (yN - y0)
                    dev_y = np.abs(y_coords - y_line).mean()
                    max_dev_y = max(1.0, self.grid_size / 3.0)
                    straight_score = 1.0 - np.clip(dev_y / max_dev_y, 0.0, 1.0)
                else:
                    straight_score = 1.0

                out_degrees_path = adj_np.sum(axis=1)[path_nodes]
                redundant_frac = float(np.mean(out_degrees_path >= 2.0))

                layout_score = float(layout_gate) * (
                    float(w["LAYOUT_ALIGN_W"]) * float(align_score)
                    + float(w["LAYOUT_STRAIGHT_W"]) * float(straight_score)
                    + float(w["LAYOUT_REDUND_W"]) * float(redundant_frac)
                )

        # -------------------------
        # -------------------------
        # Caminhos disjuntos (edge-disjoint):
        #   - FULL: caminhos edge-disjoint completos 0->(n-1) (somente se NÃO houver links cruzando zona)
        #   - PARTIAL: rotas paralelas parciais (edge-disjoint) saindo do 0 em direção ao (n-1),
        #              mesmo que ainda não cheguem ao destino.
        # Bônus:
        #   disjoint_bonus = DISJOINT_PATH_BONUS * [(full-1) + DISJOINT_PARTIAL_FACTOR*(partial-1)]
        # -------------------------
        disjoint_paths_full = 0
        disjoint_paths_partial = 0
        disjoint_bonus = 0.0
        # Filtra arestas usadas no bônus de caminhos disjuntos por qualidade do enlace (prob mínima).
        # Isso evita que o bônus incentive arestas muito longas/ruins (ex.: link direto 0->N-1).
        disjoint_p_min = float(w.get("DISJOINT_MIN_P", 0.0))
        dist_m = self._calc_distance_matrix()
        adj_disjoint = self._filter_adj_by_disjoint_quality(adj_np, dist_m, disjoint_p_min)

        # FULL (mantém a regra antiga)
        if path_exists:
            disjoint_paths_full = self._max_edge_disjoint_paths_no_zone_links(adj_disjoint, pos_np, 0, n - 1, pos_round=pos_round, zone_edge_cache=zone_edge_cache)
            if disjoint_paths_full > 1:
                disjoint_bonus += float(w.get("DISJOINT_PATH_BONUS", 0.0)) * min(float(disjoint_paths_full - 1), 10.0)

        # PARTIAL (novo)
        partial_factor = float(w.get("DISJOINT_PARTIAL_FACTOR", 0.5))
        min_prog = float(w.get("DISJOINT_PARTIAL_MIN_PROGRESS", 0.5))
        disjoint_paths_partial = self._max_edge_disjoint_partial_paths(adj_disjoint, pos_np, 0, n - 1,
                                                                       min_progress=min_prog)
        if disjoint_paths_partial > 1 and partial_factor > 0.0:
            disjoint_bonus += float(w.get("DISJOINT_PATH_BONUS", 0.0)) * float(partial_factor) * min(
                float(disjoint_paths_partial - 1), 10.0)

        # Compatibilidade: métrica "disjoint_paths" passa a refletir o melhor dos dois (FULL vs PARTIAL)
        disjoint_paths = int(max(disjoint_paths_full, disjoint_paths_partial))
        # Fitness total
        # -------------------------
        layout_terms = float(w["LAYOUT_MIX"]) * (float(nn_reward) + float(uniform_reward)) + float(layout_score)

        # Comentários "na frente" de cada termo: valor/faixa e peso associado.
        # -------------------------
        # Gate final (quase duro) para termos secundários
        # Só libera quando: confiabilidade >= ~0.9999 (soft), há caminho e zona está limpa.
        # -------------------------
        secondary_gate = float(rel_unlock_gate)
        if not path_exists:
            secondary_gate = 0.0
        if (drones_in_zone > 0) or (links_cross_zone > 0):
            secondary_gate = 0.0

        hops_score *= secondary_gate
        latency_reward *= secondary_gate
        layout_terms *= secondary_gate
        line_align_term *= secondary_gate
        # Bônus de redundância (caminhos disjuntos) NÃO deve depender do gate de objetivos secundários,
        # pois ele ajuda a empurrar a confiabilidade para a região ultra-alta.
        # Desbloqueia gradualmente a partir de 0.90 até 0.9999 (ajuste fino conforme necessário).
        # Gate do bônus de rotas paralelas: ativação "cedo" baseada em conectividade útil,
        # evitando depender de estar perto de 5-nines.
        disjoint_gate = 0.0
        if path_exists and disjoint_bonus != 0.0:
            rff0 = float(w.get("DISJOINT_GATE_RFF_START", 0.85))
            rf0 = float(w.get("DISJOINT_GATE_RF_START", 0.85))
            gate_rff = 0.0 if reach_from_first_frac <= rff0 else (float(reach_from_first_frac) - rff0) / max(1e-9, (
                        1.0 - rff0))
            gate_rf = 0.0 if reach_frac <= rf0 else (float(reach_frac) - rf0) / max(1e-9, (1.0 - rf0))
            disjoint_gate = max(0.0, min(1.0, gate_rff)) * max(0.0, min(1.0, gate_rf))

        disjoint_bonus_raw = float(disjoint_bonus)
        disjoint_bonus = disjoint_bonus_raw * float(disjoint_gate)

        fit = (
            # rel_score = REL_MAIN * (2*rel_log - 1), rel_log ∈ [0,1] => rel_score ∈ [-REL_MAIN, +REL_MAIN]
                rel_score
                # path_penalty = -PATH_MISSING se não há caminho, senão 0
                + path_penalty
                # zone_hard_penalty = -ZONE_HARD * (drones_in_zone + links_cross_zone)
                + zone_hard_penalty
                # cycle_penalty = -CYCLE_PENALTY se houver ciclo no grafo
                + cycle_penalty
                # conn_quality_term = CONN_QUALITY_COEF * connection_quality, connection_quality ∈ [0,1]
                + conn_quality_term
                # reach_last_penalty = -(1 - reach_frac) ∈ [-1, 0] (peso=1)
                + reach_last_penalty
                # unreachable_from_first_penalty = -UNREACHABLE_FROM_FIRST_PENALTY * (#nós não alcançáveis do 0)
                + unreachable_from_first_penalty
                # reach_from_first_penalty = -REACH_FROM_FIRST_W * (1 - reach_from_first_frac)
                + reach_from_first_penalty
                # invalid_relay_penalty: soma de -ISOLATED_NODE_PENALTY por nó isolado (peso=1)
                + invalid_relay_penalty
                # blocked_relay_penalty: penalizações internas (0.2 por ocorrência) (peso=1)
                + blocked_relay_penalty * 100
                # imbalance_penalty: -0.2 * |in_deg - out_deg| acumulado (peso=1)
                + imbalance_penalty * 100
                # dead_end_penalty: -0.2 por dead-end acumulado (peso=1)
                + dead_end_penalty * 100
                # hops_score = HOPS_WEIGHT * hops_gate * hop_diff, hop_diff ∈ [-1,1], hops_gate ∈ [0,1]
                + hops_score
                # latency_penalty: ≤ 0, proporcional ao excesso acima de MAX_LATENCY_MS (peso=1)
                + latency_penalty
                # latency_reward: ≥ 0, proporcional à folga abaixo de MAX_LATENCY_MS (peso=1)
                + latency_reward
                # layout_terms = LAYOUT_MIX*(nn_reward + uniform_reward) + layout_score
                + layout_terms
                # line_align_term = LINE_ALIGN_W*gate_zone*line_align_score  (puxa p/ reta 0->N-1)
                + line_align_term
                # zone_penalty = -ZONE_DRONE_COEF*drones_in_zone - ZONE_LINK_COEF*links_cross_zone
                + zone_penalty
                + zone_fix_bonus
                + zone_clean_once_bonus
                # left_links_penalty = -LEFT_LINK_COEF * left_links_count (desabilitado por padrão: LEFT_LINK_COEF=0.0)
                + left_links_penalty
                # throughput_penalty = -THROUGHPUT_PENALTY_MAX * excess, excess∈[0,1] quando throughput/THROUGHPUT_MAX_MBPS >= 0.85 (desabilitado por padrão)
                + throughput_penalty
                # bottleneck_term: foca em melhorar o pior enlace no caminho 0->N-1
                + bottleneck_term
                # disjoint_bonus (gated)
                + disjoint_bonus
        )
        info = {
            "reliability": float(reliability),
            "bottleneck_min_p": float(bottleneck_min_p),
            "bottleneck_max_dist_m": float(bottleneck_max_dist_m),
            "throughput": float(throughput),
            "num_links": int(num_links),
            "num_links_eff": int(num_links_eff) if 'num_links_eff' in locals() else int(num_links),
            "Emax_n": int(Emax_n),
            "Emax_ref": int(Emax_ref),
            "link_scale": float(link_scale),
            "link_density": float(link_density),
            "link_scale_eff": float(link_scale_eff),
            "drone_scale": float(drone_scale),
            "relay_scale": float(relay_scale),
            "connection_quality": float(connection_quality),
            "reach_frac": float(reach_frac),
            "reach_from_first_frac": float(reach_from_first_frac),
            "unreachable_from_first_count": int(unreachable_from_first_count),
            "reach_from_first_penalty": float(reach_from_first_penalty),
            "path_exists": bool(path_exists),
            "is_dag": int(is_dag),
            "cycle_penalty": float(cycle_penalty),
            "zone_hard_gate": float(zone_hard_gate),
            "max_hops": int(max_hops),
            "avg_hops": float(avg_hops),
            "nn_mean_dist": float(nn_mean_dist),
            "spread_nn_score": float(nn_reward),
            "spread_uniform_score": float(uniform_reward),
            "line_align_score": float(line_align_score),
            "line_align_term": float(line_align_term),
            "drones_in_zone": int(drones_in_zone),
            "links_cross_zone": int(links_cross_zone),
            "zone_fix_bonus": float(zone_fix_bonus),
            "zone_clean_once_bonus": float(zone_clean_once_bonus),
            "left_links_count": int(left_links_count),
            "right_links_count": int(right_links_count),
            "right_links_ratio": float(right_links_ratio),
            "left_links_penalty": float(left_links_penalty),
            "left_links_gate": int(left_links_gate),
            "throughput_penalty": float(throughput_penalty),
            "num_drones": int(self.num_drones),
            "max_drones": int(self.max_drones),
            "disjoint_paths": int(disjoint_paths),
            "disjoint_paths_full": int(disjoint_paths_full),
            "disjoint_paths_partial": int(disjoint_paths_partial),
            "disjoint_bonus_raw": float(disjoint_bonus_raw),
            "disjoint_bonus": float(disjoint_bonus),
            "disjoint_gate": float(disjoint_gate),
        }

        # Opcional: expõe contribuições reais do fit no info (bom para TensorBoard/debug)
        if self.log_fit_terms:
            info.update({
                "fit_rel_score": float(rel_score),
                "fit_rel_unlock_gate": float(rel_unlock_gate),
                "fit_secondary_gate": float(secondary_gate),
                "fit_path_penalty": float(path_penalty),
                "fit_zone_hard_penalty": float(zone_hard_penalty),
                "fit_cycle_penalty": float(cycle_penalty),
                "is_dag": int(is_dag),
                "fit_zone_hard_gate": float(zone_hard_gate),
                "fit_conn_quality": float(conn_quality_term),
                "fit_reach_last_penalty": float(reach_last_penalty),
                "fit_unreachable_from_first_penalty": float(unreachable_from_first_penalty),
                "fit_reach_from_first_penalty": float(reach_from_first_penalty),
                "fit_invalid_relay_penalty": float(invalid_relay_penalty),
                "fit_blocked_relay_penalty": float(blocked_relay_penalty),
                "fit_imbalance_penalty": float(imbalance_penalty),
                "fit_dead_end_penalty": float(dead_end_penalty),
                "fit_hops_score": float(hops_score),
                "fit_latency_penalty": float(latency_penalty),
                "fit_latency_reward": float(latency_reward),
                "fit_layout_terms": float(layout_terms),
                "fit_line_align_term": float(line_align_term),
                "fit_zone_penalty": float(zone_penalty),
                "fit_left_links_penalty": float(left_links_penalty),
                "fit_left_links_gate": int(left_links_gate),
                "fit_throughput_penalty": float(throughput_penalty),
                "fit_total": float(fit),
            })

        return float(fit), info

    @staticmethod
    def _reachability_bits(adj_on: np.ndarray) -> np.ndarray:
        """Transitive closure (alcançabilidade) usando bitsets.
        Retorna reach[i] como uint32 com bit j=1 se j é alcançável a partir de i.
        Assumimos n<=32 (no env, n é pequeno).
        """
        n = int(adj_on.shape[0])
        reach = np.zeros(n, dtype=np.uint32)
        # Inicial: vizinhos diretos + self
        for i in range(n):
            bits = np.uint32(1 << i)
            js = np.nonzero(adj_on[i])[0]
            for j in js.tolist():
                bits |= np.uint32(1 << int(j))
            reach[i] = bits
        # Warshall com bitsets
        for k in range(n):
            kbit = np.uint32(1 << k)
            rk = reach[k]
            for i in range(n):
                if (reach[i] & kbit) != 0:
                    reach[i] |= rk
        return reach

    # =========================
    # Action masking (sb3-contrib)
    # =========================
    def action_masks(self) -> np.ndarray:
        """Máscara booleana (achatada) compatível com MultiDiscrete.

        Ordem: concatena máscaras de cada dimensão do MultiDiscrete.
        - move_op: 0 = não mover; demais codificam (move_idx, step_code, dir_code).
          Aqui mascaramos apenas:
            * drones inativos
            * âncoras (0 e N-1)
        - toggle_op: permite ADD/REMOVE válidos no estado atual, e pode aplicar hard-masks
          (zona, left-link e ciclos) para impor restrições como *hard constraints*.
        """
        if self._mask_cache is not None and self._mask_cache_version == self._mask_version:
            return self._mask_cache

        nvec = np.asarray(self.action_space.nvec, dtype=np.int64)
        n = int(self.num_drones)

        # --- move_op ---
        m_moveop = np.zeros(int(nvec[0]), dtype=bool)
        m_moveop[0] = True  # no move

        if n > 2:
            per_drone = int(STEP_CELLS_N) * 4
            # drones intermediários ativos: [1 .. n-2]
            max_i = min(int(n - 2), int(self.max_drones) - 1)
            for i in range(1, max_i + 1):
                s = 1 + i * per_drone
                e = s + per_drone
                if s < m_moveop.shape[0]:
                    m_moveop[s:min(e, m_moveop.shape[0])] = True

        # --- toggle_op ---
        E = int(self._E)
        m_toggle = np.zeros(int(nvec[1]), dtype=bool)
        m_toggle[0] = True

        # Pré-definições para evitar NameError quando n<2
        src_all = self._edge_src
        dst_all = self._edge_dst
        idx_add = np.array([], dtype=np.int64)
        pos_np = None

        if n >= 2:
            # Views CPU->numpy (sem cópia) + lógica vetorizada: evita loop Python por aresta.
            adj_on = (_torch_to_numpy_cpu_view(self.adjacency_matrix[:n, :n]) > 0.5)
            out_deg = adj_on.sum(axis=1).astype(np.int32, copy=False)

            src_all = self._edge_src
            dst_all = self._edge_dst
            active = (src_all < n) & (dst_all < n)
            if np.any(active):
                idx_active = np.nonzero(active)[0]
                src_a = src_all[idx_active]
                dst_a = dst_all[idx_active]

                is_on = adj_on[src_a, dst_a]

                # REMOVE: válido se existe
                idx_rem = idx_active[is_on]
                if idx_rem.size:
                    m_toggle[1 + E + idx_rem] = True

                # ADD: válido se NÃO existe e respeita regra hard de max_out
                add_ok = (~is_on) & (out_deg[src_a] < int(self.max_out_connections))
                idx_add = idx_active[add_ok]
                if idx_add.size:
                    # (mask-relax) não bloqueia left-link nem zona
                    m_toggle[1 + idx_add] = True

                    # link direto 0<->N-1 só se p_dir >= limiar (quando limiar > 0)
                    p_min_direct = float(self.fit_w.get("DIRECT_0N1_MIN_P", 0.0))
                    if p_min_direct > 0.0 and n > 2:
                        src_add = src_all[idx_add]
                        dst_add = dst_all[idx_add]
                        direct_mask = (
                            ((src_add == 0) & (dst_add == (n - 1))) |
                            ((src_add == (n - 1)) & (dst_add == 0))
                        )
                        if np.any(direct_mask):
                            pos_np = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2]).astype(np.float32, copy=False)
                            for edge_idx in idx_add[direct_mask].tolist():
                                s = int(src_all[edge_idx])
                                d = int(dst_all[edge_idx])
                                dx_m = (float(pos_np[d, 0]) - float(pos_np[s, 0])) * float(self.square_size_m)
                                dy_m = (float(pos_np[d, 1]) - float(pos_np[s, 1])) * float(self.square_size_m)
                                d_m = float(math.hypot(dx_m, dy_m))
                                p_dir = float(self._prob_link(d_m))
                                if p_dir < p_min_direct:
                                    m_toggle[1 + edge_idx] = False

        # 4b) hard-mask (left-links): bloqueia ADD quando x_dst < x_src
        if self.hard_mask_left_links and idx_add.size:
            if pos_np is None:
                pos_np = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2]).astype(np.float32, copy=False)
            src_add = src_all[idx_add]
            dst_add = dst_all[idx_add]
            left = (pos_np[dst_add, 0] < pos_np[src_add, 0])
            if np.any(left):
                m_toggle[1 + idx_add[left]] = False

        # 4c) hard-mask (ciclos): bloqueia ADD que criaria ciclo (dst alcança src)
        if self.hard_mask_cycles and idx_add.size:
            enabled = m_toggle[1 + idx_add]
            if np.any(enabled):
                idx_chk = idx_add[enabled]
                src_chk = src_all[idx_chk]
                dst_chk = dst_all[idx_chk]
                reach = self._reachability_bits(adj_on)
                reach_dst = reach[dst_chk]
                cyc = ((reach_dst >> src_chk) & np.uint32(1)).astype(bool)
                if np.any(cyc):
                    m_toggle[1 + idx_chk[cyc]] = False

        # 5) hard-mask (zona): bloqueia ADD de arestas que cruzam zona proibida
        if self.hard_mask_zone_edges and self.zonas_proibidas and idx_add.size:
            cross_mat = self._ensure_edge_cross_mat()
            if cross_mat is not None:
                src_add = src_all[idx_add]
                dst_add = dst_all[idx_add]
                cross = cross_mat[src_add, dst_add].astype(bool, copy=False)
                if np.any(cross):
                    m_toggle[1 + idx_add[cross]] = False

        mask = np.concatenate([m_moveop, m_toggle]).astype(bool, copy=False)
        self._mask_cache = mask
        self._mask_cache_version = self._mask_version
        return mask
