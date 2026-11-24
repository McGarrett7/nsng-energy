"""Trust-aware neuro-symbolic negotiation (Sections III-IV, Algorithm 1).

Tensors are batched as ``[sequence, time, ...]``. Algorithm 1 visits entities in time
order and keeps the trust vector between them, so Eq. (9) carries tau^(t) -> tau^(t+1).
Sequences are independent and are negotiated in parallel; time steps are sequential.
"""

from dataclasses import dataclass, fields, replace

import numpy as np
from scipy.signal import lfilter
import torch
from torch import nn

from .agents import Beliefs

INITIAL_TRUST = 0.8
SIGMA_SUPPORT = SIGMA_CONFLICT = 0.2   # Eq. (4), Eq. (5)
GAMMA, LAMBDA = 0.5, 1.0               # Eq. (2): delta = gamma * min(c) * exp(-lambda * kappa)
ETA = 1.0                              # Eq. (6)
THETA_HIGH, THETA_LOW = 0.7, 0.3       # Section III.C: direct contradiction
HISTORY_DECAY = 0.95                   # EMA for kappa_ij and d_ij
AGREEMENT_TOLERANCE = 0.2              # an agreement event for kappa_ij
R1_FLOOR, R2_JUMP, R2_FACTOR = 0.8, 0.3, 0.5
R3_CORRELATION, R3_GAP = 0.9, 0.2
R4_CONFLICTS, R4_DECAY = 5, 0.9


@dataclass(frozen=True)
class Settings:
    rounds: int = 12                   # K_max
    momentum: float = 0.9              # lambda_tau
    symbolic_strength: float = 0.4     # lambda_sym
    tolerance: float = 1e-4            # epsilon_conv
    heads: int = 4
    top_k: int = 5
    rules: tuple[str, ...] = ("R1", "R2", "R3", "R4")
    trust: bool = True
    nominal_voltage: float | None = None
    voltage_rule: str = "soft"
    detector_residual: float = 0.0     # 0 follows Eq. (13) exactly


@dataclass
class GraphInputs:
    score: torch.Tensor         # [B, T, N]
    confidence: torch.Tensor    # [B, T, N]
    agreement: torch.Tensor     # [B, T, N, N] kappa_ij from past time steps
    distance: torch.Tensor      # [B, T, N, N] d_ij from past time steps
    correlated: torch.Tensor    # [B, T, N, N] corr(s_i, s_j) > 0.9 over the past
    high_voltage: torch.Tensor  # [B, T]
    valid: torch.Tensor         # [B, T] False on padding

    def select_agents(self, indices: list[int]) -> "GraphInputs":
        index = torch.as_tensor(indices, device=self.score.device)
        pair = lambda values: values.index_select(-1, index).index_select(-2, index)
        return replace(
            self, score=self.score.index_select(-1, index),
            confidence=self.confidence.index_select(-1, index),
            agreement=pair(self.agreement), distance=pair(self.distance), correlated=pair(self.correlated),
        )

    def take(self, window) -> "GraphInputs":
        return GraphInputs(*(getattr(self, field.name)[:, window] for field in fields(self)))

    @staticmethod
    def stack(items: list["GraphInputs"]) -> "GraphInputs":
        length = max(item.score.shape[1] for item in items)
        padded = {}
        for field in fields(GraphInputs):
            parts = []
            for item in items:
                values = getattr(item, field.name)
                missing = length - values.shape[1]
                if missing:
                    filler = values[:, -1:].expand(-1, missing, *values.shape[2:])
                    if field.name == "valid":
                        filler = torch.zeros_like(filler)
                    values = torch.cat((values, filler), dim=1)
                parts.append(values)
            padded[field.name] = torch.cat(parts)
        return GraphInputs(**padded)


@dataclass
class State:
    """What Algorithm 1 carries from entity x_(t-1) to x_t."""

    trust: torch.Tensor         # [B, N]
    scores: torch.Tensor        # [K+1, B, N] round states of the previous time step (R2)
    confidence: torch.Tensor    # [K+1, B, N]
    started: torch.Tensor       # [B]

    def detach(self) -> "State":
        return State(*(getattr(self, field.name).detach() for field in fields(self)))


def _lagged(values: np.ndarray, prior: float) -> np.ndarray:
    return np.concatenate((np.full_like(values[:1], prior), values[:-1]), axis=0)


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.concatenate((np.zeros_like(values[:1]), np.cumsum(values, axis=0)))
    previous = np.maximum(np.arange(1, len(cumulative)) - window, 0)
    return cumulative[1:] - cumulative[previous]


def graph_inputs(beliefs: Beliefs, voltage, nominal_voltage, device, history: int = 32) -> GraphInputs:
    """Causal pairwise history for one sequence, returned with a batch axis of one."""
    scores = beliefs.scores.astype(np.float64)
    differences = np.abs(scores[:, :, None] - scores[:, None, :])
    agree = _lagged((differences <= AGREEMENT_TOLERANCE).astype(float), 0.5)
    distance = _lagged(differences, 0.5)
    for values in (agree, distance):
        values[:] = lfilter([1 - HISTORY_DECAY], [1, -HISTORY_DECAY], values, axis=0,
                            zi=np.full_like(values[:1], HISTORY_DECAY * 0.5))[0]
    # Pearson correlation over at most ``history`` past points (Rule R3).
    past = _lagged(scores, 0.0)
    count = np.clip(np.arange(len(scores)), 1, history)
    total = _rolling_sum(past, history)
    centered = _rolling_sum(past**2, history) - total**2 / count[:, None]
    cross = _rolling_sum(past[:, :, None] * past[:, None, :], history)
    cross -= total[:, :, None] * total[:, None, :] / count[:, None, None]
    scale = np.sqrt(np.maximum(centered, 0))
    product = scale[:, :, None] * scale[:, None, :]
    correlated = (cross > R3_CORRELATION * product) & (product > 1e-10)
    correlated[:4] = False
    high_voltage = np.zeros(len(scores), dtype=bool)
    if voltage is not None and nominal_voltage is not None:
        high_voltage = np.nan_to_num(voltage, nan=0.0) > 1.05 * nominal_voltage
    as_tensor = lambda values, dtype: torch.as_tensor(values, dtype=dtype, device=device)[None]
    return GraphInputs(
        *(as_tensor(values, torch.float32) for values in (scores, beliefs.confidence, agree, distance)),
        as_tensor(correlated, torch.bool), as_tensor(high_voltage, torch.bool),
        torch.ones((1, len(scores)), dtype=torch.bool, device=device),
    )


class GraphAttention(nn.Module):
    """Eqs. (10)-(12): multi-head GAT over incoming negotiation edges and the self loop."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError("The embedding width must be divisible by the number of heads.")
        self.heads = heads
        self.project = nn.Linear(width, width, bias=False)
        self.query = nn.Parameter(torch.empty(heads, width // heads))
        self.key = nn.Parameter(torch.empty(heads, width // heads))
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.key)

    def forward(self, embedding: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        nodes = self.project(embedding).unflatten(-1, (self.heads, -1))
        query = (nodes * self.query).sum(dim=-1)
        key = (nodes * self.key).sum(dim=-1)
        logits = nn.functional.leaky_relu(query[:, :, None, :] + key[:, None, :, :], 0.2)
        logits = logits.masked_fill(~edges.transpose(1, 2)[..., None], -torch.inf)
        attention = logits.permute(0, 3, 1, 2).softmax(dim=-1)
        messages = attention @ nodes.transpose(1, 2)
        return torch.sigmoid(messages.transpose(1, 2)).flatten(-2)


class Negotiator(nn.Module):
    def __init__(self, settings: Settings, width: int = 16):
        super().__init__()
        self.settings = settings
        self.mix = nn.Parameter(torch.zeros(3))                       # alpha, beta, gamma of Eq. (7)
        self.edge_types = nn.Parameter(torch.randn(3, 4) * 0.1)      # e_ji of Eq. (8)
        self.embed = nn.Sequential(nn.Linear(7, width), nn.Tanh(), nn.Linear(width, width))
        self.attention = GraphAttention(width, settings.heads)
        self.score_head = nn.Sequential(nn.Linear(width, 8), nn.Tanh(), nn.Linear(8, 1))
        self.confidence_head = nn.Sequential(nn.Linear(width, 8), nn.Tanh(), nn.Linear(8, 1))

    def initial_state(self, batch: int, agents: int, config: Settings, device) -> State:
        empty = torch.full((config.rounds + 1, batch, agents), 0.5, device=device)
        return State(torch.full((batch, agents), INITIAL_TRUST, device=device), empty, empty.clone(),
                     torch.zeros(batch, dtype=torch.bool, device=device))

    def forward(self, inputs: GraphInputs, settings: Settings | None = None,
                state: State | None = None, trace: bool = False):
        """Negotiate every time step in order. Returns s_*, c_*, details and the carried state."""
        config = settings or self.settings
        batch, length, agents = inputs.score.shape
        if state is None:
            state = self.initial_state(batch, agents, config, inputs.score.device)
        steps = []
        for t in range(length):
            result, state = self._step(inputs.take(t), state, config, trace)
            steps.append(result)
        consensus = torch.stack([step.pop("consensus") for step in steps], dim=1)
        confidence = torch.stack([step.pop("confidence") for step in steps], dim=1)
        details = {key: torch.stack([step[key] for step in steps], dim=1) for key in steps[0]}
        return consensus, confidence, details, state

    def _step(self, x: GraphInputs, state: State, config: Settings, trace: bool):
        score, confidence, valid = x.score, x.confidence, x.valid
        agents = score.shape[1]
        eye = torch.eye(agents, dtype=torch.bool, device=score.device)
        trust = state.trust if config.trust else torch.full_like(score, INITIAL_TRUST)

        # Algorithm 1 line 2: the negotiation graph for entity x (Eqs. 4-7).
        differences = (score[:, :, None] - score[:, None, :]).abs()
        support = torch.exp(-differences.square() / (2 * SIGMA_SUPPORT**2))
        support = support * (confidence[:, :, None] * confidence[:, None, :]).sqrt()
        conflict = (1 - torch.exp(-differences.square() / (2 * SIGMA_CONFLICT**2)))
        conflict = conflict * torch.maximum(confidence[:, :, None], confidence[:, None, :])
        historical = trust[:, None, :] * x.agreement * torch.exp(-ETA * x.distance)
        weights = self.mix.softmax(dim=0)
        base = weights[0] * support + weights[1] * conflict + weights[2] * historical
        neighbors = torch.zeros_like(base, dtype=torch.bool)
        top_k = min(config.top_k, agents - 1)
        # Top-K incoming edges for every target agent j (Section IV.B).
        neighbors.scatter_(1, base.masked_fill(eye, -torch.inf).topk(top_k, dim=1).indices, True)
        decay = GAMMA * torch.exp(-LAMBDA * x.agreement)

        def conflicts(s, c):
            gap = (s[:, :, None] - s[:, None, :]).abs()
            marked = (gap > decay * torch.minimum(c[:, :, None], c[:, None, :])) & ~eye   # Eqs. (1)-(2)
            high, low = s >= THETA_HIGH, s <= THETA_LOW
            direct = marked & ((high[:, :, None] & low[:, None, :]) | (low[:, :, None] & high[:, None, :]))
            return gap, marked, direct

        def correlation_conflicts(gap):
            if "R3" not in config.rules:
                return torch.zeros_like(neighbors)
            return x.correlated & (gap > R3_GAP) & ~eye

        _, initial_marked, initial_direct = conflicts(score, confidence)
        initial_marked = initial_marked | correlation_conflicts(differences)
        base_logits = torch.logit(score.clamp(0.001, 0.999))
        active = valid.clone()
        rounds = torch.zeros(len(score), dtype=torch.int16, device=score.device)
        history_scores, history_confidence = [score], [confidence]
        states, edges, marks, rule_hits = [], [], [], []
        if trace:
            states.append(torch.stack((score, confidence, trust), dim=-1))

        for k in range(config.rounds):
            gap, marked, _ = conflicts(score, confidence)
            r3 = correlation_conflicts(gap)
            marked = marked | r3
            selected = neighbors | r3
            relation = torch.stack((support, torch.where(r3, 1.0, conflict), historical), dim=-1)
            strength = relation @ weights
            isolated = ~selected.any(dim=-1, keepdim=True) & eye
            outgoing = strength.masked_fill(~selected, 0).masked_fill(isolated, 1)
            outgoing = outgoing / outgoing.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            exchanges = outgoing.masked_fill(eye, 0)
            edge_context = (exchanges[..., None] * relation).sum(dim=1) @ self.edge_types
            beliefs = torch.stack((score, confidence, trust), dim=-1)
            latent = self.attention(self.embed(torch.cat((beliefs, edge_context), -1)), selected | eye)
            # Eq. (13); a nonzero detector_residual keeps part of the detector logit.
            new_score = (self.score_head(latent).squeeze(-1) + config.detector_residual * base_logits).sigmoid()
            new_confidence = self.confidence_head(latent).squeeze(-1).sigmoid()

            r1 = x.high_voltage if "R1" in config.rules else torch.zeros_like(valid)
            r1 = r1[:, None] & (new_score < R1_FLOOR)
            if config.voltage_rule == "hard":
                new_score = torch.where(r1, R1_FLOOR, new_score)                          # Table II
            else:
                new_score = new_score + config.symbolic_strength * r1 * (R1_FLOOR - new_score)  # Eq. (16)
            r2 = torch.zeros_like(r1)
            if "R2" in config.rules:
                previous = min(k, len(state.scores) - 1)
                r2 = state.started[:, None] & ((score - state.scores[previous]).abs() > R2_JUMP)
                new_confidence = torch.where(r2, R2_FACTOR * state.confidence[previous], new_confidence)
            r4 = torch.zeros_like(r1)
            new_trust = trust
            if config.trust:
                consistency = exchanges * (1 - (new_score[:, :, None] - new_score[:, None, :]).abs())
                degree = selected.sum(dim=-1).clamp_min(1)
                new_trust = (config.momentum * trust                                     # Eq. (9)
                             + (1 - config.momentum) * (consistency.sum(dim=-1) / degree).sigmoid())
                if "R4" in config.rules:
                    r4 = marked.sum(dim=-1) > R4_CONFLICTS
                    new_trust = torch.where(r4, R4_DECAY * new_trust, new_trust)
            change = (new_score - score).abs().amax(dim=-1)
            keep = active[:, None]
            score = torch.where(keep, new_score, score)
            confidence = torch.where(keep, new_confidence, confidence)
            trust = torch.where(keep, new_trust, trust)
            rounds = rounds + active
            history_scores.append(score)
            history_confidence.append(confidence)
            if trace:
                states.append(torch.stack((score, confidence, trust), dim=-1))
                edges.append(outgoing * keep[..., None])
                marks.append(marked & keep[..., None])
                rule_hits.append(torch.stack((r1, r2, r3.any(dim=-1), r4), dim=-1) & keep[..., None])
            active = active & (change >= config.tolerance)
            if not self.training and not active.any():
                break

        # Eqs. (18)-(19).
        consensus_weights = confidence * trust
        denominator = consensus_weights.sum(dim=-1) + 1e-8
        consensus = (consensus_weights * score).sum(dim=-1) / denominator
        agreement = 1 - (score - consensus[:, None]).abs() / 2
        consensus_confidence = (consensus_weights * agreement).sum(dim=-1) / denominator

        gap, final_marked, final_direct = conflicts(score, confidence)
        final_marked = final_marked | correlation_conflicts(gap)
        result = {
            "consensus": consensus, "confidence": consensus_confidence, "rounds": rounds,
            "initial_conflicts": initial_marked.sum(dim=(1, 2)) // 2,
            "final_conflicts": final_marked.sum(dim=(1, 2)) // 2,
            "initial_direct_conflicts": initial_direct.sum(dim=(1, 2)) // 2,
            "final_direct_conflicts": final_direct.sum(dim=(1, 2)) // 2,
            "trust": trust,
        }
        if trace:
            missing = config.rounds - len(edges)
            states.extend([states[-1]] * missing)
            edges.extend([torch.zeros_like(outgoing)] * missing)
            marks.extend([torch.zeros_like(neighbors)] * missing)
            rule_hits.extend([torch.zeros_like(rule_hits[-1]) if rule_hits else
                              torch.zeros((*score.shape, 4), dtype=torch.bool, device=score.device)] * missing)
            result.update(states=torch.stack(states, dim=1), outgoing=torch.stack(edges, dim=1),
                          conflict_edges=torch.stack(marks, dim=1), rule_hits=torch.stack(rule_hits, dim=1))

        # Carry tau, and the round states used by R2, to the next entity of each sequence.
        history_scores += [history_scores[-1]] * (config.rounds + 1 - len(history_scores))
        history_confidence += [history_confidence[-1]] * (config.rounds + 1 - len(history_confidence))
        carried = valid[None, :, None]
        next_state = State(
            torch.where(valid[:, None], trust, state.trust) if config.trust else state.trust,
            torch.where(carried, torch.stack(history_scores), state.scores),
            torch.where(carried, torch.stack(history_confidence), state.confidence),
            state.started | valid,
        )
        return result, next_state
