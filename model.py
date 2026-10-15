"""STFNN: estimates a station's value by integrating a learned field gradient from its neighbors."""

import torch as th
from torch import nn

TIME_PERIODS_H = (365 * 24, 30.5 * 24, 7 * 24, 24)
ST_DIM = 2 + 2 * len(TIME_PERIODS_H)
N_CONTINUOUS = 10
# vocabulary sizes of wind direction, weather, hour, weekday, holiday; code 0 marks missing
DISCRETE_VOCAB = (26, 18, 25, 8, 3)


def st_features(st: th.Tensor) -> th.Tensor:
    """(x, y, t) -> [x, y, sin(2 pi t / P), cos(2 pi t / P)] for the periods P in TIME_PERIODS_H.

    Args:
        st: (B, L, 3) float, t in hours.

    Returns:
        (B, L, ST_DIM) float.
    """
    phase = 2 * th.pi * st[..., 2:] / st.new_tensor(TIME_PERIODS_H)  # (B, L, P)
    return th.cat([st[..., :2], phase.sin(), phase.cos()], dim=-1)


def decoder(d_model: int, cfg, num_layers: int) -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(
        d_model=d_model, nhead=cfg.nhead, dim_feedforward=cfg.dim_feed,
        activation="gelu", batch_first=True,
    )
    return nn.TransformerDecoder(layer, num_layers=num_layers)


class FeatFusion(nn.Module):
    """Field gradient (B, L, 3) at each neighbor from its coordinates and features."""

    def __init__(self, cfg):
        super().__init__()
        dims = (cfg.wth_embed_dim,) * 2 + (cfg.time_embed_dim,) * 3
        self.codes = nn.ModuleList(nn.Embedding(v, d) for v, d in zip(DISCRETE_VOCAB, dims))
        in_dim = ST_DIM + N_CONTINUOUS + sum(dims)
        self.head = nn.Sequential(nn.Linear(in_dim, cfg.hid_dim), nn.GELU())
        self.body = nn.Sequential(*(
            m for _ in range(cfg.hid_layers)
            for m in (nn.Linear(cfg.hid_dim, cfg.hid_dim), nn.GELU())
        ))
        self.tail = nn.Linear(cfg.hid_dim, 3)

    def forward(self, st: th.Tensor, feat: th.Tensor) -> th.Tensor:
        """st: (B, L, 3); feat: (B, L, 15) = 10 continuous + 5 discrete codes."""
        disc = feat[..., N_CONTINUOUS:].long()
        codes = [emb(disc[..., i]) for i, emb in enumerate(self.codes)]
        h = self.head(th.cat([st_features(st), feat[..., :N_CONTINUOUS], *codes], dim=-1))
        return self.tail(h + self.body(h))


class RingEstimation(nn.Module):
    """Gradient at the next ring position from the previous gradient."""

    def __init__(self, cfg):
        super().__init__()
        self.step_embed = nn.Embedding(cfg.max_step + 1, cfg.step_embed_dim)
        d_model = ST_DIM + cfg.step_embed_dim
        self.net = decoder(d_model, cfg, cfg.ring_layers)
        self.act = nn.GELU()
        self.linear = nn.Linear(d_model, 3)

    def forward(self, memory: th.Tensor, end_st: th.Tensor, step: int) -> th.Tensor:
        """memory: (B, L, 3) previous gradient; end_st: (B, L, 3) next position -> (B, L, 3)."""
        assert memory.shape == end_st.shape and memory.shape[-1] == 3, f"{memory.shape} vs {end_st.shape}"
        step_code = self.step_embed.weight[step].expand(*memory.shape[:2], -1)
        tgt, mem = (th.cat([st_features(x), step_code], dim=-1) for x in (end_st, memory))
        return self.linear(self.act(self.net(tgt, mem)))


class STAttention(nn.Module):
    """Neighbor weights (B, L), softmax over L, from query and neighbor coordinates."""

    def __init__(self, cfg):
        super().__init__()
        self.decoder = decoder(ST_DIM, cfg, cfg.weight_layers)
        self.linear = nn.Linear(ST_DIM, 1)

    def forward(self, query_st: th.Tensor, ngb_st: th.Tensor) -> th.Tensor:
        h = self.decoder(tgt=st_features(ngb_st), memory=st_features(query_st))
        return self.linear(h).squeeze(-1).softmax(dim=1)


def idw_weights(dist: th.Tensor, decay: float = 0.5, eps: float = 1e-12) -> th.Tensor:
    """Inverse-distance weights with geometric decay over window steps.

    w_{w,k} = decay^{w+1} * (1 / d_{w,k}) / sum_k (1 / d_{w,k}), normalized to sum to 1 over (w, k).
    `dist` holds squared distances, so this is IDW with power 2; infinite distance gives weight 0.

    Args:
        dist: (B, W, K) float squared distances.

    Returns:
        (B, W * K) float, step-major like the neighbor axis L.
    """
    inv = dist.float().clamp_min(eps).reciprocal()
    spatial = inv / inv.sum(dim=2, keepdim=True).clamp_min(eps)
    steps = decay ** th.arange(1, dist.shape[1] + 1, device=dist.device, dtype=spatial.dtype)
    w = (spatial * steps[:, None]).flatten(1)
    return w / w.sum(dim=1, keepdim=True).clamp_min(eps)


def curl_penalty(jac: th.Tensor) -> th.Tensor:
    """Median over (S, B, L) of ||J - J^T||_F^2 for jac (S, B, L, 3, 3); non-finite entries count as 0."""
    asym = jac - jac.transpose(-1, -2)
    asym = th.where(asym.isfinite(), asym, th.zeros_like(asym))
    return asym.square().sum(dim=(-1, -2)).median()


class STFNN(nn.Module):
    """Query value = sum_l w_l * (y_l + integral of the gradient from neighbor l to the query)."""

    def __init__(self, cfg):
        super().__init__()
        self.max_step = cfg.max_step
        self.read_out = FeatFusion(cfg)
        self.ring_est = RingEstimation(cfg)
        self.st_weight = STAttention(cfg) if cfg.dym_graph else None

    def integrate(self, start: th.Tensor, grad0: th.Tensor, end: th.Tensor):
        """Value change from `start` to `end` (B, L, 3) in max_step rings -> ((B, L), curl ())."""
        step_vec = (end - start) / self.max_step  # (B, L, 3)
        unit = step_vec / step_vec.norm(dim=2, keepdim=True)
        grads, pos = [grad0], start
        for step in range(self.max_step):
            pos = pos + step_vec
            grads.append(self.ring_est(grads[-1], pos, step))
        grads = th.stack(grads)  # (S + 1, B, L, 3)
        delta = (grads.sum(dim=0) * unit).sum(dim=2)

        # finite-difference Jacobian: (S, B, L, 3, 1) * (1, B, L, 1, 3) -> (S, B, L, 3, 3)
        jac = (grads[1:] - grads[:-1]).unsqueeze(-1) * step_vec.reciprocal().unsqueeze(-2)
        return delta, curl_penalty(jac)

    def forward(self, query_st: th.Tensor, ngb: th.Tensor, dist: th.Tensor):
        """Predicts the normalized label of each query.

        Args:
            query_st: (B, 3) query coordinates and hour.
            ngb: (B, L, 3 + F) neighbor coordinates, hour, features and label (last).
            dist: (B, W, K) squared query-neighbor distances, L = W * K.

        Returns:
            pred: (B,) float; curl: () float curl penalty.
        """
        ngb_st, ngb_feat, ngb_label = ngb[..., :3], ngb[..., 3:-1], ngb[..., -1]
        query_st = query_st.unsqueeze(1).expand_as(ngb_st)
        delta, curl = self.integrate(ngb_st, self.read_out(ngb_st, ngb_feat), query_st)
        weight = idw_weights(dist) if self.st_weight is None else self.st_weight(query_st, ngb_st)
        return ((ngb_label + delta) * weight).sum(dim=1), curl
