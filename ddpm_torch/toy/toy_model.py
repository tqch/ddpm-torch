import torch
import torch.nn as nn
try:
    from ..functions import get_timestep_embedding
    from ..modules import Linear, Sequential
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ddpm_torch.functions import get_timestep_embedding
    from ddpm_torch.modules import Linear, Sequential

DEFAULT_NORMALIZER = nn.LayerNorm
DEFAULT_NONLINEARITY = nn.LeakyReLU(negative_slope=0.02, inplace=True)


class TemporalLayer(nn.Module):
    normalize = DEFAULT_NORMALIZER
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(self, in_features, out_features, temporal_features):
        super(TemporalLayer, self).__init__()
        self.norm1 = self.normalize(in_features)
        self.fc1 = Linear(in_features, out_features, bias=False)
        self.norm2 = self.normalize(out_features)
        self.fc2 = Linear(out_features, out_features, bias=False)
        self.enc = Linear(temporal_features, out_features)

        self.skip = nn.Identity() if in_features == out_features else Linear(in_features, out_features, bias=False)

    def forward(self, x, t_emb):
        out = self.fc1(self.nonlinearity(self.norm1(x)))
        out += self.enc(t_emb)
        out = self.fc2(self.nonlinearity(self.norm2(out)))
        skip = self.skip(x)
        return out + skip


class Decoder(nn.Module):
    normalize = DEFAULT_NORMALIZER
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(self, in_features, mid_features, num_temporal_layers):
        super(Decoder, self).__init__()

        self.in_fc = Linear(in_features, mid_features, bias=False)
        self.temp_fc = Sequential(*([TemporalLayer(
            mid_features, mid_features, mid_features), ] * num_temporal_layers))
        self.out_norm = self.normalize(mid_features)
        self.out_fc = Linear(mid_features, in_features)
        self.t_proj = nn.Sequential(
            Linear(mid_features, mid_features),
            self.nonlinearity)
        self.mid_features = mid_features

    def forward(self, x, t):
        t_emb = get_timestep_embedding(t, self.mid_features)
        t_emb = self.t_proj(t_emb)
        out = self.in_fc(x)
        out = self.temp_fc(out, t_emb=t_emb)
        out = self.out_fc(self.out_norm(out))
        return out


if __name__ == "__main__":
    model = Decoder(2, 128, 2)
    print(model, flush=True)
    out = model(torch.randn(16, 2), t=torch.randint(1000, size=(16, )))
    print(out.shape)
