import torch
import torch.nn as nn
from modules import Linear, Sequential
from functions import get_timestep_embedding


DEFAULT_NORMALIZER = nn.LayerNorm
DEFAULT_NONLINEARITY = nn.ReLU(inplace=True)


class TemporalLayer(nn.Module):
    normalize = DEFAULT_NORMALIZER
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(self, in_features, out_features, temporal_features):
        super(TemporalLayer, self).__init__()
        self.bn1 = self.normalize(in_features)
        self.fc1 = Linear(in_features, out_features, bias=False)
        self.bn2 = self.normalize(out_features)
        self.fc2 = Linear(out_features, out_features, bias=False)
        self.enc = Linear(temporal_features, out_features)

        self.skip = nn.Identity() if in_features == out_features else Linear(in_features, out_features, bias=False)

    def forward(self, x, t_emb):
        out = self.fc1(self.nonlinearity(self.bn1(x)))
        out += self.enc(t_emb)
        out = self.fc2(self.nonlinearity(self.bn2(out)))
        skip = self.skip(x)
        return out + skip


class Decoder(nn.Module):
    normalize = DEFAULT_NORMALIZER
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(self, in_features, mid_features, num_temporal_layers):
        super(Decoder, self).__init__()

        self.fc_in = Linear(in_features, mid_features, bias=False)
        self.fc_temp = Sequential(*([TemporalLayer(
            mid_features, mid_features, mid_features), ] * num_temporal_layers))
        self.bn_out = self.normalize(mid_features)
        self.fc_out = Linear(mid_features, in_features)
        self.emb = nn.Sequential(
            Linear(mid_features, mid_features),
            self.nonlinearity)
        self.mid_features = mid_features

    def forward(self, x, t):
        t_emb = get_timestep_embedding(t, self.mid_features)
        t_emb = self.emb(t_emb.to(x.device))
        out = self.fc_in(x)
        out = self.fc_temp(out, t_emb=t_emb)
        out = self.fc_out(self.bn_out(out))
        return out


if __name__ == "__main__":
    model = Decoder(2, 128, 2)
    print(model, flush=True)
    out = model(torch.randn(16, 2), t=torch.randint(1000, size=(16, )))
    print(out.shape)
