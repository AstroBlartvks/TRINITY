import torch
import torch.nn as nn

class TRINITY(nn.Module):
    """
    TRINITY Model

    Три ветки -> три lambda через softmax:
      data_feat  = LSTM(x_{t-W:t})                 (hidden,)
      garch_feat = Proj(garch_t)                   (hidden,)
      news_feat  = Tanh(Linear(768 → K))(emb_t)    (K,)
      news_h     = Linear(K → hidden)(news_feat)   (hidden,)
      gate_input = [BiLSTM(x); news_feat; garch_1d]  (64 + K + 1,)
      [λ_data, λ_garch, λ_news] = softmax(MLP(gate_input))
      combined = λ_data · data_feat + λ_garch · garch_feat + λ_news · news_h
      y_t      = Head(combined)
    """
    def __init__(self, hidden=128, dropout=0.2, k_news=16, bert_dim=768, device="cuda"):
        super().__init__()
        self.device = device
        self.k_news = k_news

        # Data branch
        self.data_lstm = nn.LSTM(1, hidden, 3, batch_first=True, dropout=dropout)
        self.data_bn   = nn.BatchNorm1d(hidden)

        # GARCH branch
        self.garch_proj = nn.Sequential(
            nn.Linear(1, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, hidden),
        )

        # News branch
        self.news_projector = nn.Sequential(
            nn.Linear(bert_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, k_news),   nn.Tanh(),
        )
        self.news_to_hidden = nn.Linear(k_news, hidden)

        # Gate: BiLSTM(x) + news_feat + garch_scalar -> 3 lambda
        gate_in_dim = 64 + k_news + 1          # 64=BiLSTM out, +1 for GARCH scalar
        self.gate_lstm = nn.LSTM(1, 32, 2, batch_first=True, bidirectional=True)
        self.gate_bn   = nn.BatchNorm1d(64)
        self.gate_head = nn.Sequential(
            nn.Linear(gate_in_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 3),
        )

        self.out_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, garch_signal, news_emb):
        """
        x:            (B, window)   нормализованная история σ^2
        garch_signal: (B,)          GARCH прогноз (нормализованный)
        news_emb:     (B, 768)      дневной RuBERT CLS эмбеддинг
        """
        x3 = x.unsqueeze(-1)                               # (B, W, 1)

        # Data branch
        out, _    = self.data_lstm(x3)
        data_feat = self.data_bn(out[:, -1, :])            # (B, hidden)

        # GARCH branch
        garch_feat = self.garch_proj(garch_signal.unsqueeze(-1))   # (B, hidden)

        # News branch
        news_feat = self.news_projector(news_emb)          # (B, k_news)
        news_h    = self.news_to_hidden(news_feat)         # (B, hidden)

        g_out, _  = self.gate_lstm(x3)
        g_feat    = self.gate_bn(g_out[:, -1, :])          # (B, 64)
        gate_in   = torch.cat(
            [g_feat, news_feat, garch_signal.unsqueeze(-1)],  # (B, 64+K+1)
            dim=-1
        )
        weights   = torch.softmax(self.gate_head(gate_in), dim=-1)  # (B, 3)

        lam_data  = weights[:, 0]
        lam_garch = weights[:, 1]
        lam_news  = weights[:, 2]

        combined = (lam_data.unsqueeze(-1)  * data_feat
                    + lam_garch.unsqueeze(-1) * garch_feat
                    + lam_news.unsqueeze(-1)  * news_h)

        output = self.out_head(combined).squeeze(-1)       # (B,)
        return output, lam_data, lam_garch, lam_news

    def get_news_components(self, news_emb):
        if news_emb.dim() == 1:
            news_emb = news_emb.unsqueeze(0)
        with torch.no_grad():
            return self.news_projector(news_emb.to(self.device)).cpu()


