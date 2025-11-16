import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def trans_to_cuda(x):
    return x.cuda() if torch.cuda.is_available() else x

def trans_to_cpu(x):
    return x.cpu() if torch.cuda.is_available() else x


class AttenMixer(nn.Module):
    def __init__(self, opt, n_node):
        super().__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.levels = 3  # L = 3 (last 1, 2, 3 items)
        self.heads = opt.heads
        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.heads, batch_first=True)

        # Query generators (Deep Sets: sum + MLP)
        self.query_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            ) for _ in range(self.levels)
        ])

        # Lp pooling
        self.p = 4

        # Final prediction
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, mask):
        # inputs: [B, L], mask: [B, L]
        hidden = self.embedding(inputs)  # [B, L, H]

        # Generate multi-level queries
        queries = []
        for l in range(1, self.levels + 1):
            # Take last l items
            last_l = hidden[:, -l:, :]  # [B, l, H]
            summed = last_l.sum(dim=1)  # Deep Sets: сумма
            query = self.query_mlps[l - 1](summed)  # [B, H]
            queries.append(query.unsqueeze(1))  # [B, 1, H]

        queries = torch.cat(queries, dim=1)  # [B, L, H]

        # Multi-head attention
        attn_out, _ = self.attn(queries, hidden, hidden)  # [B, L, H]

        # Lp pooling по уровням
        attn_pooled = (attn_out ** self.p).mean(dim=1) ** (1 / self.p)  # [B, H]

        # Последний элемент
        last_idx = mask.sum(dim=1) - 1
        last_hidden = hidden[torch.arange(hidden.size(0)), last_idx]

        # Финальное представление
        session_emb = self.linear_transform(torch.cat([attn_pooled, last_hidden], dim=-1))

        # Предсказание
        scores = torch.matmul(session_emb, self.embedding.weight[1:].t())
        return scores


def forward(model, i, data):
    inputs, mask, targets = data.get_slice(i)
    inputs = trans_to_cuda(torch.tensor(inputs).long())
    mask = trans_to_cuda(torch.tensor(mask).long())
    targets = trans_to_cuda(torch.tensor(targets).long())

    scores = model(inputs, mask)
    return targets, scores


def train_test(model, train_data, test_data, epoch):
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size, epoch)

    for i in slices:
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()

    model.scheduler.step()
    avg_loss = total_loss / len(slices)
    print(f'Epoch {epoch} | Loss: {avg_loss:.4f}')

    # Оценка
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size, epoch)

    with torch.no_grad():
        for i in slices:
            targets, scores = forward(model, i, test_data)
            sub_scores = scores.topk(20)[1]
            sub_scores = trans_to_cpu(sub_scores).numpy()
            for score, target in zip(sub_scores, targets):
                hit.append(np.isin(target - 1, score))
                if np.isin(target - 1, score):
                    rank = np.where(score == target - 1)[0][0] + 1
                    mrr.append(1.0 / rank)
                else:
                    mrr.append(0.0)

    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr