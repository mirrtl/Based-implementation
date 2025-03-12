import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
from einops import rearrange
# Датасет
class MQARDataset(Dataset):
    def __init__(self, vocab_size=100, seq_len=32, min_kv_pairs=2, max_kv_pairs=8, num_samples=1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.min_kv_pairs = min_kv_pairs
        self.max_kv_pairs = max_kv_pairs
        self.num_samples = num_samples
        self.data = self._generate_data()

    def _generate_data(self):
        return [self._generate_sequence() for _ in range(self.num_samples)]

    def _generate_sequence(self):
        num_kv_pairs = random.randint(self.min_kv_pairs, self.max_kv_pairs)
        seq = []
        kv_store = {}
        targets = []

        # Генерация key-value пар
        for _ in range(num_kv_pairs):
            key = random.randint(0, self.vocab_size-1)
            value = random.randint(0, self.vocab_size-1)
            seq.extend([key, value])
            kv_store[key] = value

        # Генерация запросов с контролем длины
        max_queries = min(len(kv_store), self.seq_len - len(seq))
        queries = random.sample(list(kv_store.keys()), max_queries)
        for query in queries:
            seq.append(query)
            targets.append(kv_store[query])

        # Обрезка и дополнение последовательности
        seq = seq[:self.seq_len]
        seq += [random.randint(0, self.vocab_size-1) for _ in range(self.seq_len - len(seq))]
        targets = targets[:self.seq_len] + [-100]*(self.seq_len - len(targets))

        return torch.tensor(seq), torch.tensor(targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Разложение
class TaylorExp(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim  # feature_dim=4
        self.r2 = math.sqrt(2)
        self.rd = math.sqrt(input_dim)
        self.rrd = math.sqrt(math.sqrt(input_dim))

    def forward(self, x: torch.Tensor):
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        term1 = torch.ones_like(x[..., :1])  # +1
        term2 = x / self.rrd                 # +input_dim
        term3 = x2 / self.rd                 # +input_dim²
        return torch.cat([term1, term2, term3], dim=-1) 

class LinearAttention(nn.Module):
    def __init__(self, d_model: int, feature_dim: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.feature_dim = feature_dim 
        self.expanded_dim = 1 + feature_dim + feature_dim**2  
        self.head_dim = d_model // num_heads  

        self.feature_map = TaylorExp(input_dim=self.feature_dim)

        self.proj_q = nn.Linear(d_model, self.feature_dim * num_heads)
        self.proj_k = nn.Linear(d_model, self.feature_dim * num_heads)
        self.proj_v = nn.Linear(d_model, self.head_dim * num_heads)


        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, kv_state: torch.Tensor, k_state: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        # Проекции
        q = rearrange(self.proj_q(x), 'b l (h d) -> b h l d', h=self.num_heads, d=self.feature_dim)
        k = rearrange(self.proj_k(x), 'b l (h d) -> b h l d', h=self.num_heads, d=self.feature_dim)
        v = rearrange(self.proj_v(x), 'b l (h d) -> b h l d', h=self.num_heads, d=self.head_dim)

        # Применение feature map (расширяет до 21)
        q = self.feature_map(q)
        k = self.feature_map(k) 

        outputs = []
        for t in range(seq_len):
            # Обновление состояний
            kv_update = torch.einsum('b h d, b h c -> b h d c', k[:, :, t], v[:, :, t])
            kv_state = kv_state + kv_update
            k_state = k_state + k[:, :, t]

            # Вычисление внимания
            num = torch.einsum('b h d, b h d c -> b h c', q[:, :, t], kv_state)
            denom = torch.einsum('b h d, b h d -> b h', q[:, :, t], k_state) + 1e-6  
            out = num / denom.unsqueeze(-1)
            outputs.append(out)

        # [seq_len, batch, heads, head_dim] -> [batch, seq_len, heads*head_dim]
        x = rearrange(outputs, 't b h d -> b t (h d)')

        return self.dropout(self.out_proj(x)), kv_state.detach(), k_state.detach()

class SlidingWindow(nn.Module):
    def __init__(self, window_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = head_dim


        self.register_buffer('k_buf', torch.zeros(1, num_heads, window_size, head_dim))
        self.register_buffer('v_buf', torch.zeros(1, num_heads, window_size, head_dim))

    def forward(self, new_k: torch.Tensor, new_v: torch.Tensor):
        # new_k и new_v: [batch, heads, seq_len, head_dim]
        batch_size = new_k.size(0)

        # Берём последний элемент последовательности
        new_k = new_k[:, :, -1:, :]  # [batch, heads, 1, head_dim]
        new_v = new_v[:, :, -1:, :]  # [batch, heads, 1, head_dim]

        # Расширяем буферы до текущего batch_size
        k_buf = self.k_buf.expand(batch_size, -1, -1, -1)
        v_buf = self.v_buf.expand(batch_size, -1, -1, -1)

        updated_k = torch.cat([k_buf[:, :, 1:], new_k], dim=2)
        updated_v = torch.cat([v_buf[:, :, 1:], new_v], dim=2)

        # Обновляем буферы с новыми значениями
        self.k_buf.data.copy_(updated_k.detach().mean(dim=0, keepdim=True))
        self.v_buf.data.copy_(updated_v.detach().mean(dim=0, keepdim=True))

        return updated_k, updated_v
class BasedBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, window_size: int, feature_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.head_dim = d_model // num_heads

        self.linear_attn = LinearAttention(d_model, feature_dim, num_heads)
        self.sliding_window = SlidingWindow(window_size, num_heads, self.head_dim)
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def window_attention(self, x):
        batch_size, seq_len, _ = x.shape


        q = rearrange(self.proj_q(x), 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(self.proj_k(x), 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(self.proj_v(x), 'b l (h d) -> b h l d', h=self.num_heads)

        window_k, window_v = self.sliding_window(k, v)

        scores = torch.einsum('b h q d, b h w d -> b h q w', q, window_k)
        attn = torch.softmax(scores / math.sqrt(self.head_dim), dim=-1)
        return rearrange(torch.einsum('b h q w, b h w d -> b h q d', attn, window_v),
                        'b h l d -> b l (h d)')

    def forward(self, x: torch.Tensor, kv_state: torch.Tensor, k_state: torch.Tensor):
        # Линейное внимание
        lin_out, kv_state, k_state = self.linear_attn(x, kv_state, k_state)
        x = self.norm(x + lin_out)

        # Оконное внимание
        win_out = self.window_attention(x)
        x = self.norm(x + self.dropout(self.out_proj(win_out)))
        return x, kv_state, k_state

class BasedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config['vocab_size'], config['d_model'])
        self.layers = nn.ModuleList([
            BasedBlock(
                d_model=config['d_model'],
                num_heads=config['num_heads'],
                window_size=config['window_size'],
                feature_dim=config['feature_dim'] 
            ) for _ in range(config['num_layers'])
        ])
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'])

    def init_states(self, batch_size):
        device = next(self.parameters()).device
        expanded_dim = self.layers[0].linear_attn.expanded_dim
        head_dim = self.layers[0].linear_attn.head_dim

        return [(
            torch.zeros(batch_size, self.config['num_heads'], expanded_dim, head_dim).to(device),
            torch.zeros(batch_size, self.config['num_heads'], expanded_dim).to(device)
        ) for _ in self.layers]

    def forward(self, input_ids, states=None):
        x = self.embed(input_ids)
        states = states or self.init_states(input_ids.size(0))
        new_states = []

        for i, layer in enumerate(self.layers):
            x, *layer_states = layer(x, *states[i])
            new_states.append(layer_states)

        return self.lm_head(x), new_states


def train():
    config = {
        'vocab_size': 300,
        'd_model': 512,
        'num_layers': 6,
        'num_heads': 8,
        'feature_dim': 8,  
        'window_size': 128,
        'batch_size': 5,
        'seq_len': 25,
        'lr': 8e-4,
        'epochs': 2
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BasedLM(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    dataset = MQARDataset(
        vocab_size=config['vocab_size'],
        seq_len=config['seq_len']
    )




    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(inputs)
            
            # Расчет потерь
            loss = criterion(logits.view(-1, config['vocab_size']), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            # Расчет accuracy
            predictions = logits.argmax(dim=-1)
            mask = targets != -100
            correct = (predictions[mask] == targets[mask]).sum().item()
            total_correct += correct
            total_samples += mask.sum().item()
            
            total_loss += loss.item()
            
            if batch_idx % 3 == 0:
                accuracy = correct / mask.sum().item() if mask.sum().item() > 0 else 0
                print(f'Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f} | Acc: {accuracy:.2%}')
        
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
        print(f'Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_accuracy:.2%}')

if __name__ == '__main__':
    train()