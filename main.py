from custom_tokens import custom_tokenizer, get_infrequent_tokens, mask_tokens
from indexing import encode
from model import Transformer
import pandas as pd
import torch, sys, random
import os

# мой вс тупит с кодировкой
sys.stdout.reconfigure(encoding='utf-8')

# Загружаем Excel
df = pd.read_excel("result.xlsx")

# Берём второй столбец
texts = df["text"]

# Соединяем в одну строку с разделителем <END>
corpus = f" <END> ".join(texts.astype(str))

# Список спецтокенов
spec_tokens = ["<END>", "<UNK>"]

# Токенизация
tokens = custom_tokenizer(corpus, spec_tokens)

infreq_tokens = get_infrequent_tokens(tokens, min_count=2)
tokens = mask_tokens(tokens, infreq_tokens)


print(tokens[:10])  # первые 10 токенов
print("Всего токенов:", len(tokens))
print("Всего уникальных токенов:", len(set(tokens)))

# --- 3. Словарь ---
vocab_list = list(set(tokens))
word2idx = {w: i for i, w in enumerate(vocab_list)}
idx2word = {i: w for w, i in word2idx.items()}

# --- 4. Кодировка ---
enc = encode(tokens, vocab_list)  # (seq_len_total,)
print(enc)

# --- 5. Разделение на батчи ---
seq_len = 16
chunks = [enc[i:i+seq_len] for i in range(0, len(enc), seq_len)]
chunks = [c for c in chunks if len(c) == seq_len]
src = torch.stack(chunks)  # (batch, seq_len)

SOS_idx = word2idx["<END>"]  # токен старта
tgt = torch.cat([torch.full((src.size(0), 1), SOS_idx), src[:, :-1]], dim=1)

# --- 6. Маски ---
src_mask = torch.ones(src.size(0), 1, src.size(1), dtype=torch.bool)
tgt_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool)).unsqueeze(0)

# --- 7. Инициализация модели ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src_vocab = len(vocab_list)
tgt_vocab = len(vocab_list)

model = Transformer(src_vocab, tgt_vocab).to(device).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

src, tgt = src.to(device), tgt.to(device)
src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

# --- Путь к файлу сохранения ---
checkpoint_path = "transformer_checkpoint.pth"

# --- Загрузка, если есть сохранение ---
start_epoch = 1
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Загружено сохранение: эпоха {start_epoch - 1}")

# --- 8. Обучение ---
epochs = 10
for epoch in range(start_epoch, epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(src, tgt, src_mask, tgt_mask)  # (batch, seq_len, vocab)
    
    # 🔹 БЫЛО: сравнивалось с src
    # loss = criterion(out.view(-1, tgt_vocab), src.view(-1))
    # 🔹 СТАЛО: теперь сравнивается с tgt (правильно)
    loss = criterion(out.view(-1, tgt_vocab), tgt.view(-1))

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")

    # --- Сохранение ---
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item()
    }, checkpoint_path)
    print(f"Сохранено в {checkpoint_path}")


# --- 10. Тест генерации ---
def generate_from_enc(model, enc_tensor, idx2word, SOS_idx, max_len=10):
    model.eval()
    src = enc_tensor.unsqueeze(0).to(device)  # (1, seq_len)
    src_mask = torch.ones(1, 1, src.size(1), dtype=torch.bool, device=device)
    
    generated = torch.full((1, 1), SOS_idx, dtype=torch.long, device=device)
    for _ in range(max_len):
        tgt_mask = torch.tril(torch.ones(generated.size(1), generated.size(1), dtype=torch.bool, device=device)).unsqueeze(0)
        probs = model(src, generated, src_mask, tgt_mask)

        # 🔹 БЫЛО: argmax (зацикливалось)
        # next_token = torch.argmax(probs[:, -1, :], dim=-1, keepdim=True)

        # 🔹 СТАЛО: добавила temperature + сэмплирование
        probs = torch.softmax(probs[:, -1, :] / 0.8, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)
    
    tokens_out = [idx2word[int(i)] for i in generated[0, 1:]]  # без SOS
    return tokens_out


test_seq = chunks[random.randint(0, len(chunks)-1)]
gen_tokens = generate_from_enc(model, test_seq, idx2word, SOS_idx)
print("Сгенерированный текст:", " ".join(gen_tokens))


# --- 11. Функция общения с моделью ---
def chat_with_model(model, word2idx, idx2word, SOS_idx, seq_len=16, max_gen_len=10):
    print("Начинаем чат с моделью! (введите 'exit' для выхода)")
    model.eval()
    
    history_tokens = []  # для накопления контекста
    
    while True:
        user_input = input("Вы: ")
        if user_input.lower() == "exit":
            break
        
        # Токенизация и замена редких слов
        user_tokens = custom_tokenizer(user_input, spec_tokens)
        user_tokens = mask_tokens(user_tokens, infreq_tokens)
        
        history_tokens.extend(user_tokens)
        # Оставляем только последние seq_len токенов
        context_tokens = history_tokens[-seq_len:]
        
        # Кодировка
        context_enc = encode(context_tokens, vocab_list)
        context_enc = torch.tensor(context_enc, dtype=torch.long).to(device) if not isinstance(context_enc, torch.Tensor) else context_enc.detach().clone().long().to(device)

        
        # Генерация ответа
        gen_tokens = generate_from_enc(model, context_enc, idx2word, SOS_idx, max_len=max_gen_len)
        
        print("Модель:", " ".join(gen_tokens))
        
        # Добавляем ответ модели в контекст для поддержания диалога
        history_tokens.extend(gen_tokens)


# --- 12. Запуск чата ---
chat_with_model(model, word2idx, idx2word, SOS_idx)