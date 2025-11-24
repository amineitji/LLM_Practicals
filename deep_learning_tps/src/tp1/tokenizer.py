import re
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

print("="*60)
print("TP1 - PARTIE 1: TOKENIZERS")
print("="*60)

# 1. Charger texte
with open('data/the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

print(f"\n1. Nombre de caractères: {len(raw_text)}")
print(f"Premiers 100 caractères:\n{raw_text[:100]}")

# 2. Tokenizer simple (whitespace)
print("\n2. Tokenizer simple (whitespace):")
simple_tokens = re.split(r'(\s)', raw_text[:299])
simple_tokens = [t for t in simple_tokens if t]
print(f"Nombre de tokens: {len(simple_tokens)}")
print(f"Tokens: {simple_tokens}")

# 3. Tokenizer avancé
print("\n3. Tokenizer avancé (whitespace + ponctuation):")
advanced_pattern = r'([,.:;?_!"()\']|--|\s)'
advanced_tokens = re.split(advanced_pattern, raw_text[:299])
advanced_tokens = [t.strip() for t in advanced_tokens if t.strip()]
print(f"Nombre de tokens: {len(advanced_tokens)}")
print(f"Tokens: {advanced_tokens}")

# 4. Stats tokens
print("\n4. Statistiques sur le texte complet:")
all_tokens = re.split(advanced_pattern, raw_text)
all_tokens = [t.strip() for t in all_tokens if t.strip()]
unique_tokens = sorted(set(all_tokens))

print(f"Nombre total de tokens: {len(all_tokens)}")
print(f"Nombre de tokens uniques: {len(unique_tokens)}")

# 5. Vocabulaire
print("\n5. Création du vocabulaire:")
vocab = {token: idx for idx, token in enumerate(unique_tokens)}
print(f"Taille vocabulaire: {len(vocab)}")
print("\nPremiers 20 tokens du vocabulaire:")
for i, (token, idx) in enumerate(list(vocab.items())[:20]):
    print(f"  ('{token}', {idx})")

# 6. SimpleTokenizerV1
print("\n6. SimpleTokenizerV1:")

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}
    
    def encode(self, text):
        pattern = r'([,.:;?_!"()\']|--|\s)'
        tokens = re.split(pattern, text)
        tokens = [t.strip() for t in tokens if t.strip()]
        ids = [self.str_to_int[t] for t in tokens]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

tokenizer_v1 = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know,"""
ids = tokenizer_v1.encode(text)
print(f"Text: {text}")
print(f"Encoded: {ids}")
print(f"Decoded: {tokenizer_v1.decode(ids)}")

# 7. Test mots inconnus
print("\n7. Test avec mots inconnus:")
test_text = "Hello, do you like tea. Is this-- a test?"
try:
    tokenizer_v1.encode(test_text)
except KeyError as e:
    print(f"❌ Erreur: Token '{e.args[0]}' inconnu")

# 8. SimpleTokenizerV2 avec tokens spéciaux
print("\n8. SimpleTokenizerV2 (avec <|unk|> et <|endoftext|>):")

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.str_to_int['<|endoftext|>'] = len(vocab)
        self.str_to_int['<|unk|>'] = len(vocab) + 1
        self.int_to_str = {v: k for k, v in self.str_to_int.items()}
    
    def encode(self, text):
        pattern = r'([,.:;?_!"()\']|--|\s|<\|endoftext\|>)'
        tokens = re.split(pattern, text)
        tokens = [t.strip() for t in tokens if t.strip()]
        ids = [self.str_to_int.get(t, self.str_to_int['<|unk|>']) for t in tokens]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

tokenizer_v2 = SimpleTokenizerV2(vocab)
print(f"Vocab size: {len(tokenizer_v2.str_to_int)}")
print(f"<|endoftext|> token id: {tokenizer_v2.str_to_int['<|endoftext|>']}")
print(f"<|unk|> token id: {tokenizer_v2.str_to_int['<|unk|>']}")

# 9. Test tokenizer V2
print("\n9. Test SimpleTokenizerV2:")
test_v2 = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
encoded_v2 = tokenizer_v2.encode(test_v2)
decoded_v2 = tokenizer_v2.decode(encoded_v2)
print(f"Text: {test_v2}")
print(f"Encoded: {encoded_v2}")
print(f"Decoded: {decoded_v2}")

# 10. GPT2 BytePair Tokenizer
print("\n10. GPT2 BytePair Tokenizer:")
gpt2_tokenizer = tiktoken.get_encoding("gpt2")
print(f"Vocab size: {gpt2_tokenizer.n_vocab}")

test_gpt2 = "Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace."
encoded_gpt2 = gpt2_tokenizer.encode(test_gpt2, allowed_special={"<|endoftext|>"})
decoded_gpt2 = gpt2_tokenizer.decode(encoded_gpt2)
print(f"\nText: {test_gpt2}")
print(f"Encoded: {encoded_gpt2}")
print(f"Decoded: {decoded_gpt2}")

# Visualisation distribution tokens
print("\n11. Visualisation distribution des tokens:")
token_counts = Counter(all_tokens)
most_common = token_counts.most_common(20)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 20 tokens
tokens, counts = zip(*most_common)
axes[0].barh(range(len(tokens)), counts, color='steelblue')
axes[0].set_yticks(range(len(tokens)))
axes[0].set_yticklabels([f"'{t}'" if len(t) < 10 else f"'{t[:7]}...'" for t in tokens])
axes[0].set_xlabel('Fréquence', fontsize=12)
axes[0].set_title('Top 20 Tokens les Plus Fréquents', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()

# Distribution longueurs
token_lengths = [len(t) for t in all_tokens]
axes[1].hist(token_lengths, bins=30, color='coral', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Longueur du token', fontsize=12)
axes[1].set_ylabel('Fréquence', fontsize=12)
axes[1].set_title('Distribution des Longueurs de Tokens', fontsize=14, fontweight='bold')
axes[1].axvline(sum(token_lengths)/len(token_lengths), color='red', 
                linestyle='--', linewidth=2, label=f'Moyenne: {sum(token_lengths)/len(token_lengths):.2f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/tp1_tokenizer_stats.png', dpi=150, bbox_inches='tight')
print("✓ Visualisation sauvegardée: outputs/tp1_tokenizer_stats.png")
plt.close()
