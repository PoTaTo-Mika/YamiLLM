from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import trainers, Tokenizer, models, normalizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset
import json
import os

Qwen_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-72B', trust_remote_code=True)
Yami_tokenizer = Tokenizer(models.BPE())

Yami_tokenizer.pre_tokenizer = Whitespace()
Yami_tokenizer.normalizer = normalizers.NFKC()

trainer = trainers.BpeTrainer(
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    vocab_size=150000
)

dataset = load_dataset('text',data_files = 'data.txt')

def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset['train']), batch_size):
        yield dataset['train'][i: i + batch_size]["text"]

Yami_tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset['train']))

Yami_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=Yami_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

output_dir = "../../checkpoints"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
fast_tokenizer.save_pretrained(output_dir)

vocab = fast_tokenizer.get_vocab()
vocab_file = os.path.join(output_dir, "vocab.json")
with open(vocab_file, 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"New tokenizer has been saved to {output_dir}")


