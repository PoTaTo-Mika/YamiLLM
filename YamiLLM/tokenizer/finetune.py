from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# 加载已有的分词器
tokenizer = PreTrainedTokenizerFast.from_pretrained("checkpoints/tokenizer.json")
base_tokenizer = tokenizer.backend_tokenizer

# 准备新的训练数据
new_texts = ["text.txt"]

# 创建一个新的训练器，保留原有的特殊标记和词汇表大小
trainer = BpeTrainer(
    special_tokens=tokenizer.special_tokens_map.values(),
    vocab_size=base_tokenizer.get_vocab_size()
)

# 继续训练
base_tokenizer.train_from_iterator(new_texts, trainer=trainer)

# 更新 PreTrainedTokenizerFast
new_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=base_tokenizer,
    unk_token=tokenizer.unk_token,
    pad_token=tokenizer.pad_token,
    cls_token=tokenizer.cls_token,
    sep_token=tokenizer.sep_token,
    mask_token=tokenizer.mask_token,
)

# 保存更新后的分词器
new_tokenizer.save_pretrained("../../checkpoints")
