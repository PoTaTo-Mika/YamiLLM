from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# �������еķִ���
tokenizer = PreTrainedTokenizerFast.from_pretrained("checkpoints/tokenizer.json")
base_tokenizer = tokenizer.backend_tokenizer

# ׼���µ�ѵ������
new_texts = ["text.txt"]

# ����һ���µ�ѵ����������ԭ�е������Ǻʹʻ���С
trainer = BpeTrainer(
    special_tokens=tokenizer.special_tokens_map.values(),
    vocab_size=base_tokenizer.get_vocab_size()
)

# ����ѵ��
base_tokenizer.train_from_iterator(new_texts, trainer=trainer)

# ���� PreTrainedTokenizerFast
new_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=base_tokenizer,
    unk_token=tokenizer.unk_token,
    pad_token=tokenizer.pad_token,
    cls_token=tokenizer.cls_token,
    sep_token=tokenizer.sep_token,
    mask_token=tokenizer.mask_token,
)

# ������º�ķִ���
new_tokenizer.save_pretrained("../../checkpoints")
