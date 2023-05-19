# Multilingual classification inference code

fine-tuning 순서에 따른 성능 비교

* train_goemotions_xlmroberta.py : Pretrained XLMRoBERTa에 GoEmotions fine tuning 하는 code
* finetune_kote_xlmroberta.py : 위 goEmotions fine-tuning 후 KOTE data를 fine-tuning 하는 code
* kote_goemotion_xlmroberta.py : GoEmotions 와 KOTE 를 concat 후 fine-tuning 하는 code

Multilingual vs Monolingual pretrained model 성능 비교

* train_kote_koroberta.py - KOTE datasets을 Kor RoBERTa 에 fine-tuning
* train_kote_xlmroberta.py - KOTE datasets을 XLMRoBERTa 에 fine-tuning

