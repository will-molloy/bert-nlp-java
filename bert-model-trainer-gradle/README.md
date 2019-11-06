# BERT model trainer gradle

```
pip install -r requirements.txt
```
* ***NOTE*** this might not work, may have to do something like:
```
python3.6 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.13.1-py3-none-any.whl
```

```
python3.6 run_classifier.py \
--task_name=cola \
--do_train=true \
--data_dir=./input-data \
--vocab_file=./pre-trained-model/vocab.txt \
--bert_config_file=./pre-trained-model/bert_config.json \
--init_checkpoint=./pre-trained-model/bert_model.ckpt \
--max_seq_length=64 \
--train_batch_size=64 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=./bert-output \
--servable_dir=./bert-output \
--do_serve=true
```
