# +
from datasets import load_dataset
from transformers import ViTImageProcessor
from transformers import ViTForImageClassification
import torch
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
from transformers import Trainer
from peft import LoraConfig, get_peft_model


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }
def compute_metrics(p, metric):
    ans = metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    return ans
def transform(example_batch, processor):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in example_batch['img']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    return inputs

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def train(cfg):
    model_name = cfg.model.model_name
    dset_name = cfg.data.dset_name
    metric_name = cfg.training.metric_name
    output_dir = cfg.data.output_dir
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs
    learning_rate = cfg.training.learning_rate
    
    metric = load_metric(metric_name)
    ds = load_dataset(dset_name)
    prepared_ds = ds.with_transform(lambda x :transform(x, processor))

    labels = ds['train'].features['label'].names
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )
    
    if cfg.model.peft_type == 'none':
        pass
    elif cfg.model.peft_type == 'lora':
        lora_cfg = dict(cfg.peft)
        lora_cfg['modules_to_save'] = list(lora_cfg['modules_to_save'])
        lora_cfg['target_modules'] = list(lora_cfg['target_modules'])
        config = LoraConfig(**lora_cfg)
        model = get_peft_model(model, config)
        print_trainable_parameters(model)
    

    
    training_args = TrainingArguments(
      output_dir=output_dir,
      per_device_train_batch_size=batch_size,
      evaluation_strategy="steps",
      num_train_epochs=epochs,
      fp16=True,
      save_steps=0.2,
      eval_steps=0.2,
      logging_steps=10,
      learning_rate=learning_rate,
      save_total_limit=2,
      remove_unused_columns=False,
      push_to_hub=False,
      metric_for_best_model="accuracy",
      load_best_model_at_end=True,
      report_to='wandb',
      label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=lambda x :compute_metrics(x, metric),
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["test"],
        tokenizer=processor,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(prepared_ds['test'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
