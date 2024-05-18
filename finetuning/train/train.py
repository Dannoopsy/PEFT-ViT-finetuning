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
import wandb
import omegaconf
from omegaconf import DictConfig, OmegaConf
from data_utils import collate_fn, compute_metrics, transform


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
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project="VIT PEFT", config=config_dict
    )

    
    model_name = cfg.model.model_name
    dset_name = cfg.data.dset_name
    metric_name = cfg.training.metric_name
    output_dir = cfg.data.output_dir
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs
    learning_rate = cfg.training.learning_rate
    scheduler = cfg.training.scheduler
    
    
    metric = load_metric(metric_name)
    ds = load_dataset(dset_name)
    if cfg.data.test_size > 0:
        ds = ds['train'].train_test_split(test_size=cfg.data.test_size)
    
    prepared_ds = ds.with_transform(lambda x :transform(x, processor, cfg.data.img_name, cfg.data.label_name, cfg.data.is_gray))

    labels = ds['train'].features[cfg.data.label_name].names
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    
    
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )
    
    
    if cfg.peft.peft_type == 'none':
        pass
    
    elif cfg.peft.peft_type == 'freeze':
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        if cfg.peft.n_layers_no_freeze > 0:
            for p in model.vit.layernorm.parameters():
                p.requires_grad = True
        for i in range(len(model.vit.encoder.layer) - cfg.peft.n_layers_no_freeze, len(model.vit.encoder.layer)):
            for p in model.vit.encoder.layer[i].parameters():
                p.requires_grad = True
        
    elif cfg.peft.peft_type == 'lora':
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
      lr_scheduler_type=scheduler
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=lambda x :compute_metrics(x, metric),
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds[cfg.data.test_name],
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
