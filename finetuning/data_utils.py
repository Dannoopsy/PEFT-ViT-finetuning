# +
import torch
import numpy as np
from PIL import Image
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }
def compute_metrics(p, metric):
    ans = metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    return ans


def transform(example_batch, processor, img_name, label_name, is_gray):
    # Take a list of PIL images and turn them to pixel values
    if is_gray:
        inputs = processor([Image.merge("RGB", (x, x, x)) for x in example_batch[img_name]], return_tensors='pt')
    else:
        inputs = processor([x for x in example_batch[img_name]], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch[label_name]
    return inputs
