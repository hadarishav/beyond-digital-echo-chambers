# Beyond Digital “Echo Chambers”: The Role of Viewpoint Diversity in Political Discussion

Code, sample data, and other supplementary material for the paper "Beyond Digital “Echo Chambers”: The Role of Viewpoint Diversity in Political Discussion" accepted at WSDM'23

The paper can be found here: [Beyond Digital “Echo Chambers”: The Role of Viewpoint Diversity in Political Discussion](https://arxiv.org/abs/2212.09056)

Our online talk for WSDM 2023 will be posted soon.

If you use our work, please cite us:

```
@article{hada2022beyond,
  title={Beyond Digital" Echo Chambers": The Role of Viewpoint Diversity in Political Discussion},
  author={Hada, Rishav and Ebrahimi Fard, Amir and Shugars, Sarah and Bianchi, Federico and Rossini, Patricia and Hovy, Dirk and Tromble, Rebekah and Tintarev, Nava},
  journal={arXiv preprint arXiv:2212.09056},
  year={2022}
}
```

Each folder in this repository contains separate readme with instructions.

fragmentation_computation.py: code to compute fragmentation values. Takes conversation network constructed in “conversation_retrieval/3_conversation_reconstruction.py” as input.

```bash
python fragmentation_computaion.py
```

representation.py: code to compute representation values. Takes a list of conversations as input. Each conversation in the list is a list of labels per tweet. Ex. [[L1,L2,L2,L4],…..,[L4,L3,L1,L1,L2]].

```bash
python representation.py
```

dyadic_interaction.py: code to compute dyadic interaction values.

## Classifiers Training

To train the 4 classifiers (immigration relevance, immigration claim, daylight relevance, daylight claims) we 
make use of the standard HuggingFace [fine-tuning interface](https://huggingface.co/docs/transformers/training).
The model we fine-tuned is [BERTweet](https://huggingface.co/vinai/bertweet-base). Note that for the immigration claim prediction, we forced dataset balancing during training.
Nonetheless, all our models are trained using weighted cross entropy loss, that can be replicated with the following tuner:

```python

import torch
from torch import nn
from transformers import Trainer

class WeightedTrainer(Trainer):
    def __init__(self, internal_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.internal_weights = internal_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").to("cuda")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits').to("cuda")
        logits = logits.double()
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.internal_weights).to("cuda"))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

To create the weights for the labels, you can use sklearn

```python
from sklearn.preprocessing import LabelEncoder
import sklearn
import pandas as pd

train = pd.read_csv("train_data.csv")

le = LabelEncoder()

train["labels"] = le.fit_transform(train["labels"])

class_labels_for_w = list(range(0, len(le.classes_)))
weights = sklearn.utils.class_weight.compute_class_weight(class_weight="balanced",
                                                            classes=class_labels_for_w,
                                                            y=train["labels"].values.tolist())
```

These weights can be then passed to the Trainer

```python

trainer = WeightedTrainer(
    model=model,  
    args=training_args,  
    train_dataset=tokenized_train, 
    eval_dataset=tokenized_valid,  
    compute_metrics=compute_metrics,
    internal_weights=weights,
    
)
trainer.train()

```
