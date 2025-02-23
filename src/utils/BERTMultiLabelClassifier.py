# ----------------------------------- *** ---------------------------------------
#                                   CLASE 1: 

# "BERTMultiLabelClassifier" como modelo personalizado basado en BertModel Hugging Face Transformers
import torch
import torch.nn as nn
from transformers import BertModel

class BERTMultiLabelClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(BERTMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
