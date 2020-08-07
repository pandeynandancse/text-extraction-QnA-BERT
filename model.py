import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        #load multilingual bert model
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)

	
	    #bert base uncased has 768 outputs  
        #here 2 outputs ==>> sentiment and selected text
        self.l0 = nn.Linear(768, 2)



    def forward(self, ids, mask, token_type_ids):
        #sentiment is not used here ===>>> use it also for better score


        #here output o1 has been used ==>>> if u  want then you can use output o2 as shown in sentiment-with-bert-onnx repository
        sequence_output, pooled_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        #sequence output looks like as follows dimension
        #(batch_size, num_tokens, 768)



        logits = self.l0(sequence_output) #(batch_size, num_tokens, 2)



        #split above output into two parts
        #(batch_size, num_tokens, 1) of start_logits, (batch_size, num_tokens, 1)  of end_logits
        start_logits, end_logits = logits.split(1, dim=-1)
        


        start_logits = start_logits.squeeze(-1) #shape is (batch_size, num_tokens) as last dimension i.e. -1 has been removed via squeeze
        end_logits = end_logits.squeeze(-1)     #shape is (batch_size, num_tokens) as last dimension i.e. -1 has been removed via squeeze



        return start_logits, end_logits