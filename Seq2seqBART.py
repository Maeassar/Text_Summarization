import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from torchtext.data.metrics import bleu_score

class Seq2SeqBART(nn.Module):
    def __init__(self, device, model_name='facebook/bart-base'):
        super(Seq2SeqBART, self).__init__()
        self.device = device
        self.bart_model = BartForConditionalGeneration.from_pretrained(model_name).to(device)


    def forward(self, input_text, target_text):
        encoder_input_ids = input_text
        decoder_input_ids = target_text
        encoder_input_ids = encoder_input_ids.to(self.device)
        #print(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)
        outputs = self.bart_model(input_ids=encoder_input_ids, labels=decoder_input_ids)
        #print("outputs", outputs)
        return outputs, outputs.logits

    def generate_summary(self, input_text, trg_text, max_length=128):
        encoder_input_ids = input_text
        encoder_input_ids = encoder_input_ids.to(self.device)
        #print("encoder_input_ids",encoder_input_ids )
        summary_ids = self.bart_model.generate(encoder_input_ids, max_length=max_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        #print("summary_ids", summary_ids)
        summary = torch.tensor(summary_ids[0])
        #print("summary.size()", summary.size())
        #print("summary", summary)
        #print("trg_text.size()", trg_text.size())
        #print("trg_text", trg_text)

        return summary

