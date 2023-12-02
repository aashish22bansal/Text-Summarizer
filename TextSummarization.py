from flask import Flask
TextSummarization = Flask(__name__)

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig


tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

@TextSummarization.route('/input')
def send_output(sequence):
    inputs = tokenizer([sequence], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'])
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
