from flask import Flask
app = Flask(__name__)
server = app.server

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig


tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

@app.route('/')
def hello_world():
    return 'Hello world!'

@app.route('/input/<input_data>')
def send_output(input_data):
    sequence = input_data
    inputs = tokenizer([sequence], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'])   
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return f'<h1>Summary = {summary}</h1>'
