import json
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, jsonify, request, Response
from flask_restful import reqparse, abort, Api, Resource
from transformers import TFAutoModel, AutoTokenizer, TFGPT2LMHeadModel

#import model, sample, encoder

os.environ["FLASK_DEBUG"] = "1"
#os.environ["KMP_SETTINGS"] = "1"
#os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

def load_tokenizer_and_model(model_name_or_path):
  return AutoTokenizer.from_pretrained(model_name_or_path, from_pt=True), TFGPT2LMHeadModel.from_pretrained(model_name_or_path, from_pt=True)


def interact_model(model_name='model', model_size_context=768, seed=99, nsamples=5, batch_size=1,
                   length=8, temperature=1, top_k=10, top_p=0.85, models_dir=''):

    models_dir = os.path.expanduser(os.path.expandvars(models_dir))

    tok, model = load_tokenizer_and_model(models_dir)

    if length > model_size_context:
        raise ValueError("Can't get samples longer than window size: %s" % model_size_context)

    class Autocomplete(Resource):
        def get(self): return ''

        def post(self):
            body = request.get_json(force=True)
            if body['text'] == "": return

            context_tokens = tok.encode(body['text'], return_tensors="tf")
            generated = 0
            predictions = []
            out = model.generate(context_tokens, 
                                 num_return_sequences=nsamples,
                                 max_length=length, 
                                 do_sample=True, 
                                 top_k=top_k, 
                                 top_p=top_p, 
                                 temperature=temperature)

            for i in range(nsamples):
                generated += 1
                text = tok.decode(out[i])
                predictions.append(str(text))

            return Response(json.dumps({'result':predictions}), status=200, mimetype='application/json')

    app = Flask(__name__)
    api = Api(app)
    api.add_resource(Autocomplete, '/autocomplete')

    if __name__ == '__main__':
        app.run('0.0.0.0', port=3030, debug=False, use_reloader=False)

interact_model(models_dir="~/transformers_code_generator/models_fin")
