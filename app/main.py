from os import path
from fastapi import FastAPI, Body
from transformers import pipeline


def format_model_name(model_name: str):
    return model_name.replace("/", "-").lower()


def load_model(type: str, model: str):
    try:
        model_name = format_model_name(model)
        if (path.exists("/models/" + model_name)):
            return pipeline(
                type, model="/models/" + model_name, tokenizer="/models/" + model_name)
        else:
            model = pipeline(type, model)
            model.save_pretrained("/models/" + model_name)
            return model
    except Exception as e:
        raise (e)


app = FastAPI()


@app.post("/load")
async def load(body=Body(...)):
    load_model(body['task'], body['model'])
    return 'done'


@app.post("/run")
async def run(body=Body(...)):
    model = load_model(body['task'], body['model'])
    result = model(*body['args'])
    return result
