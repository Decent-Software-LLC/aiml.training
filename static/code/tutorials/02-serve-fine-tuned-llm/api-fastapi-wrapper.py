from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Request(BaseModel):
    prompt: str
    max_tokens: int = 256

@app.post("/generate")
def generate(req: Request):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=req.max_tokens, do_sample=False)
    return {"text": tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)}
