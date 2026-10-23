"""Qwen3.5-4B EVALUATION — REAL RUN for tutorial verification.
Held-out Dolly (never trained on). Paired A/B base vs adapter on the SAME prompts.
Heuristic (ref-anchored, primary) + LLM-as-judge (secondary) + perplexity sanity.
4-bit, sequential base/ft, on a 12 GB GPU. Phase checkpoints -> resume-safe.
"""
import os, json, time, math, random
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ.setdefault("CPATH", "/tmp/usr/include:/tmp/usr/include/python3.11")

# ── paths: point these at your own model / adapter / run dirs ──────────────
BASE    = "/path/to/models/qwen35-4b"
ADAPTER = "/path/to/runs/q35-4b-lora/adapter"
RUN     = "/path/to/runs/q35-4b-lora"

# ── judge endpoint: URL from the env, key from the env (never hardcoded) ──
JUDGE_URL   = os.environ.get("LITELLM_JUDGE_URL", "https://YOUR-LITELLM-ENDPOINT/v1")
JUDGE_KEY   = os.environ["LITELLM_JUDGE_API_KEY"]  # never hardcode; set it in the environment
JUDGE_MODEL = os.environ.get("LITELLM_JUDGE_MODEL", "decent-chat-fast")

N_EVAL = 150; MAX_NEW = 256
random.seed(42)
J = lambda t: print(json.dumps(t, default=str), flush=True)

def bnb4():
    from bitsandbytes import nn as bnn  # noqa
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM
from peft import PeftModel
tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def load_model(with_adapter):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=__import__("torch").bfloat16, bnb_4bit_use_double_quant=True)
    M = (PeftModel.from_pretrained(Qwen3_5ForCausalLM.from_pretrained(BASE, torch_dtype=__import__("torch").bfloat16,
          quantization_config=bnb, device_map="cuda:0"), ADAPTER) if with_adapter
         else Qwen3_5ForCausalLM.from_pretrained(BASE, torch_dtype=__import__("torch").bfloat16,
          quantization_config=bnb, device_map="cuda:0"))
    return M

import torch
def perplexity(M, texts, max_len=256):
    tl = tt = 0
    for t in texts:
        b = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(M.device)
        with torch.inference_mode(): o = M(**b, labels=b["input_ids"])
        n = (b["input_ids"] != tok.pad_token_id).sum().item()
        tl += o.loss.item() * n; tt += n
    return math.exp(tl/tt)

def gen(M, ex, max_new=MAX_NEW):
    msgs = [{"role":"user","content":ex["prompt"]}]
    inp = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(M.device)
    with torch.inference_mode():
        o = M.generate(**inp, do_sample=False, max_new_tokens=max_new,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    new = o[0, inp["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip()

def build_eval():
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train").select(range(14000, 15000))
    rows = []
    for ex in ds:
        instr = (ex["instruction"] + (("\n\n" + ex["context"]) if ex.get("context") else "")).strip()
        if not ex.get("response") or len(ex["response"]) < 8: continue
        rows.append({"prompt": instr, "reference": ex["response"].strip()})
        if len(rows) >= N_EVAL: break
    return rows

def heuristic(base_out, ft_out, ref):
    refs = set(w.lower() for w in ref.split())
    def score(o):
        toks = [w.lower() for w in o.replace("\n"," ").split()]
        if not refs: return 0
        return sum(t in refs for t in toks) / len(refs)
    return score(ft_out), score(base_out)

def judge_call(instruction, ref, a, b):
    import urllib.request
    P = ("You are an impartial judge comparing two assistant responses to the same instruction.\n"
         "Judge ONLY: (1) follows the instruction, (2) correctness, (3) helpfulness without padding.\n"
         "Do NOT favor longer answers or a particular style.\n\nInstruction:\n" + instruction[:800] +
         "\n\nReference answer (ground truth, if any):\n" + (ref[:500] or "(none)") +
         "\n\nResponse A:\n" + a[:1200] + "\n\nResponse B:\n" + b[:1200] +
         "\n\nReply with EXACTLY one token: A, B, or TIE.")
    req = urllib.request.Request(JUDGE_URL + "/chat/completions",
        data=json.dumps({"model":JUDGE_MODEL,"messages":[{"role":"user","content":P}],"temperature":0.0,"max_tokens":8}).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+JUDGE_KEY})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                t = json.loads(r.read())["choices"][0]["message"]["content"].strip().upper()
            return "A" if t.startswith("A") else "B" if t.startswith("B") else "TIE"
        except Exception as e:
            if attempt == 3: return None
            time.sleep(2 ** (attempt + 1))

def wilson(k, n, z=1.96):
    if n == 0: return (0.0,0.0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0.0,c-h), min(1.0,c+h))

def judge_preflight():
    try:
        return judge_call("Say the word tie.", "", "Response A", "Response B") is not None
    except Exception:
        return False

def main():
    rows = json.load(open(RUN+"/eval_set.json")) if os.path.exists(RUN+"/eval_set.json") else build_eval()
    json.dump(rows, open(RUN+"/eval_set.json","w"))
    J({"phase":"eval_set","n":len(rows)})
    # Load or generate both arms (cached in gen_*.jsonl)
    for name, wa in [("base",False),("fine_tuned",True)]:
        if os.path.exists(RUN+f"/gen_{name}.jsonl"):
            rows = json.load(open(RUN+f"/gen_{name}.jsonl"))
            J({"phase":f"gen_{name}","reused":True}); continue
        M = load_model(wa)
        t0=time.time()
        ppl = perplexity(M, [tok.apply_chat_template([{"role":"user","content":r["prompt"]},{"role":"assistant","content":r["reference"]}],tokenize=False) for r in rows])
        outs=[]
        for i,r in enumerate(rows): outs.append(gen(M, r))
        for r,o in zip(rows,outs): r[name]=o
        json.dump(rows, open(RUN+f"/gen_{name}.jsonl","w"))
        J({"phase":f"gen_{name}","ppl":round(ppl,3),"wall_s":round(time.time()-t0),"n":len(rows)})
        del M; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    use_judge = judge_preflight()
    J({"phase":"judge_preflight","enabled":use_judge})
    W=L=T=0; hwin=hloss=htie=0
    detail=[]
    judge_args = []
    for i,r in enumerate(rows):
        ft_first = random.random() < 0.5
        a,b = (r["fine_tuned"],r["base"]) if ft_first else (r["base"],r["fine_tuned"])
        sf,sb = heuristic(r["base"], r["fine_tuned"], r["reference"])
        if sf>sb: hwin+=1
        elif sb>sf: hloss+=1
        else: htie+=1
        if not use_judge:
            T+=1; verdict="JUDGE_OFF"
        else:
            judge_args.append((i, r["prompt"], r["reference"], a, b, ft_first))
            verdict=None
        detail.append({"i":i,"heuristic":"FT" if sf>sb else "BASE" if sb>sf else "TIE","judge":verdict,
                       "ft_ref":round(sf,3),"base_ref":round(sb,3),"ft_first":ft_first})
    if use_judge and judge_args:
        J({"phase":"judge_start","n":len(judge_args),"concurrency":10,"model":JUDGE_MODEL})
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(judge_call, p, rf, a, b): (i, ft_first) for (i,p,rf,a,b,ft_first) in judge_args}
            done = 0
            for fut in as_completed(futs):
                i, ft_first = futs[fut]
                v = fut.result()
                if v is None: T+=1; detail[i]["judge"]="JUDGE_ERR"
                elif v=="TIE": T+=1; detail[i]["judge"]="TIE"
                else:
                    ft_won = (v=="A") if ft_first else (v=="B")
                    W+=ft_won; L+=(not ft_won); detail[i]["judge"]="FT" if ft_won else "BASE"
                done+=1
                if done%25==0: J({"phase":"judge","done":done,"W":W,"L":L,"T":T,"h_win":hwin})
    n=W+L
    judge = {"wins":W,"losses":L,"ties":T,
             "win_rate":round(W/n,3) if n else None,
             "ci95":[round(wilson(W,n)[0],3),round(wilson(W,n)[1],3)] if n else None,
             "enabled":use_judge,
             "verdict":("IMPROVED" if (n and wilson(W,n)[0]>0.5) else "NOT_CONFIRMED" if n else "JUDGE_UNAVAILABLE")}
    hn=hwin+hloss
    heuristic_res={"win":hwin,"loss":hloss,"tie":htie,
             "win_rate":round(hwin/hn,3) if hn else None,
             "ci95":[round(wilson(hwin,hn)[0],3),round(wilson(hwin,hn)[1],3)] if hn else None,
             "verdict":("IMPROVED" if (hn and wilson(hwin,hn)[0]>0.5) else "NOT_CONFIRMED")}
    result={"n":len(rows),"judge":judge,"heuristic_ref_anchored":heuristic_res,"samples":detail}
    json.dump(result, open(RUN+"/eval-log.json","w"))
    J({"phase":"RESULT","judge":judge,"heuristic":heuristic_res})

if __name__=="__main__": main()
