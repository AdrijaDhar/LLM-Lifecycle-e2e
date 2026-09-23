"""A local web UI to chat with a trained checkpoint. Stdlib only - no new deps.

    uv run python scripts/chat_server.py checkpoints/135m-dpo-0921-1805/last
    uv run python scripts/chat_server.py checkpoints/135m-grpo-0921-1902/last --port 8801

Then open http://127.0.0.1:8800 (or whatever --port you passed).

Non-streaming by design: the server runs the full generation, then returns one
JSON response. Simpler than wiring up chunked/SSE streaming over stdlib
http.server, at the cost of a "thinking..." wait instead of a token-by-token
typewriter effect - a reasonable trade for a local dev tool.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from forge.data.chat import ChatFormat
from forge.model.config import PRESETS
from forge.model.generate import generate
from forge.model.transformer import Transformer
from forge.tokenizer.bpe import BPETokenizer
from forge.tokenizer.stream import StreamDecoder
from forge.training import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]

PAGE = """<!doctype html><meta charset="utf-8">
<title>Forge Chat</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{--bg:#12100d;--panel:#1a1613;--line:#332c25;--ink:#ece5d8;--dim:#a89c89;--faint:#6f6455;--ember:#ff9a44;--teal:#5fc2b6;}
*{box-sizing:border-box;}
html,body{height:100%;}
body{margin:0;background:var(--bg);color:var(--ink);font-family:'IBM Plex Sans',sans-serif;display:flex;flex-direction:column;}
header{padding:16px 20px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:12px;flex-shrink:0;}
header h1{font-family:'Big Shoulders Display',sans-serif;font-size:22px;text-transform:uppercase;margin:0;font-weight:800;}
header .ckpt{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--faint);}
#log{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:14px;max-width:760px;width:100%;margin:0 auto;}
.msg{max-width:80%;padding:10px 14px;line-height:1.55;font-size:14.5px;white-space:pre-wrap;}
.msg.user{align-self:flex-end;background:#22201c;border:1px solid var(--line);border-radius:10px 10px 2px 10px;}
.msg.assistant{align-self:flex-start;background:var(--panel);border-left:2px solid var(--ember);border-radius:2px 10px 10px 10px;}
.msg.pending{color:var(--faint);font-family:'IBM Plex Mono',monospace;font-size:12.5px;}
form{border-top:1px solid var(--line);padding:14px 20px;display:flex;gap:10px;max-width:760px;width:100%;margin:0 auto;flex-shrink:0;}
textarea{flex:1;background:var(--panel);border:1px solid var(--line);color:var(--ink);font-family:'IBM Plex Sans',sans-serif;font-size:14.5px;padding:10px 12px;resize:none;height:44px;}
textarea:focus{outline:2px solid var(--ember);outline-offset:-1px;}
button{background:var(--ember);color:#1a1310;border:none;font-weight:600;padding:0 22px;cursor:pointer;font-family:'IBM Plex Sans',sans-serif;font-size:14px;}
button:disabled{opacity:0.4;cursor:default;}
button:hover:not(:disabled){filter:brightness(1.08);}
</style>
<header><h1>Forge Chat</h1><span class="ckpt" id="ckptname"></span></header>
<div id="log"></div>
<form id="f"><textarea id="msg" placeholder="Ask something..." autofocus></textarea><button id="send">Send</button></form>
<script>
const log = document.getElementById('log');
const form = document.getElementById('f');
const input = document.getElementById('msg');
const btn = document.getElementById('send');
let history = [];

fetch('/info').then(r=>r.json()).then(d=>{
  document.getElementById('ckptname').textContent = `${d.preset} · step ${d.step}`;
});

function addMsg(role, text){
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const text = input.value.trim();
  if(!text) return;
  addMsg('user', text);
  history.push({role:'user', content:text});
  input.value = '';
  btn.disabled = true;
  const pending = addMsg('pending', 'thinking...');
  try{
    const r = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({messages: history})});
    const d = await r.json();
    pending.remove();
    addMsg('assistant', d.reply);
    history.push({role:'assistant', content:d.reply});
  }catch(err){
    pending.textContent = 'error: ' + err;
  }
  btn.disabled = false;
  input.focus();
});
input.addEventListener('keydown', (e)=>{
  if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); form.requestSubmit(); }
});
</script>
"""


def build_handler(model, tok, fmt, state, args):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt_, *a):  # quiet the default per-request stderr spam
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/info":
                self._json({"preset": state["preset"], "step": state["step"]})
            else:
                self._json({"error": "not found"}, 404)

        def do_HEAD(self):
            # Some browsers/proxies probe with HEAD before GET; BaseHTTPRequestHandler
            # 501s anything without an explicit handler, which can look like the page
            # "won't open" even though GET works fine.
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path != "/chat":
                self._json({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            messages = payload.get("messages", [])

            ids: list[int] = []
            for m in messages:
                if m["role"] == "user":
                    ids += [fmt.user_id] + tok.encode(m["content"]) + [fmt.end_id]
                else:
                    ids += [fmt.assistant_id] + tok.encode(m["content"]) + [fmt.end_id]
            ids += [fmt.assistant_id]

            dec = StreamDecoder(tok)
            chunks = [
                dec.push(tid)
                for tid in generate(
                    model, ids, max_new_tokens=args.tokens, temperature=args.temperature,
                    top_k=args.top_k, eot_id=fmt.end_id, repetition_penalty=args.repetition_penalty,
                )
            ]
            chunks.append(dec.flush())
            self._json({"reply": "".join(chunks).strip()})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data" / "tokenizer" / "fw32k-chat.bpe.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--repetition-penalty", type=float, default=1.3)
    args = ap.parse_args()

    st = json.loads((args.ckpt / "state.json").read_text())
    tok = BPETokenizer()
    tok.load(str(args.tokenizer))
    fmt = ChatFormat.register(tok)
    cfg = PRESETS[st["preset"]]
    cfg.vocab_size = tok.vocab_size

    print(f"loading {args.ckpt} ...")
    model = Transformer(cfg)
    if st.get("quantized"):
        nn.quantize(model, group_size=st.get("group_size", 64), bits=st["bits"])
    checkpoint.load(args.ckpt, model)
    model.eval()
    mx.eval(model.parameters())
    print(f"ready: {st['preset']} ({cfg.n_params / 1e6:.1f}M params){' [quantized]' if st.get('quantized') else ''}")

    handler = build_handler(model, tok, fmt, st, args)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"-> http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
