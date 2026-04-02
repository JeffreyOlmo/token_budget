"""Reference model server for KL divergence computation.

Adapted from the original ref_server.py to handle the response_mask-based
format used by the token budget experiment. Instead of a single prompt_length
cutoff, we use a binary mask to identify which tokens are model-generated.
"""

import io
import json
import os
import queue
import threading

import torch


# ── Serialization (same as original) ─────────────────

def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()

def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b), weights_only=True)

def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()

def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist


if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from bottle import request
    import bottle

    from config import model_path, ref_port

    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    print(f"Loading reference model from {model_path}...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        _attn_implementation="sdpa").to('cuda')
    ref_model.eval()
    ref_model.requires_grad_(False)
    print("Reference model loaded.")

    def get_per_token_logps(input_ids):
        logits = ref_model(input_ids).logits
        logits = logits[:, :-1, :]
        ids = input_ids[:, 1:]
        per_token_logps = []
        for logits_row, ids_row in zip(logits, ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_lp = torch.gather(
                log_probs, dim=1, index=ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_lp)
        return torch.stack(per_token_logps)

    raw_queue = queue.LifoQueue()
    result_queue = queue.Queue()

    app = bottle.Bottle()

    @app.route('/upload', method='POST')
    def do_upload():
        dd = request.body.read()
        dd = bytes_list_to_list(dd)
        # Format: [metadata, input_ids, response_mask, rewards, gen_logps?, budget_ids?]
        data = {'base': json.loads(dd[0])}
        data['inputs'] = bytes_to_tensor(dd[1])
        data['response_mask'] = bytes_to_tensor(dd[2])
        data['rewards'] = bytes_to_tensor(dd[3])
        # Remaining fields are passed through as raw bytes (gen_logps, budget_ids, etc.)
        data['extra_blobs'] = [dd[i] for i in range(4, len(dd))]
        raw_queue.put(data)
        print('receive', data['inputs'].shape, 'rewards:', data['rewards'])
        return b'ok'

    @app.route('/get', method='GET')
    def do_get():
        if result_queue.empty():
            return b'empty'
        return result_queue.get()

    def run_server():
        bottle.run(app, host='0.0.0.0', port=ref_port, server='tornado')

    threading.Thread(target=run_server, daemon=False).start()
    print(f"Ref server running on port {ref_port}")

    while True:
        d = raw_queue.get()
        inputs = d['inputs'].to(ref_model.device)

        # Skip sequences that are too long
        if inputs.shape[1] > 4096:
            print(f"Skipping batch with seq_len={inputs.shape[1]}")
            continue

        with torch.inference_mode():
            ref_logps = get_per_token_logps(inputs)
        # ref_logps is (B, L-1) — full sequence logprobs

        data = [
            json.dumps(d['base']).encode(),
            tensor_to_bytes(d['inputs'].cpu()),
            tensor_to_bytes(d['response_mask'].cpu()),
            tensor_to_bytes(d['rewards'].cpu()),
            tensor_to_bytes(ref_logps.cpu()),
        ]
        # Pass through extra blobs (gen_logps, budget_ids, etc.)
        for blob in d.get('extra_blobs', []):
            data.append(blob)

        result_queue.put(make_bytes_list(data))
