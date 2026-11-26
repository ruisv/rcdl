"""Check the three-stage split against FunASR end to end, and save NPU references.

For each clip: FunASR's own front end (fbank + LFR + CMVN) -> zero-pad to T ->
encoder -> predictor -> CIF on the CPU (tail at the true end) -> decoder ->
argmax. The text must equal FunASR's own transcription of the same clip.
"""
import json
import sys

import numpy as np
import onnxruntime as ort
from funasr import AutoModel

T, N, D = 400, 100, 512


def cif(alphas, hidden, valid, threshold=1.0, tail=0.45, max_tokens=N):
    embeds, frame, integ, count = np.zeros((max_tokens, D), np.float64), np.zeros(D), 0.0, 0
    for t in range(valid + 1):
        is_tail = t == valid
        a = tail if is_tail else max(0.0, float(alphas[t]))
        h = np.zeros(D) if is_tail else hidden[t].astype(np.float64)
        completion = 1.0 - integ
        integ += a
        fire = integ >= threshold
        cur = completion if fire else a
        frame += cur * h
        if fire:
            if count < max_tokens:
                embeds[count] = frame
            count += 1
            integ -= 1.0
            frame = (a - cur) * h
    return embeds.astype(np.float32), min(count, max_tokens)


def main(model_dir, audio_dir, out_npz):
    am = AutoModel(model=model_dir, device="cpu", disable_update=True)
    front = am.kwargs["frontend"]
    vocab = json.load(open(f"{model_dir}/tokens.json", encoding="utf-8"))
    enc = ort.InferenceSession("paraformer_encoder.onnx", providers=["CPUExecutionProvider"])
    pred = ort.InferenceSession("paraformer_predictor.onnx", providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession("paraformer_decoder.onnx", providers=["CPUExecutionProvider"])
    import soundfile as sf
    import torch
    refs = {}
    rows = [l.rstrip("\n").split("\t") for l in open(f"{audio_dir}/transcripts.tsv", encoding="utf-8")]
    ok = True
    for k, (name, want) in enumerate(rows):
        pcm, sr = sf.read(f"{audio_dir}/{name}", dtype="float32")
        assert sr == 16000
        feats, flen = front(torch.from_numpy(pcm)[None], torch.tensor([len(pcm)]))
        feats = feats[0, : int(flen[0])].numpy()
        valid = min(len(feats), T)
        x = np.zeros((1, T, 560), np.float32)
        x[0, :valid] = feats[:valid]
        e = enc.run(None, {"speech": x})[0]
        alphas, hidden = pred.run(None, {"enc": e})
        emb, count = cif(alphas[0], hidden[0], valid)
        logits = dec.run(None, {"context": e, "token_count": np.array([count], np.float32),
                                "bias": np.zeros((1, 1, D), np.float32), "acoustic": emb[None]})[0]
        ids = logits[0, :count].argmax(-1)
        text = "".join(vocab[i] for i in ids if not vocab[i].startswith("<"))
        funasr_text = am.generate(input=f"{audio_dir}/{name}")[0]["text"].replace(" ", "")
        good = text == want.replace(" ", "")
        ok &= good
        print(f"{name}: split {text!r} | FunASR {funasr_text!r} | want {want!r} {'OK' if good else 'MISMATCH'}")
        refs.update({f"{k}/in/speech": x, f"{k}/out/enc": e, f"{k}/valid": np.array(valid),
                     f"{k}/out/alphas": alphas, f"{k}/out/hidden": hidden,
                     f"{k}/in/token_count": np.array([count], np.float32),
                     f"{k}/in/acoustic": emb[None], f"{k}/out/logits": logits})
    np.savez_compressed(out_npz, **refs)
    print("all match" if ok else "SOME MISMATCH")


if __name__ == "__main__":
    main(*sys.argv[1:4])
