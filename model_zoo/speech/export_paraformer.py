"""Paraformer-large contextual (FunASR) -> three fixed-shape ONNX stages.

The split and the tensor contract are the ones RCDL's ParaformerAsr reads:
  encoder    speech [1,T,560]                       -> enc [1,T,512]
  predictor  enc [1,T,512]                          -> alphas [1,T+1], hidden [1,T+1,512]
  decoder    context [1,T,512], token_count [1],
             bias [1,1,512], acoustic [1,N,512]     -> logits [1,N,V]
T = 400 LFR frames (24 s), N = 100 tokens.

What is NOT in the graphs, on purpose:
- CIF. The predictor stops at the per-frame weights; the integrate-and-fire loop
  runs on the CPU, where the tail weight can go at the true end of speech
  instead of the end of the padding.
- Length masks. The encoder sees the zero-padded window as valid, the same as
  the build RCDL's decoder logic was written against. The decoder's token mask
  is clip(count - i, 0, 1), arithmetic on a float count, so no shape depends on
  a value and nothing needs a dynamic arange.
"""
import argparse

import torch
from funasr import AutoModel
from funasr.register import tables

T, N, D = 400, 100, 512


class Encoder(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, speech):
        out, _ = self.enc(speech, torch.tensor([T], dtype=torch.int32))
        return out


class Predictor(torch.nn.Module):
    def __init__(self, pred):
        super().__init__()
        self.pred = pred

    def forward(self, enc):
        mask = torch.ones(1, 1, T)
        alphas, _ = self.pred.forward_cnn(enc, mask)                      # [1, T]
        alphas = torch.cat([alphas, torch.zeros(1, 1)], dim=1)           # [1, T+1]
        hidden = torch.cat([enc, torch.zeros(1, 1, D)], dim=1)           # [1, T+1, D]
        return alphas, hidden


class Decoder(torch.nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.d = dec
        self.register_buffer("pos", torch.arange(N, dtype=torch.float32).view(1, N))

    def forward(self, context, token_count, bias, acoustic):
        d = self.d
        tgt_mask = torch.clamp(token_count.view(1, 1) - self.pos, 0.0, 1.0)   # [1, N]
        tgt_mask, _ = d.prepare_mask(tgt_mask)
        _, memory_mask = d.prepare_mask(torch.ones(1, T))
        x, tgt_mask, memory, memory_mask, _ = d.model.decoders(acoustic, tgt_mask, context, memory_mask)
        _, _, x_self_attn, x_src_attn = d.last_decoder(x, tgt_mask, memory, memory_mask)
        ctx_mask, _ = d.prepare_mask(torch.ones(1, 1))
        ctx_mask = ctx_mask.transpose(2, 1).unsqueeze(1)
        cx, tgt_mask, _, _, _ = d.bias_decoder(x_self_attn, tgt_mask, bias, memory_mask=ctx_mask)
        x = torch.cat([x_src_attn, cx], dim=2)
        x = d.bias_output(x.transpose(1, 2)).transpose(1, 2)
        x = x_self_attn + x
        if d.model.decoders2 is not None:
            x, tgt_mask, memory, memory_mask, _ = d.model.decoders2(x, tgt_mask, memory, memory_mask)
        x, tgt_mask, memory, memory_mask, _ = d.model.decoders3(x, tgt_mask, memory, memory_mask)
        return d.output_layer(d.after_norm(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    am = AutoModel(model=a.model_dir, device="cpu", disable_update=True)
    m = am.model.eval()
    enc = tables.encoder_classes.get("SANMEncoderExport")(m.encoder, onnx=True).eval()
    pred = tables.predictor_classes.get("CifPredictorV2Export")(m.predictor, onnx=True).eval()
    dec = tables.decoder_classes.get("ContextualParaformerDecoderExport")(m.decoder, onnx=True).eval()
    vocab = m.decoder.output_layer.out_features
    jobs = [
        (Encoder(enc), (torch.randn(1, T, 560),), ["speech"], ["enc"], "paraformer_encoder.onnx"),
        (Predictor(pred), (torch.randn(1, T, D),), ["enc"], ["alphas", "hidden"], "paraformer_predictor.onnx"),
        (Decoder(dec), (torch.randn(1, T, D), torch.tensor([7.0]), torch.zeros(1, 1, D), torch.randn(1, N, D)),
         ["context", "token_count", "bias", "acoustic"], ["logits"], "paraformer_decoder.onnx"),
    ]
    with torch.no_grad():
        for mod, args, ins, outs, name in jobs:
            torch.onnx.export(mod, args, f"{a.out}/{name}", input_names=ins, output_names=outs,
                              opset_version=17, do_constant_folding=True)
            print(">>", name)
    print("vocab", vocab)


if __name__ == "__main__":
    main()
