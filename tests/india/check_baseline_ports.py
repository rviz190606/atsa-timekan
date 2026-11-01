"""Check native R outputs/input gradients against the original Python classes.

Python is used only for this cross-language verification, never for training.
Run after: Rscript indian_learned_baselines.R --task self_test
"""
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def original_classes(source, names, namespace, functional_normalization=False):
    tree = ast.parse(source)
    if functional_normalization:
        # Upstream x_enc /= stdev prevents gradients w.r.t. the input in recent
        # PyTorch. Only for this gradient check, use the mathematically identical
        # out-of-place division. Ordinary training inputs do not require gradients.
        class OutOfPlaceDivision(ast.NodeTransformer):
            count = 0
            def visit_AugAssign(self, node):
                if (isinstance(node.op, ast.Div) and isinstance(node.target, ast.Name)
                        and node.target.id == "x_enc" and isinstance(node.value, ast.Name)
                        and node.value.id == "stdev"):
                    self.count += 1
                    return ast.copy_location(ast.Assign(targets=[node.target],
                        value=ast.BinOp(left=ast.Name(id="x_enc", ctx=ast.Load()),
                            op=ast.Div(), right=node.value)), node)
                return node
        transform = OutOfPlaceDivision()
        tree = ast.fix_missing_locations(transform.visit(tree))
        assert transform.count == 1
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert {node.name for node in classes} == set(names)
    exec(compile(ast.Module(body=classes, type_ignores=[]), "official_reference", "exec"), namespace)


def main():
    root = Path(__file__).resolve().parents[2]
    sources = json.loads((root / "third_party/india_baselines/baseline_official_sources.json").read_text())
    directory = root / "results/india/learned_baselines/port_checks"
    shared = dict(torch=torch, nn=nn, F=F, np=np, sqrt=math.sqrt)
    dlinear = shared.copy()
    original_classes(sources["DLinear.py"], ["moving_avg", "series_decomp", "Model"], dlinear)
    itransformer = shared.copy()
    original_classes(sources["SelfAttention_Family.py"], ["FullAttention", "AttentionLayer"], itransformer)
    original_classes(sources["Transformer_EncDec.py"], ["EncoderLayer", "Encoder"], itransformer)
    original_classes(sources["Embed.py"], ["DataEmbedding_inverted"], itransformer)
    original_classes(sources["iTransformer.py"], ["Model"], itransformer, functional_normalization=True)
    print("Gradient check only: upstream in-place normalization uses equivalent out-of-place division.")
    config = SimpleNamespace(seq_len=8, pred_len=4, enc_in=3, individual=False,
        d_model=8, d_ff=16, n_heads=2, e_layers=2, dropout=0., activation="gelu",
        output_attention=False, use_norm=True, factor=5, class_strategy="projection",
        embed="timeF", freq="h")
    report = []
    for name, namespace in [("DLinear", dlinear), ("iTransformer", itransformer)]:
        model = namespace["Model"](config).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.linspace(-.1, .1, parameter.numel()).reshape(parameter.shape))
        x = torch.linspace(-1, 2, 48).reshape(2, 8, 3).requires_grad_(True)
        y = model(x) if name == "DLinear" else model(x, None, None, None)
        y.square().sum().backward()
        for label, value in [("output", y), ("input_gradient", x.grad)]:
            r_value = np.loadtxt(directory / f"{name}_{label}.csv")
            expected = value.detach().numpy().flatten(order="F")
            np.testing.assert_allclose(r_value, expected, atol=5e-6, rtol=3e-5)
            error = float(np.max(np.abs(r_value - expected)))
            print(f"{name} {label}: max absolute R/Python difference {error:.3g}")
            report.append(dict(model=name, tensor=label, max_absolute_difference=error))
    (directory / "official_parity_report.json").write_text(json.dumps(report, indent=2))
    print("OFFICIAL_DLINEAR_AND_ITRANSFORMER_FORWARD_AND_GRADIENT_PARITY_OK")


if __name__ == "__main__":
    main()
