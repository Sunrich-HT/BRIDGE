"""Minimal smoke test: build a BRIDGE model without a backbone and verify
that the triangular refinement + memory update + injector all wire up.
Runs on CPU; uses a tiny config so it finishes in seconds.
"""

from __future__ import annotations

import torch

from bridge import BridgeConfig, BridgeForCausalLM, HierarchicalMemory


def main() -> None:
    config = BridgeConfig(
        backbone_hidden_size=128,
        backbone_vocab_size=1000,
        d_o=64,
        d_l=64,
        d_m_tier=64,
        cross_attention_heads=4,
        habit_bank_size=32,
        slow_depth=2,
    )

    model = BridgeForCausalLM(config, backbone=None)
    model.eval()

    B, T = 2, 16
    input_ids = torch.randint(0, config.backbone_vocab_size, (B, T))

    # First call -- memory is initialized from the persona representation.
    # (When a real backbone is attached, pass `persona_input_ids` instead and
    # the wrapper runs `_encode_persona` for you.)
    persona_repr = torch.randn(B, config.backbone_hidden_size)
    out = model(input_ids=input_ids, persona_repr=persona_repr, labels=input_ids)

    print("logits           :", tuple(out.logits.shape))
    print("loss             :", float(out.loss))
    print("final residual   :", float(out.refinement.final_residual.mean()))
    print("contraction proxy:", float(out.refinement.contraction_proxy.mean()))
    print("o_K              :", tuple(out.o_K.shape))
    print("l_K              :", tuple(out.l_K.shape))
    print("m_K (concat)     :", tuple(out.m_K_concat.shape))

    # Drive a few more turns to see memory evolve.
    state = out.memory_state
    drifts = []
    for _ in range(5):
        out = model(input_ids=input_ids, memory_state=state)
        state = out.memory_state
        drifts.append(float(HierarchicalMemory.personality_drift(state).mean()))
    print("personality drift trajectory:", drifts)

    V = HierarchicalMemory.lyapunov_energy(
        state, (config.gamma_e, config.gamma_a, config.gamma_p)
    ).mean()
    bound = model.model.memory.lyapunov_bound(state)
    print(f"V(m_t) = {float(V):.4f}, theoretical V_max = {float(bound):.4f}, "
          f"ratio = {float(V) / float(bound):.4f}")


if __name__ == "__main__":
    main()
