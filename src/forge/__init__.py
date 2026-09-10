"""forge — an end-to-end small-LLM lifecycle toolkit.

Two tracks share this library:
  - Track A: build a compact model from scratch (tokenizer -> pretrain -> SFT ->
    DPO -> GRPO -> eval -> quantize), all local on Apple Silicon via MLX.
  - Track B ("TerraForge"): fine-tune a small open model to write Terraform, with
    `terraform validate`/`plan` + OPA policies as verifiable rewards. Lives in
    `forge.terra`.
"""

__version__ = "0.1.0"


def main() -> None:
    print("forge — see docs/roadmap.md")
