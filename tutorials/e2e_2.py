import argparse
import numpy as np
import torch
from torch.export import export

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# ===============================
# Argument
# ===============================

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

MODEL = args.model
print("MODEL:", MODEL)

# ===============================
# Load Model
# ===============================

if MODEL == "mobilenetv3":
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    torch_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).eval()
    input_shape = (1, 3, 224, 224)
    example_args = (torch.randn(*input_shape),) # 실수 입력

elif MODEL == "densenet169":
    from torchvision.models import densenet169, DenseNet169_Weights
    torch_model = densenet169(
        weights=DenseNet169_Weights.DEFAULT
    ).eval()
    input_shape = (1,3,224,224)

elif MODEL == "resnet18":
    from torchvision.models import resnet18, ResNet18_Weights
    # 모델 로드 및 평가 모드 설정
    torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    input_shape = (1, 3, 224, 224)
    # 이미지 모델이므로 float32 실수 입력 사용
    example_args = (torch.randn(*input_shape),)

elif MODEL == "resnet50":
    from torchvision.models import resnet50, ResNet50_Weights
    # 모델 로드 및 평가 모드 설정
    torch_model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
    input_shape = (1, 3, 224, 224)
    example_args = (torch.randn(*input_shape),)

elif MODEL == "gpt2":
    import torch._dynamo
    torch._dynamo.disable()
    import torch

    from transformers import GPT2LMHeadModel, GPT2Config

    # =========================
    # 1. config (핵심 설정)
    # =========================
    config = GPT2Config.from_pretrained("gpt2")

    config.use_cache = False
    config.return_dict = True
    config._attn_implementation = "eager"   # ⭐ SDPA 제거 (핵심)

    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config).eval()

    # =========================
    # 2. input (고정 shape)
    # =========================
    input_shape = (1, 128)

    input_ids = torch.randint(0, config.vocab_size, input_shape, dtype=torch.long)
    attention_mask = torch.ones(input_shape, dtype=torch.long)

    example_args = (input_ids, attention_mask)

    # =========================
    # 3. wrapper (출력 단순화)
    # =========================
    class GPT2Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask=None):
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,          # 명시적으로 차단
                return_dict=True
            )

            return outputs.logits

    wrapped_model = GPT2Wrapper(model).eval()

    # =========================
    # 4. ONNX export
    # =========================
    onnx_path = "gpt2.onnx"
    torch.onnx.export(
        wrapped_model,
        example_args,
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes=None,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX  # 핵심
    )
    print(f"ONNX model saved to {onnx_path}")

    # =========================
    # 5. 반환
    # =========================
    torch_model = wrapped_model
else:
    raise RuntimeError(f"Unsupported model: {MODEL}")

# ===============================
# Example Input
# ===============================

example_args = (torch.randn(*input_shape),)

# ===============================
# Export Torch
# ===============================

with torch.no_grad():
    exported_program = export(torch_model, example_args)

# ===============================
# Convert to TVM IR
# ===============================

mod = from_exported_program(exported_program, keep_params_as_input=True)
mod, params = relax.frontend.detach_params(mod)

# ===============================
# Tuning setup
# ===============================

target = tvm.target.Target("nvidia/geforce-rtx-4090")

TOTAL_TRIALS = 20000
work_dir = f"tuning_logs_{MODEL}"

print("Trials:", TOTAL_TRIALS)
print("Work dir:", work_dir)

# ===============================
# Run tuning pipeline
# ===============================

pipeline = relax.get_pipeline(
    "static_shape_tuning",
    target=target,
    total_trials=TOTAL_TRIALS,
    work_dir=work_dir
)

mod = pipeline(mod)

# ===============================
# Build
# ===============================

ex = relax.build(mod, target=target)

# ===============================
# Run VM
# ===============================

dev = tvm.device(target.kind.name, 0)

vm = relax.VirtualMachine(ex, dev)

gpu_data = tvm.nd.array(
    np.random.rand(*input_shape).astype("float32"), dev
)

gpu_params = [tvm.nd.array(p, dev) for p in params["main"]]

out = vm["main"](gpu_data, *gpu_params)

print("Finished tuning:", MODEL)

# ===============================
# Build & Input Prep
# ===============================

# 모델별로 입력 데이터 타입과 생성 방식을 다르게 설정
if MODEL == "gpt2":
    # GPT-2는 정수형 토큰 ID가 필요함
    raw_data = np.random.randint(0, 50257, input_shape).astype("int64")
else:
    # CNN 모델은 실수형 이미지가 필요함
    raw_data = np.random.rand(*input_shape).astype("float32")

gpu_data = tvm.nd.array(raw_data, dev)