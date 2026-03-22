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
    input_shape = (1,3,224,224)

elif MODEL == "densenet169":
    from torchvision.models import densenet169, DenseNet169_Weights
    torch_model = densenet169(weights=DenseNet169_Weights.DEFAULT).eval()
    input_shape = (1,3,224,224)


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
# Run tuning
# ===============================

mod = relax.get_pipeline(
    "static_shape_tuning",
    target=target,
    total_trials=TOTAL_TRIALS,
    work_dir = work_dir
)(mod)

# ===============================
# Build
# ===============================

ex = relax.build(mod, target=target)

dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(ex, dev)

gpu_data = tvm.nd.array(
    np.random.rand(*input_shape).astype("float32"), dev
)

gpu_params = [tvm.nd.array(p, dev) for p in params["main"]]

out = vm["main"](gpu_data, *gpu_params)

print("Finished tuning:", MODEL)