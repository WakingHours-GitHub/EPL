''' Convert PyTorch model to ONNX model '''

import os
import torch
import numpy as np
import onnx

model_path = ""
print(model_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dummy_input = torch.randn(1, 3, 112, 112, device='cpu')
img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
img = img.astype(np.float32)
img = (img / 255. - 0.5) / 0.5  # torch style norm
img = img.transpose((2, 0, 1))
img = torch.from_numpy(img).unsqueeze(0).float()

model = torch.load(model_path)
model = model.cpu()
model.eval()
# print(model)

save_name = "model.onnx"
torch.onnx.export(model, img, save_name, keep_initializers_as_inputs=False, verbose=False, opset_version=11)

model = onnx.load(save_name)
graph = model.graph
graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
onnx.save(model, save_name)

os.system("zip model.zip model.onnx")

