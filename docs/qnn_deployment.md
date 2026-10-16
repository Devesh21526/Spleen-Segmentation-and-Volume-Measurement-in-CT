# Edge deployment: QNN (Snapdragon HTP) and TensorRT

The laptop-side half of this is in the repository and runs today. The target-side half
needs the Qualcomm AI Engine Direct (QNN) SDK on an x86 Linux host and a board to run on,
so the commands below are written out rather than executed.

## 1. Graph preparation (runs here)

```bash
python -m src.models.export_onnx                 # PyTorch -> ONNX, opset 17
python -m src.models.qnn_prepare                 # -> models/spleen_unet_qnn.onnx
```

`qnn_prepare` makes three changes and then re-checks the graph against the original:

| Change | Why | Cost |
|---|---|---|
| `PRelu` → `LeakyRelu` (17 nodes) | every slope in this network is a single scalar, and converters cover LeakyRelu far better than per-tensor PRelu | exact, 0.0 difference |
| `BatchNormalization` folded into the preceding `ConvTranspose` (4 nodes) | one fewer op type to support, and no normalisation left to quantise separately | 3.1e-06 max logit difference |
| batch dimension pinned to 1 | QNN and TensorRT both plan for static shapes | none |

Op types left for the converter: `Conv`, `ConvTranspose`, `Add`, `Concat`, `LeakyRelu`.

## 2. Conversion (x86 Linux host with the QNN SDK)

```bash
source $QNN_SDK_ROOT/bin/envsetup.sh

qnn-onnx-converter \
  --input_network models/spleen_unet_qnn.onnx \
  --input_dim image 1,1,96,96,96 \
  --output_path spleen_unet.cpp

qnn-model-lib-generator -c spleen_unet.cpp -b spleen_unet.bin -t x86_64-linux-clang
qnn-context-binary-generator \
  --model libspleen_unet.so \
  --backend libQnnHtp.so \
  --binary_file spleen_unet_htp
```

Quantised build, once float works:

```bash
qnn-onnx-converter \
  --input_network models/spleen_unet_qnn.onnx \
  --input_dim image 1,1,96,96,96 \
  --input_list calibration/patches.txt \
  --act_bw 8 --weight_bw 8 \
  --output_path spleen_unet_w8a8.cpp
```

`calibration/patches.txt` lists raw float32 patches saved after the same pre-processing the
model was trained with: RAS, 1.5 × 1.5 × 2 mm, HU window −57…164 scaled to 0–1. A few hundred
96³ patches sampled across patients, half spleen-centred and half background, mirrors the
training crop ratio. Calibrating on anything else — unwindowed HU, a different spacing — produces
ranges the network never sees.

## 3. What to check on the target, in order

1. **3D kernel coverage.** `Conv3d` and `ConvTranspose3d` support on the HTP depends on the SDK
   version. If the converter drops them to CPU, the NPU buys nothing; a 2.5D or slice-wise
   variant of the architecture would be the fallback, and that is a retraining decision, not a
   conversion flag.
2. **Fixed shapes.** The 96³ sliding window is a natural fit: the patch size is constant and the
   window loop stays on the CPU.
3. **Dice after quantisation, not just logit error.** Boundary voxels flip first, and the number
   this project reports is a volume, so re-measure Dice and volume error on held-out scans.
4. **Pre-processing cost.** Resampling a 512 × 512 × N volume can cost more wall time than the
   network. Profile it separately before concluding the NPU helped.
5. **One process at a time.** Killing a QNN process mid-inference can leave the HTP protection
   domain allocated; the practical symptom is every later run dropping to a fraction of its speed
   until the board is rebooted.

## TensorRT (Jetson)

```bash
trtexec --onnx=models/spleen_unet_qnn.onnx --fp16 --saveEngine=spleen_unet_fp16.plan \
        --shapes=image:1x1x96x96x96
```

TensorRT has supported 3D convolutions for longer, so fp16 there is the lower-risk first target.
INT8 needs the same calibration set as above.

## Dynamic int8 on the x86 CPU (tried, and it does not run)

```bash
python -m src.models.quantize_onnx
```

19.3 MB → 7.3 MB, but ONNX Runtime's CPU provider has no `ConvInteger` kernel for 3D
convolution, so the quantised graph fails to load:

```
NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10) node
```

That is a CPU-provider gap rather than a property of the model: int8 3D convolution is exactly
the kind of kernel an NPU backend does provide. The file-size reduction is what carries over.
