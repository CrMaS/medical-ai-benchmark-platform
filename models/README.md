# Model artifact

Place a trained PyTorch exported-program classifier at:

```text
models/skin_lesion_classifier.pt2
```

The model contract is:

- input: one `float32` tensor shaped `[1, 3, 224, 224]`
- color: RGB
- resize: direct resize to `224 x 224`
- normalization mean: `[0.485, 0.456, 0.406]`
- normalization standard deviation: `[0.229, 0.224, 0.225]`
- output: logits tensor shaped `[1, 7]`
- class order: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`

The output may also be a tuple whose first item is the logits tensor, or a
dictionary containing it under `logits`.

Export an already trained and validated PyTorch model with:

```python
model.eval()
example = torch.zeros(1, 3, 224, 224)
exported_model = torch.export.export(model, (example,))
torch.export.save(exported_model, "models/skin_lesion_classifier.pt2")
```

Legacy TorchScript `.ts` artifacts with the same contract are also accepted.
Load only artifacts from a trusted source; serialized model files are executable
application components, not passive data.

Only deploy weights whose training data, label order, preprocessing, external
validation, and intended population are documented. The model artifact is
ignored by Git and should be distributed through an appropriate model registry.
