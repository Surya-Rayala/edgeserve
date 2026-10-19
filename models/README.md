# Local model folders

A place for the model folders this machine serves. Export a model here, or copy in a folder that
TensorRT Edge-LLM 0.11 already exported (its `llm` folder), and serve it:

```bash
edgeserve export <checkpoint> models/<name>
edgeserve serve --model models/<name>
```

Git ignores everything in this folder except this file: model weights are large and have their own
licenses. How to export models, including your own fine-tuned, pruned or quantized checkpoints:
[docs/EXPORT.md](../docs/EXPORT.md).
