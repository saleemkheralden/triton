# Triton

## Docker

```bash
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/models:/models nvcr.io/nvidia/tritonserver:24.01-py3 tritonserver --model-repository=/models
```

This would run a docker container with the three ports `8000-8082` open which are

```markdown
HTTPService at 8000
GRPCInferenceService at 8001
Metrics Service at 8002
```

## Model repository

For documentation: [see here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)

## Model "compilation" (Onnx)

For documentation: [see here](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)

If the model needs to be converted to a format that triton suuports, i.e. `.onnx`

Before starting you need to install `transformers` and `optimum` packages.

I used: `pip install transformers "optimum[onnxruntime]"`

After downloading the model from huggingface, you should run

```bash
optimum-cli export onnx --model <hf_name_or_path_to_model> <name_of_compiled_modle> --task <choose_from_optimum-cli_list>
```

Note that at this point you can deploy the model to triton, but it's not recommended since the model isn't optimized for production usecase.

## Deploy model to production

At this point you should have a `.onnx` model compiled.

In this section the model would be converted into `.engine/.plan`.

the python package will be used here, most common is the cli method.
