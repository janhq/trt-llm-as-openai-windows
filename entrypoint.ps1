setx Path "%Path%;C:\Program Files\Python310\Lib"

pip install --upgrade pip

pip install -r C:\workspace\app\requirements.txt

pip install C:\workspace\app\TensorRT-9.2.0.5\python\tensorrt-9.2.0.post12.dev5-cp310-none-win_amd64.whl

pip install tensorrt_llm==0.7.1 --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu121

python -c "import tensorrt_llm; print(tensorrt_llm._utils.trt_version())"

# This is hard-coded for llama2 13B model with mounted folder at C:\workspace\model
cd C:\workspace\app\TensorRT-LLM-0.7.1\
python examples\llama\build.py --model_dir C:\\workspace\\model\\llama2_13b\\tokenizer --quant_ckpt_path C:\\workspace\\model\\llama2_13b\\ckpt_int4\\llama_tp1_rank0.npz --dtype float16 --use_gpt_attention_plugin float16 --use_gemm_plugin float16 --use_weight_only --weight_only_precision int4_awq --per_group --enable_context_fmha --max_batch_size 1 --max_input_len 3500 --max_output_len 1024 --output_dir C:\\workspace\\model\\llama2_13b\\engine2

# Run the flask server in development mode at port 8080 given the model is mounted
cd C:\workspace\app\ && python app.py --trt_engine_path C:\\workspace\\model\\llama2_13b\\engine2 --trt_engine_name llama_float16_tp1_rank0.engine --tokenizer_dir_path C:\\workspace\\model\\llama2_13b\\tokenizer --port 8080