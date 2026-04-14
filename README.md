# Symbolic_VLA

## libero_vla_labeler
1. Download the libero dataset and put it under `~/`. Additionally, download the bddl files in libero dataset and put them in the libero dataset folder as `libero_90_bddl` and `libero_10_bddl`
2. Configure GPT and Gemini api key:
   `echo 'export GEMINI_API_KEY="..."' >> ~/.bashrc`
   `echo 'export OPENAI_API_KEY="..."' >> ~/.bashrc`
4. In virtual environment, run `python main.py --dataset_dir /home/tiandy/libero_100 --suite libero_90 --config config/config.yaml --output output_90`
