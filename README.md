# Symbolic_VLA

## libero_vla_labeler
1. Download the libero dataset. Additionally, download the bddl files in libero dataset and put them in the libero dataset folder as `libero_90_bddl` and `libero_10_bddl`
2. Configure GPT and Gemini api key:
   ```python
   echo 'export GEMINI_API_KEY="..."' >> ~/.bashrc
   echo 'export OPENAI_API_KEY="..."' >> ~/.bashrc
   ```
3. In virtual environment, run `python main.py --dataset_dir /path/to/dataset --suite libero_90 --config config/config.yaml --output output_90`
4. `view_hdf5_frames.py` is for visualization of the demo. Launch this GUI to verify the segmentation points and annotations.
