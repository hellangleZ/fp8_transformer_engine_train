# fp8_transformer_engine_train
How to use FP8 to train LLM with multiple GPU


1- First install cudnn 8.92  cuda 12.1 Torch 2.3.1 
cudnn: https://developer.nvidia.com/rdp/cudnn-archive

2-Then install transformer_engine

pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

3- wait for make completed

4- download the two python files to same folder

5- use this command to trigger train:   torchrun --nproc_per_node=2 m_gpu.py

![image](https://github.com/user-attachments/assets/bbcc21a8-2ad4-4353-abea-130cac4de902)
![image](https://github.com/user-attachments/assets/d7094f46-e16c-4a76-a9af-451df9673a2f)





