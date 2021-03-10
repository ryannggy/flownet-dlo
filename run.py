
import os

path = os.getcwd()
path += '/../tachysense-gray-50-6464/train'

cleanPath = path + '/clean'
finalPath = path + '/final'

# for subdir, dirs, files in os.walk(finalPath):
#     for d in dirs:
#         dir_tmp = d.split('/')
#         dirNo = dir_tmp[len(dir_tmp)-1]
os.system(f"python main.py --inference --model FlowNet2S --save_flow --inference_dataset MpiSintelFinal \
--inference_dataset_root /home/sawsn/tachysense-gray-50-6464/train/ --inference_visualize \
--resume /home/sawsn/FlowNet2S_model_best.pth.tar --save /home/sawsn/flownet-dlo/inference_results/final \
--skip_training --skip_validation")

# for subdir, dirs, files in os.walk(cleanPath):
#     for d in dirs:
#         dir_tmp = d.split('/')
#         dirNo = dir_tmp[len(dir_tmp)-1]
os.system(f"python main.py --inference --model FlowNet2S --save_flow --inference_dataset MpiSintelClean \
--inference_dataset_root /home/sawsn/tachysense-gray-50-6464/train/ --inference_visualize \
--resume /home/sawsn/FlowNet2S_model_best.pth.tar --save /home/sawsn/flownet-dlo/inference_results/clean \
--skip_training --skip_validation")
