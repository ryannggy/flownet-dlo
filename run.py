
import os

path = os.getcwd()
path += '/../tachysense-gray-50-6464/train'

cleanPath = path + '/clean'
finalPath = path + '/final'

for subdir, dirs, files in os.walk(finalPath):
    for d in dirs:
        dir_tmp = d.split('/')
        dirNo = dir_tmp[len(dir_tmp)-1]
        os.system(f"python main.py --inference --model FlowNet2S --save_flow --inference_dataset ImagesFromFolder \
        --inference_dataset_root /home/sawsn/tachysense-gray-50-6464/train/final/{dirNo} \
        --inference_visualize \
        --resume /home/sawsn/FlowNet2S_model_best.pth.tar \
        --save /home/sawsn/flownet-dlo/inference_results/final/{dirNo}")

for subdir, dirs, files in os.walk(cleanPath):
    for d in dirs:
        dir_tmp = d.split('/')
        dirNo = dir_tmp[len(dir_tmp)-1]
        os.system(f"python main.py --inference --model FlowNet2S --save_flow --inference_dataset ImagesFromFolder \
        --inference_dataset_root /home/sawsn/tachysense-gray-50-6464/train/clean/{dirNo} \
        --inference_visualize \
        --resume /home/sawsn/FlowNet2S_model_best.pth.tar \
        --save /home/sawsn/flownet-dlo/inference_results/clean/{dirNo}")
