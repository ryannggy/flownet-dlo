
import os

path = os.getcwd()
path += '/../tachysense-gray-50-6464/train'

cleanPath = path + '/clean'
finalPath = path + '/final'

for subdir, dirs, files in os.walk(finalPath):
    for d in dirs:
        dir_tmp = d.split('/')
        dirNo = dir_tmp[len(dir_tmp)-1]; print(f"Working on: {dirNo}")
        os.system(f"CUDA_VISIBLE_DEVICES=1 python main.py --inference --model FlowNet2S --save_flow \
        --inference_dataset ImagesFromFolder --inference_visualize \
        --inference_dataset_root /home/sawsn/tachysense-gray-50-6464/train/clean/{dirNo} \
        --resume /home/sawsn/FlowNet2S_model_best.pth.tar \
        --save /home/sawsn/flownet-dlo/inference_results/final/{dirNo}")

for subdir, dirs, files in os.walk(cleanPath):
    for d in dirs:
        dir_tmp = d.split('/')
        dirNo = dir_tmp[len(dir_tmp)-1]; print(f"Working on: {dirNo}")
        os.system(f"CUDA_VISIBLE_DEVICES=1 python main.py --inference --model FlowNet2S --save_flow \
        --inference_dataset ImagesFromFolder --inference_visualize \
        --inference_dataset_root /home/sawsn/tachysense-gray-50-6464/train/final/{dirNo} \
        --resume /home/sawsn/FlowNet2S_model_best.pth.tar \
        --save /home/sawsn/flownet-dlo/inference_results/clean/{dirNo}")

# os.system(f"python main.py --inference --model FlowNet2S --save_flow --inference_dataset MpiSintelFinal \
# --inference_dataset_root /home/sawsn/tachysense-gray-50-6464/train/final --inference_visualize \
# --resume /home/sawsn/FlowNet2S_model_best.pth.tar --save /home/sawsn/flownet-dlo/inference_results/final/ \
# --skip_training --skip_validation")

# os.system(f"python main.py --inference --model FlowNet2S --save_flow --inference_dataset MpiSintelClean \
# --inference_dataset_root /home/sawsn/tachysense-gray-50-6464/train/clean --inference_visualize \
# --resume /home/sawsn/FlowNet2S_model_best.pth.tar --save /home/sawsn/flownet-dlo/inference_results/clean/ \
# --skip_training --skip_validation")



# for subdir, dirs, files in os.walk(finalPath):
#     for d in dirs:
#         dir_tmp = d.split('/')
#         dirNo = dir_tmp[len(dir_tmp)-1]; print(f"Working on: {dirNo}")
#         os.system(f"python main.py --inference --model FlowNet2S --save_flow --inference_dataset ImagesFromFolder \
#         --inference_dataset_root /home/sawsn/tachysense-gray-50-6464/train/final/{dirNo} --inference_visualize \
#         --resume /home/sawsn/FlowNet2S_model_best.pth.tar --save /home/sawsn/flownet-dlo/inference_results/final/{dirNo} \
#         --skip_training --skip_validation --inference_n_batches 1")
#         quit()

# for subdir, dirs, files in os.walk(cleanPath):
#     for d in dirs:
#         dir_tmp = d.split('/')
#         dirNo = dir_tmp[len(dir_tmp)-1]; print(f"Working on: {dirNo}")
#         os.system(f"python main.py --inference --model FlowNet2S --save_flow --inference_dataset ImagesFromFolder \
#         --inference_dataset_root /home/sawsn/tachysense-gray-50-6464/train/clean/{dirNo} --inference_visualize \
#         --resume /home/sawsn/FlowNet2S_model_best.pth.tar --save /home/sawsn/flownet-dlo/inference_results/clean/{dirNo} \
#         --skip_training --skip_validation")

