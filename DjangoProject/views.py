import base64
from PIL import Image
from django.shortcuts import render

from .forms import ImageUploadForm
import numpy as np
from django.views.decorators.http import require_GET
from matplotlib import pyplot as plt
from segmentation_models_pytorch.encoders.timm_resnest import model_name
from torch.cuda import device
from torchvision import transforms
from PIL import ImageEnhance
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

import io
import sys
import torch
sys.path.append('E:/1555bishe/project')
import os
os.chdir('E:/1555bishe/project')
from U_Net_train import config as UNetconfig
from FPN_train import config as FPNconfig
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@csrf_exempt
def model_Pred(input_image, model_name):
    input_image = Image.open(input_image).convert("L")

    if 'UNet' in model_name:
        config = UNetconfig
    elif 'FPN' in model_name:
        config = FPNconfig

    input_image = config.transform(input_image)  # 应用预处理转换
    input_image = input_image.unsqueeze(0)  # 添加批次维度

    model = config.backbone.to(device)

    # 加载模型权重
    model.load_state_dict(torch.load(f'E:/1555bishe/project/model/{model_name}.pth', map_location=device))
    model.eval()  # 设置模型为评估模式
    input_image = input_image.to(device)

    with torch.no_grad():
        output = model(input_image)
        output = torch.sigmoid(output)  # 应用sigmoid激活函数
        output_thresholded = (output > 0.5).float()  # 阈值处理，得到二进制掩膜

    output_image = output_thresholded[0].cpu().numpy().squeeze()

    return output_image


@csrf_exempt
def predict_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = request.FILES['image']
            original_image_rgba = Image.open(uploaded_image).convert('RGBA')

            model_name = request.POST.get('model_name')
            masked_image = model_Pred(uploaded_image, model_name)

            # 创建掩膜图像并转为灰度
            mask_image = Image.fromarray((masked_image * 255).astype('uint8')).convert("L")  # 转为灰度掩膜

            # 将掩膜图像转换为 RGBA
            mask_image_rgba = Image.new("RGBA", mask_image.size)
            for x in range(mask_image.width):
                for y in range(mask_image.height):
                    pixel_value = mask_image.getpixel((x, y))
                    if pixel_value > 0:  # 白色区域
                        mask_image_rgba.putpixel((x, y), (0, 255, 0, 80))  # 半透明绿色
                    else:  # 黑色区域
                        mask_image_rgba.putpixel((x, y), (0, 0, 0, 0))  # 透明

            # 确保掩膜图像和原始图像大小一致
            mask_image_rgba = mask_image_rgba.resize(original_image_rgba.size, Image.LANCZOS)

            # 叠加掩膜到原始图像
            final_image = Image.alpha_composite(original_image_rgba, mask_image_rgba)

            # 保存结果图像为 PNG 格式并转换为 Base64
            response_image_io = io.BytesIO()
            final_image.save(response_image_io, format='PNG')  # 使用 PNG 格式
            response_image_io.seek(0)
            img_base64 = base64.b64encode(response_image_io.getvalue()).decode('utf-8')

            return JsonResponse({'image': f'data:image/png;base64,{img_base64}'})

    return JsonResponse({'error': 'Invalid request'}, status=400)
from django.views.decorators.http import require_GET

# 假设 MODEL_DIR 是模型文件存放目录
MODEL_DIR = 'E:/1555bishe/project/model'

@require_GET
def get_model_names(request):
    # 获取所有以 .pth 结尾的模型文件名，并去掉后缀
    model_names = [f[:-4] for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
    return JsonResponse({'model_names': model_names})
def index(request):
    return render(request, 'index.html')