import base64

from PIL import Image, ImageOps
from torch.cuda import device
from django.views.decorators.csrf import csrf_exempt
import io
import sys
import torch
sys.path.append('E:/1555bishe/project')
from DjangoProject.models import User,Record
import os
os.chdir('E:/1555bishe/project')
from Input_and_Classify import *
from U_Net_train import config as UNetConfig
from DeepLab_Train import config as DeepLabConfig
from Linknet_Train import config as LinknetConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import openai
openai.api_base = "https://api.chatanywhere.tech/v1"  # 替换为专属 API URL（网页6）
openai.api_key = "sk-LXZqKYRCJH8kRe1s1ghDFM1E20ipbIadPKyyM8FSwOTtp7Wi"  # 从 GitHub 获取的 Key


tumor_type=""
confidence=0.0
ai_answer=""


def pad_image(image,fill=0):
    """
    将图片填充到目标尺寸（正方形），确保宽高能够严格缩放到32的倍数
    """
    # 若原图尺寸大于目标尺寸，先居中裁剪,目标尺寸为宽高的最大值
    target=max( image.width, image.height) #确保填充后形状为正方形，填充后的正方形的边长是之前最长的那条边
    # 计算填充尺寸（确保宽高严格等于目标尺寸）
    delta_width = target - image.width
    delta_height = target - image.height
    padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
    # 填充图片
    padded_image = ImageOps.expand(image, padding, fill=fill)
    return padded_image


@csrf_exempt
def model_Pred(input_image, model_name):

    if 'unet' in model_name:
        config = UNetConfig  # 引用UNet的配置
    elif 'deeplab' in model_name:
        config = DeepLabConfig  # DeepLabV3+的配置
    elif 'linknet' in model_name:
        config = LinknetConfig

    input_image = config.transform(input_image)  # 应用预处理转换
    input_image = input_image.unsqueeze(0)  # 添加批次维度

    model = config.backbone.to(device)

    # 加载模型权重
    model.load_state_dict(torch.load(f'E:/1555bishe/project/model/{model_name}.pth', map_location=device),strict=False)
    model.eval()  # 设置模型为评估模式

    input_image = input_image.to(device)

    with torch.no_grad():
        output = model(input_image)
        output = torch.sigmoid(output)  # 应用sigmoid激活函数
        output_thresholded = (output > 0.5).float()  # 阈值处理，得到二进制掩膜

    output_image = output_thresholded[0].cpu().numpy().squeeze()

    return output_image


@csrf_exempt
def infer_location(request):#推断肿瘤位置
     '''有肿瘤，需要检测肿瘤位置，需要询问ai该类型肿瘤的相关介绍和治疗建议'''

     input_image = request.FILES['image']
     input_image = Image.open(input_image).convert("L")
     if input_image.width !=input_image.height :#要求输入为正方形，防止缩放后某一边长度不能被32整除
            input_image = pad_image(input_image)#填充为正方形
     original_image_rgba = input_image.convert('RGBA')

     model_name = request.POST['model_name']
     masked_image = model_Pred(input_image, model_name)
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

     response_image_io.seek(0)# 意思是把文件指针移动到开头（第 0 个字节），否则后面读出来是空的。
     img_base64 = base64.b64encode(response_image_io.getvalue()).decode('utf-8')
     '''
        response_image_io.getvalue() 会把整个图片内容（字节数据）取出来。
        base64.b64encode(...) 把图片内容转成 Base64 字节串。
        .decode('utf-8') 把字节串转成字符串，方便用于 JSON、HTML 或文本传输。
        '''
     return JsonResponse(
         {'image': f'data:image/png;base64,{img_base64}',

          } )


from django.utils.timezone import now




@csrf_exempt
def infer_type(request):#该接口是推断肿瘤类型
     input_image = request.FILES['image']
     '''分类'''
     class_names = ["胶质瘤", "脑膜瘤", "无肿瘤", "垂体瘤"]  # 获取类别名称（需与训练数据集顺序一致）

     # 执行推理,调用Input_and_Classify.py中的inference函数
     result = inference(input_image, class_names)
     username = request.session.get('username')
     tumor_type = result['class_name']
     confidence = result['confidence']

     with connection.cursor() as cursor:
         cursor.execute("INSERT INTO record (username, tumor_name, confidence, created_at) VALUES (%s, %s, %s, %s)",
                        [username, tumor_type, confidence, now()])
     return JsonResponse(
         {
     'confidence': confidence,
     'tumor_type': f'{tumor_type}',

     }
     )


@csrf_exempt
def infer_ai(request):   #该接口返回ai的回答
    tumor_type = request.POST['tumor_type']
    model_type = request.POST['model_type']
    response = openai.ChatCompletion.create(  # 每天100次免费额度
        model=model_type,
        messages=[{"role": "user", "content": f"介绍{tumor_type}，并给出治疗建议和治疗方案，推荐几个中国国内的医院治疗该疾病"}]
    )
    ai_answer = response.choices[0].message['content']
    return JsonResponse(
        {
            'ai_answer': ai_answer,
        }
    )
from django.shortcuts import render
from django.http import JsonResponse
from django.db import connection



@csrf_exempt
def auth_view(request):
    """
    处理登录和注册（POST）：
      - action: 'login' 或 'register'
      - username, password: 字符串
      - age: 可选，字符串形式数字
      - gender: 可选，'M'/'F'/'O'
    返回 JSON: {'status':'success'|'error', 'message'..., 'redirect':...}
    """


    action   = request.POST.get('action')
    username = request.POST.get('username')
    password = request.POST.get('password')
    age      = request.POST.get('age')
    gender   = request.POST.get('gender')

    # ---------- 登录逻辑 ----------
    if action == 'login':
        try:
            user = User.objects.get(username=username)  # 精确查询 :contentReference[oaicite:7]{index=7}
        except User.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': '用户名或密码错误'})

        if user.password == password:
            # 简易会话：将用户名写入 session :contentReference[oaicite:8]{index=8}
            request.session['username'] = username
            redirect_url = '/index/' if user.status == 0 else '/statistics/'
            return JsonResponse({'status': 'success', 'redirect': redirect_url})
        else:
            return JsonResponse({'status': 'error', 'message': '用户名或密码错误'})

    # ---------- 注册逻辑 ----------
    elif action == 'register':
        # 检查用户名是否已存在
        if User.objects.filter(username=username).exists():  # exists() 快速判断 :contentReference[oaicite:9]{index=9}
            return JsonResponse({'status': 'error', 'message': '用户名已被占用'})

        # 创建新用户实例并保存
        user = User(
            username=username,
            password=password,
            status=0,
            age=int(age) if age else None,
            gender=gender if gender in dict(User.GENDER_CHOICES) else None
        )
        user.save()  # 调用 Model.save() 持久化 :contentReference[oaicite:10]{index=10}

        return JsonResponse({'status': 'success', 'message': '注册成功，请登录'})

    # 无效 action
    else:
        return JsonResponse({'status': 'error', 'message': '未知的操作类型'})


@csrf_exempt
def get_history_view(request):
    username = request.session.get('username')
    if not username:
        return JsonResponse({'error': '未登录'}, status=401)

    with connection.cursor() as cursor:
        cursor.execute("SELECT tumor_name, confidence, created_at FROM record WHERE username = %s ORDER BY created_at DESC", [username])
        rows = cursor.fetchall()
        records = [{
            'tumor_name': r[0],
            'confidence': f"{float(r[1]) * 100:.2f}%",
            'created_at': str(r[2]).split('.')[0].split('+')[0]  # 去掉 .微秒 和 +时区
        } for r in rows]

        return JsonResponse({'records': records})

'''
指定时间段的四类肿瘤比例统计
'''



from django.db.models import Max, Count


@csrf_exempt
def stats_api(request):
    """
    统计普通用户首条（最新）记录中的肿瘤类型分布，
    且可按日期、年龄段、性别筛选：

      - age_min:  可选，最小年龄（含）
      - age_max:  可选，最大年龄（含）
      - gender:   可选，性别 ('M','F')
    返回 JSON 列表：[{'tumor_name': ..., 'count': ...}, ...]
    """
    # 获取筛选参数

    age_min = request.GET.get('age_min')
    age_max = request.GET.get('age_max')
    gender = request.GET.get('gender')

    # 初始化过滤条件
    filters = {}

    # 普通用户
    filters['status'] = 0

    if age_min:
        filters['age__gte'] = age_min
    if age_max:
        filters['age__lte'] = age_max
    if gender:
        filters['gender'] = gender

    # 获取符合条件的用户
    users = User.objects.filter(**filters)

    # 获取每个用户最新的记录
    latest_records = (
        Record.objects
        .filter(username__in=users.values('username'))  # 限定为符合条件的用户
        .values('username')
        .annotate(latest_time=Max('created_at'))  # 获取每个用户最新的记录
    )

    # 获取这些记录的肿瘤类型并进行统计
    tumor_count = (
        Record.objects
        .filter(username__in=[record['username'] for record in latest_records])
        .values('tumor_name')
        .annotate(count=Count('tumor_name'))
        .values('tumor_name', 'count')
    )

    # 转换为响应数据格式
    data = list(tumor_count)
    return JsonResponse(data, safe=False)  # 返回 JSON 数据
'''
指定时间段内各个月有多少人得肿瘤
'''
@csrf_exempt
def monthly_stats_api(request):
    start = request.GET.get('start', '2024-06-01')  # 获取请求中的起始日期
    end = request.GET.get('end', '2025-05-31')      # 获取请求中的结束日期

    # SQL 查询：获取每个用户在每月的最新记录
    sql = """
    SELECT month, tumor_name, COUNT(*) AS tumor_count
    FROM (
        SELECT strftime('%%Y-%%m', created_at) AS month, username, tumor_name, created_at,
               ROW_NUMBER() OVER (
                   PARTITION BY username, strftime('%%Y-%%m', created_at) ORDER BY created_at DESC
               ) AS rn
        FROM record
        WHERE created_at BETWEEN '%(start)s' AND '%(end)s'
    )
    WHERE rn = 1
    GROUP BY month, tumor_name
    ORDER BY month, tumor_name;
    """ % {'start': start, 'end': end}

    # 执行查询并获取结果
    with connection.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()

    # 定义肿瘤类型
    tumor_types = ['脑膜瘤', '胶质瘤', '血管瘤', '无肿瘤']

    # 构建按月统计的空字典
    month_data = {}

    # 生成所有月份（从开始到结束日期）
    from datetime import datetime
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')

    current_date = start_date
    while current_date <= end_date:
        month_str = current_date.strftime('%Y-%m')
        month_data[month_str] = {tumor: 0 for tumor in tumor_types}  # 初始化每个月的肿瘤类型计数
        current_date = datetime(current_date.year, current_date.month + 1, 1) if current_date.month < 12 else datetime(current_date.year + 1, 1, 1)

    # 处理查询结果，按月和肿瘤类型填充数据
    for row in rows:
        month, tumor_name, count = row
        if month in month_data:
            month_data[month][tumor_name] = count

    # 转换为返回的数据结构
    data = []
    for month, tumor_counts in month_data.items():
        data.append({'month': month, **tumor_counts})

    return JsonResponse(data, safe=False)




def login(request):
    return render(request, 'login.html')

def get_username_view(request):
    username = request.session.get('username')
    if username:
        return JsonResponse({'username': username})
    else:
        return JsonResponse({'username': None}, status=401)
def index(request):
    return render(request, 'index.html')

def statistics(request):
    return render(request, 'statistics.html')