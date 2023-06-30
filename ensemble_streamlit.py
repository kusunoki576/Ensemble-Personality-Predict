import os
import cv2
from numpy import expand_dims
from PIL import Image
from torchvision import models
import torch.nn as nn

import torch
import pandas
import seaborn

import json
import numpy as np

from torchvision import transforms

import streamlit as st


def get_model():
    # model = models.vgg16()
    # num_ftrs = model.classifier[6].in_features
    # model.classifier[6] = torch.nn.Linear(num_ftrs, 16)  # assuming your task has 2 classes
    # model.load_state_dict(torch.load('model_VGG16_Augment_1.pth'))

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 16)
    model_name = "model_best.pth"
    model.load_state_dict(torch.load(model_name))
    model.eval()

    return model


def get_json():
    # Load from a JSON file
    with open('mbti_types.json', 'r') as f:
        mbti_types = json.load(f)
    # mbti_types

    mbti_names_jp = {
        'ENFJ': '主人公: 他人のために行動でき情熱的で利他主義',
        'ENFP': '運動家: コミュニケーション能力が高くリーダー格',
        'ENTJ': '指揮官: 共通の目標に向けて人々をひとつにまとめる',
        'ENTP': '討論者: 頭の回転が早くおしゃべり',
        'ESFJ': '領事: 明るく活動的で情に厚く礼儀正しい',
        'ESFP': 'エンターテイナー: 人を楽しませるのが得意',
        'ESTJ': '幹部: 高いリーダーシップと意欲を持ち、組織を引っ張る',
        'ESTP': '起業家: 外交的で人との付き合いが得意',
        'INFJ': '提唱者: 誠実で責任感の強い理想主義者',
        'INFP': '仲介者: 周りの人を思いやる優しい性格',
        'INTJ': '建築家: 論理的な完璧主義者',
        'INTP': '論理学者: 難しい問題を解決するのが得意',
        'ISFJ': '擁護者: 優しく責任感も強いため、人の支えとなる',
        'ISFP': '冒険家: 創造力、エネルギー、自主性に溢れている',
        'ISTJ': '管理者: 冷静沈着で計略的、ルールに忠実',
        'ISTP': '巨匠: 好奇心旺盛で新しい情報を探索することが好き'
    }

    # Convert MBTI types to "TYPE / Japanese name" format
    for key in mbti_types:
        mbti_type = mbti_types[key]
        if mbti_type in mbti_names_jp:
            mbti_types[key] = f"{mbti_type} / {mbti_names_jp[mbti_type]}"

    return mbti_types


def clip_face(img: Image, model_yolov7):
    img_np = np.array(img)

    with torch.no_grad():
        results = model_yolov7([img_np])

    if len(results.xyxy[0]) == 0:
        print("顔が検出できない")
        return None

    # Convert tensor to CPU and then to numpy array
    results_np = results.xyxy[0].cpu().numpy()

    # Filter the results by confidence score
    results_np = results_np[results_np[:, 4] >= 0.25]

    if len(results_np) == 0:
        print("Confidence score 0.25以上の顔が検出できない")
        return None

        # Find the result with the highest confidence
    max_conf_index = np.argmax(results_np[:, 4])
    result = results_np[max_conf_index]

    xmin, ymin, xmax, ymax, conf, cls = result
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

    # Cut out the detected object from the original image
    cropped_img_np = img_np[ymin:ymax, xmin:xmax]
    # Convert numpy array back to PIL Image
    cropped_img = Image.fromarray(cropped_img_np)

    # cv2.imshow(cropped_img.resize((320, 320)))

    return cropped_img


def convert_image(image: Image) -> Image:
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # paste using alpha channel as mask
        image = background
    elif image.mode == "P":
        image = image.convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")
    else:
        # If the image is already an RGB image, return it as is
        pass
    return image


def predict_compose(path: str, model, model_yolov7):
    # 画像の処理1
    image_path = path
    image = Image.open(image_path)
    # image = image.convert("RGB")
    image = convert_image(image)
    image = clip_face(image, model_yolov7)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)  # ★
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # Move the input and model to GPU for speed if available
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to('cuda')
    #     model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class_index = torch.argmax(probabilities).item()

    return predicted_class_index


def ensemble_predict(path, model_yolov7, mbti_types):
    # 画像の処理
    image_path = path
    image = Image.open(image_path)
    image = convert_image(image)
    image = clip_face(image, model_yolov7)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        # input_batch = image.to('cpu')
        input_tensor = preprocess(image)  # ★
        input_batch = input_tensor.unsqueeze(0)

        # 各モデルの予測結果を取得
        model_resnet50_best = models.resnet50(pretrained=False)
        model_resnet50_best.fc = nn.Linear(2048, 16)
        model_name = "model_best.pth"
        model_resnet50_best.load_state_dict(torch.load(model_name))
        model_resnet50_best.eval()
        model_resnet50_best.to("cpu")
        output_best = model_resnet50_best(input_batch)

        model_vgg16 = models.vgg16(pretrained=False)
        num_ftrs = model_vgg16.classifier[6].in_features
        model_vgg16.classifier[6] = torch.nn.Linear(num_ftrs, 16)
        model_vgg16.load_state_dict(torch.load('model_VGG16_Augment_1.pth'))
        model_vgg16.to("cpu")
        output_vgg16 = model_vgg16(input_batch)

        model_resnet50_1 = models.resnet50(pretrained=False)
        model_resnet50_1.fc = nn.Linear(2048, 16)
        model_name = "model_ResNet_1.pth"
        model_resnet50_1.load_state_dict(torch.load(model_name))
        model_resnet50_1.eval()
        model_resnet50_1.to("cpu")
        output_resnet50_1 = model_resnet50_1(input_batch)

        output = (output_best * 5.5 + output_vgg16 + output_resnet50_1) / 2

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_index = torch.argmax(probabilities).item()

        max_value, max_index = torch.max(probabilities, dim=0)
        values, indices = torch.sort(probabilities, dim=0, descending=True)
        # print(f"1st predict: idx: type: {mbti_types[str(max_index.item())]}, {max_index.item()}, value: {max_value}")
        # print(f"2nd predict: idx: type: {mbti_types[str(indices[1].item())]}, {indices[1].item()}, second: {values[1]}")

        st.write(f"1st predict: type: {max_index.item()} {mbti_types[str(max_index.item())]}, score: {max_value}")
        st.write(f"2nd predict: type: {indices[1].item()}{mbti_types[str(indices[1].item())]}, score: {values[1]}")

        return predicted_class_index


def main():
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    model_yolov7 = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model='best.pt')

    model = get_model()

    types = get_json()
    # print(types)

    # path = "../test_images/ele.png"

    uploaded_file = st.file_uploader("Choose a image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        test_image = Image.open(uploaded_file)
        st.image(test_image, caption='Uploaded Image.')

        path = uploaded_file

        model_yolov7.to("cpu")
        model.to("cpu")
        predicted_class_index = ensemble_predict(path, model_yolov7, types)
        predicted_mbti_type = types[str(predicted_class_index)]
        # print("predicted MBTI:", predicted_mbti_type)
        st.write("predicted MBTI:", predicted_mbti_type)


if __name__ == "__main__":
    main()
