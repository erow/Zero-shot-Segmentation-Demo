# Zero-shot Segmentation Demo

该项目包含 zero-shot segmentation的两种用例。并且提供本地运行版本，以及在线版本。

## Segmentation

## Instance Retrieval


# 本地版本

```python demos/retrive.py   --pretrained_weights checkpoint.pth  --k_img imgs/sample5.JPEG --q_img imgs/mix.jpg```

```python main.py --img imgs/mix.jpg  --pretrained_weights checkpoint.pth```

# 在线应用
# Image Matching API
## Overview
请注意，这个示例仅用于说明API接口的基本结构。实际实现时，您需要根据您的需求和技术选择来完成图像匹配和其他相关功能。

Person: 
- 请设计详细的API接口文档并用markdown格式回答

ChatGPT: 
- 以下是使用Markdown格式编写的详细API接口文档：
# Image Matching API

本API提供图像匹配服务，允许用户提交两张图片（"key" 和 "query"），并根据用户在"query"图像上选择的区域，返回匹配的物体。
## 1. Submit

**Endpoint** : `/submit`

**Method** : `POST`

**Description** : 用户在前端上传两张图片（"key" 和 "query"），服务器收到后将图片保存在云端并生成唯一的image id。返回image id。
### Request 
- **Content-Type** : `multipart/form-data`ParameterTypeDescriptionkey_imageFile用户上传的"key"图片query_imageFile用户上传的"query"图片
### Response 
- **Content-Type** : `application/json`ParameterTypeDescriptionkey_image_idString"key"图片的唯一IDquery_image_idString"query"图片的唯一ID
#### Example

**Request** 

```http

POST /submit HTTP/1.1
Content-Type: multipart/form-data; boundary=----WebKitFormBoundaryyrV7KO0GIxCAQejs

------WebKitFormBoundaryyrV7KO0GIxCAQejs
Content-Disposition: form-data; name="key_image"; filename="key_image.jpg"
Content-Type: image/jpeg

(binary content)
------WebKitFormBoundaryyrV7KO0GIxCAQejs
Content-Disposition: form-data; name="query_image"; filename="query_image.jpg"
Content-Type: image/jpeg

(binary content)
------WebKitFormBoundaryyrV7KO0GIxCAQejs--
```



**Response** 

```json

{
  "key_image_id": "550e8400-e29b-41d4-a716-446655440000",
  "query_image_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
}
```

## 3. Get Image

**Endpoint** : `/get_image`

**Method** : `GET`

**Description** : 前端根据image id获取对应的图片。
### RequestParameterTypeDescriptionimage_idString图片的唯一ID
### Response 
- **Content-Type** : `image/jpeg`
#### Example

**Request** 

```http

GET /get_image?image_id=550e8400-e29b-41d4-a716-446655440000 HTTP/1.1
```



**Response** 

```http

HTTP/1.1 200 OK
Content-Type: image/jpeg

(binary content)
```

## 2. match

**Endpoint** : `/match`

**Method** : `POST`

**Description** : 用户选择"query"图像中的某个区域后，触发查询请求。前端发送"key" 和 "query"的image id，以及点击图片的坐标位置给服务器。服务器返回一个列表包含所有匹配的图片。
### Request 
- **Content-Type** : `application/json`ParameterTypeDescriptionkey_image_idString"key"图片的唯一IDquery_image_idString"query"图片的唯一IDcoordsObject用户点击的坐标位置，包含"x"和"y"属性
### Response 
- **Content-Type** : `application/json`ParameterTypeDescriptionimage_idString匹配图片的唯一IDurlString匹配图片的URL
#### Example

**Request** 

```http

POST /select HTTP/1.1
Content-Type: application/json

{
  "key_image_id": "550e8400-e29b-41d4-a716-446655440000",
  "query_image_id": "6ba7b810-9
}
```

## 2. segment

**Endpoint** : `/segment`

**Method** : `POST`

**Description** : 用户选择"query"图像中的某个区域后，触发查询请求。前端发送"query"的image id，以及点击图片的坐标位置给服务器。服务器返回一个列表包含所有匹配的图片。
### Request 
- **Content-Type** : `application/json`ParameterTypeDescriptionkey_image_idString"key"图片的唯一IDquery_image_idString"query"图片的唯一IDcoordsObject用户点击的坐标位置，包含"x"和"y"属性
### Response 
- **Content-Type** : `application/json`ParameterTypeDescriptionimage_idString匹配图片的唯一IDurlString匹配图片的URL
#### Example

**Request** 

```http

POST /select HTTP/1.1
Content-Type: application/json

{
  "key_image_id": "550e8400-e29b-41d4-a716-446655440000",
  "query_image_id": "6ba7b810-9
}
```