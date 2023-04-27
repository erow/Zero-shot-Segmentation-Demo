from flask import Flask, request, jsonify, send_from_directory
import uuid
import os

app = Flask(__name__)

# 文件上传路径
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def save_image(image, image_id):
    image_folder = os.path.join(UPLOAD_FOLDER, image_id)
    os.makedirs(image_folder)
    image.save(os.path.join(image_folder, 'origin.png'))

# submit 接口
@app.route('/submit', methods=['POST'])
def submit():
    key_image = request.files['key_image']
    query_image = request.files['query_image']
    
    key_image_id = str(uuid.uuid4())
    query_image_id = str(uuid.uuid4())
    
    save_image(key_image, key_image_id)
    save_image(query_image, query_image_id)
    
    return jsonify({'key_image_id': key_image_id, 'query_image_id': query_image_id})

# get_image 接口
@app.route('/get_image', methods=['GET'])
def get_image():
    image_id = request.args.get('image_id')
    return send_from_directory(os.path.join(UPLOAD_FOLDER, image_id), 'origin.png')

# select 接口
@app.route('/select', methods=['POST'])
def select():
    data = request.get_json()
    
    key_image_id = data['key_image_id']
    query_image_id = data['query_image_id']
    coords = data['coords']  # 用户点击的坐标位置，如：{'x': 100, 'y': 200}

    # TODO: 实现图像匹配逻辑
    
    # 示例返回数据
    result = [
        {'image_id': 'matched_image_id_1', 'url': 'https://example.com/matched_image_1.jpg'},
        {'image_id': 'matched_image_id_2', 'url': 'https://example.com/matched_image_2.jpg'}
    ]
    
    return jsonify(result)



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
