import json

with open('/home/zzx/Experiment/Data/UTDAC2020/annotations/instances_val2017.json', 'r') as f:
    coco_data = json.load(f)

# 获取标签信息
images = coco_data['images']
print('图片数量：', len(images))

with open('img_id.txt', 'w') as f:
    for image in images:
        f.write(f"{image['id']} {image['file_name']}\n")