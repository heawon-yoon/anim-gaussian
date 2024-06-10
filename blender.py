import bpy
import math
from math import radians
from mathutils import Matrix
from mathutils import Vector
import json

# 获取相机和物体
camera = bpy.context.scene.camera
camera.location = (0.0, 0.0, 5.0)
# 相机数据
cam_data = camera.data
# 物体中心点
object_center = (0.0, 0.0, 0.0)
object_center = Vector(object_center)
# 旋转的次数和角度（以弧度为单位）
num_rotations = 24
rotation_angle_degrees = -15
rotation_angle_radians = math.radians(rotation_angle_degrees)

output_path = '/Users/yoon/Desktop/heawon’s MacBook Pro/panda3d/humans/blender/'


# 对准物体
def look_at(obj, target):
    # 'obj'是我们要操作的对象，'target'是我们要让物体对准的目标
    direction = target - obj.location
    # 让物体的'-Z'轴对准目标，同时'Y'轴向上
    rot_quat = direction.to_track_quat('-Z', 'Z')
    # 应用旋转
    obj.rotation_euler = rot_quat.to_euler()


# 让相机对准物体
look_at(camera, object_center)


camera_datas = []

def render(img_nm):
    # 获取渲染尺寸
    scene = bpy.context.scene
    render = scene.render
    width = render.resolution_x
    height = render.resolution_y
    res_x = width
    res_y = height
    # 计算内参矩阵
    focal_length = cam_data.lens  # 相机焦距
    print('flcal_length : ',focal_length)
    sensor_width = cam_data.sensor_width  # 传感器宽度
    print('sensor_width',sensor_width)
    sensor_height = cam_data.sensor_height if cam_data.sensor_fit == 'VERTICAL' else sensor_width * height / width
    px = width / 2
    py = height / 2
    fx = width * focal_length / sensor_width
    fy = height * focal_length / sensor_height
    K = [
        [fx, 0, px],
        [0, fy, py],
        [0, 0, 1]
    ]

    # 计算sensor的宽度和高度
    if cam_data.sensor_fit == 'AUTO':
        if res_x > res_y:
            sensor_width = cam_data.sensor_width
            sensor_height = cam_data.sensor_width * res_y / res_x
        else:
            sensor_height = cam_data.sensor_height
            sensor_width = cam_data.sensor_height * res_x / res_y
    elif cam_data.sensor_fit == 'HORIZONTAL':
        sensor_width = cam_data.sensor_width
        sensor_height = cam_data.sensor_width * res_y / res_x
    else:  # VERTICAL
        sensor_height = cam_data.sensor_height
        sensor_width = cam_data.sensor_height * res_x / res_y

    # 计算fovx和fovy
    cam_lens = cam_data.lens
    fovx = 2 * math.atan(sensor_width / (2 * cam_lens)) * (180 / math.pi)  # 转换为度
    fovy = 2 * math.atan(sensor_height / (2 * cam_lens)) * (180 / math.pi)  # 转换为度

    world_to_camera_matrix = camera.matrix_world.inverted()
    print('world_to_camera_matrix',world_to_camera_matrix)
    # 获取外参矩阵（世界坐标系到相机坐标系的转换矩阵）
    c2w_matrix = camera.matrix_world
    print('c2w_matrix',c2w_matrix)
    c2w_list = []
    for row in c2w_matrix:
        r = [e for e in row]  # 将每一行中的元素转换为列表
        c2w_list.append(r)

    # 分解外参矩阵
    translation, rotation, scale = world_to_camera_matrix.decompose()
    # 转换为4x4矩阵
    rotation_matrix = rotation.to_matrix().to_4x4()
    translation_matrix = Matrix.Translation(translation)
    RT = translation_matrix @ rotation_matrix

    # 创建字典以保存相机参数
    camera_params = {
        'intrinsics': K,
        'extrinsics': {
            'c2w_matrix': c2w_list,
            'translation': translation[:],
            'rotation': rotation[:]
        },
        'width': width,
        'height': height,
        'fovx': fovx,
        'fovy': fovy,
        'img_id': img_nm
    }

    camera_datas.append(camera_params)



# 围绕Y轴旋转
for i in range(num_rotations):
    if i==0 :
        # 更新场景，以便更改可见
        bpy.context.view_layer.update()

        # 打印出新的相机位置
        print(f"After rotation {i + 1}: {camera.location}")
        render_frame = str(i).zfill(6)
        img_nm = f'image_{render_frame}'
        render(img_nm)
        bpy.context.scene.render.filepath = f'{output_path}image_{render_frame}'
        bpy.ops.render.render(write_still=True)  # 渲染并写入图片
        continue
    if i == 12 :
        rotation_angle_degrees = -31
        rotation_angle_radians = math.radians(rotation_angle_degrees)
    # 创建旋转矩阵
    rotation_matrix = Matrix.Rotation(rotation_angle_radians, 4, 'Y')

    # 计算相对于物体中心点的相机位置
    relative_camera_location = camera.location - object_center

    # 旋转相对位置
    rotated_relative_location = rotation_matrix @ relative_camera_location

    # 更新相机的位置
    camera.location = rotated_relative_location + object_center
    look_at(camera, object_center)
    # 更新场景，以便更改可见
    bpy.context.view_layer.update()

    # 打印出新的相机位置
    print(f"After rotation {i + 1}: {camera.location}")
    render_frame = str(i).zfill(6)
    img_nm = f'image_{render_frame}'
    render(img_nm)
    bpy.context.scene.render.filepath = f'{output_path}image_{render_frame}'
    bpy.ops.render.render(write_still=True)  # 渲染并写入图片

# 将字典转换为JSON字符串并保存到文件
with open('/Users/yoon/Desktop/heawon’s MacBook Pro/panda3d/cameras.json', 'w') as f:
    json.dump(camera_datas, f, indent=None)