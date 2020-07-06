# -*- coding: utf-8 -*-


import dlib
import cv2
 
 
#源程序是用sys.argv从命令行参数去获取训练模型，精简版我直接把路径写在程序中了
predictor_path = "./data/data_dlib/shape_predictor_68_face_landmarks.dat"
 
faces_path = "./data/data_faces/face_1.jpeg"
#与人脸检测相同，使用dlib自带的frontal_face_detector作为人脸检测器
detector = dlib.get_frontal_face_detector()
 
#使用官方提供的模型构建特征提取器
predictor = dlib.shape_predictor(predictor_path)
#读取图片
img = cv2.imread(faces_path)
 
#与人脸检测程序相同,使用detector进行人脸检测 dets为返回的结果
dets = detector(img, 1)
#使用enumerate 函数遍历序列中的元素以及它们的下标
#下标k即为人脸序号
#left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
#top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
for k, d in enumerate(dets):
    print("dets{}".format(d))
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
    k, d.left(), d.top(), d.right(), d.bottom()))
 
    #使用predictor进行人脸关键点识别 shape为返回的结果
    shape = predictor(img, d)
    #获取第一个和第二个点的坐标（相对于图片而不是框出来的人脸）
    print("Part 0: {}, Part 1: {} ...".format(shape.part(0),  shape.part(1)))
 
    #绘制特征点
    for index, pt in enumerate(shape.parts()):
        print('Part {}: {}'.format(index, pt))
        pt_pos = (pt.x, pt.y)
        cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)
 
 
 
cv2.imshow('test2', img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()