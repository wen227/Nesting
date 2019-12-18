# -*- coding: utf-8 -*-
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.path import Path
from operator import itemgetter
from shapely.geometry import asPolygon


def inpolygon(xq, yq, xv, yv):
    """
    判断点是否在多边形内
    :type xq: np.ndarray
    :type yq: np.ndarray
    :type xv: np.ndarray
    :type yv: np.ndarray
    """
    # 合并xv和yv为顶点数组
    vertices = np.vstack((xv, yv)).T
    # 定义Path对象
    path = Path(vertices)
    # 把xq和yq合并为test_points
    test_points = np.hstack([xq.reshape(xq.size, -1), yq.reshape(yq.size, -1)])
    # 得到一个test_points是否严格在path内的mask，是bool值数组
    _in = path.contains_points(test_points)
    # 得到一个test_points是否在path内部或者在路径上的mask
    _in_on = path.contains_points(test_points, radius=-1e-10)
    # 得到一个test_points是否在path路径上的mask
    _on = _in ^ _in_on
    return _in_on


def get_boxes_data(path):
    """
    获取零件数据，用字典存储
    :return:
    """
    # 读取csv至字典
    csv_file = open(path, "r", encoding='utf-8')
    reader = csv.reader(csv_file)
    # 建立空字典
    data = {}
    boxes = []
    for item in reader:
        # 忽略csv文件的第一行
        if reader.line_num == 1:
            continue
            # 字典data存储外轮廓数据
        data[item[1]] = item[3]
        # eval()函数用来执行一个字符串表达式
        pos_list = eval(item[3])  # item[3]为外轮廓数据
        pos_np = np.array(pos_list)  # 转为np_array类型
        x_np = pos_np[:, 0]  # 切片 取出列
        y_np = pos_np[:, 1]
        # 生成该零件的外接矩形
        x1 = min(x_np)
        y1 = min(y_np)
        x2 = max(x_np)
        y2 = max(y_np)
        bl_point = np.array([x1, y1])  # 矩形左下角坐标
        # 找出多边形最左侧的顶点坐标
        y_ = []
        for id_ in range(len(x_np)):
            if x_np[id_] == x1:
                y_.append(y_np[id_])
        y_left = np.array([min(y_), max(y_)])  # 多边形最左侧顶点的y轴坐标
        h = y2 - y1 + 5
        w = x2 - x1 + 5
        s = h * w
        # boxes存储矩形数据
        boxes.append({'id': item[1], 'x1': x1, 'y1': y1, 'h': h, 'w': w, 'space': s, 'data': pos_list,
                      'bl_point': bl_point, 'x_list': x_np, 'y_list': y_np, 'isuse': False, 'y_left': y_left,
                      'batch': item[0], 'cloth': item[5]})
    csv_file.close()
    return boxes


def PolyArea(x, y):
    """返回多边形面积

    """
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def tranfer_(x_np, y_np):
    """将x_np,y_np两个ndarray，转换成（x,y) ndarray

    """
    tmp_li = []
    for k in range(len(x_np)):
        tmp_li.append([x_np[k], y_np[k]])
    xy_np = np.array(tmp_li)
    return xy_np


def get_rate(path, mianliao_h):
    # 加载零件数据
    boxes = get_boxes_data(path)
    # print(PolyArea(boxes[0]['x_list'], boxes[0]['y_list']))
    # print(boxes[0]['space'])
    # 按宽度对boxes进行排序
    boxes = sorted(boxes, key=itemgetter('w'), reverse=True)
    # 零件个数
    boxes_len = len(boxes)
    # 初始化
    move_list = []
    move_xy = np.array([0.0, 0.0])
    flag = 0
    for i in range(boxes_len):
        move_list.append(move_xy)
    cur_point = np.array([0.0, 0.0])
    init_point = np.array([0.0, 0.0])
    # 排列零件
    for i in range(boxes_len):
        if boxes[i]['isuse']:
            continue
        cur_point = init_point
        available_h = mianliao_h
        last_up = 0.0  # 用于计算两零件之间的高度间距(存储上一次y轴坐标）
        last_box = {'x_list': [cur_point[0], 20000.0], 'y_list': [cur_point[1], cur_point[1]], 'id': 0}
        for j in range(i, boxes_len):
            move = boxes[j]['bl_point'] - cur_point
            # rec_width = boxes[j]['w']
            # rec_height = boxes[j]['h']
            # x_min = cur_point[0]
            # y_min = cur_point[1]
            # x_list = np.array([x_min, x_min, x_min + rec_width, x_min + rec_width, x_min])  # 第j个矩形的所有轮廓点的x坐标集合
            # y_list = np.array([y_min, y_min + rec_height, y_min + rec_height, y_min, y_min])
            # message = {'cur_point': cur_point, 'move': move, 'rec_x': x_list, 'rec_y': y_list}
            if boxes[j]['h'] <= available_h and boxes[j]['isuse'] is False:
                # plt.plot(message['rec_x'], message['rec_y'], linewidth=0.5)  # 这一行是用来显示矩形边缘的
                boxes[j]['isuse'] = True
                move_list[j] = move

                # 存储移动后的多边形顶点坐标
                for k in range(len(boxes[j]['x_list'])):
                    boxes[j]['x_list'][k] = boxes[j]['x_list'][k] - move_list[j][0]
                    boxes[j]['y_list'][k] = boxes[j]['y_list'][k] - move_list[j][1]
                # 绘制多边形
                plt.plot(boxes[j]['x_list'], boxes[j]['y_list'], linewidth=0.3)

                # 计算gap
                gap_h = boxes[j]['y_left'][0] - move[1] - last_up
                last_up = boxes[j]['y_left'][1] - move[1]
                # 搜索放置在gap间的小零件
                for small in range(j + 1, boxes_len):
                    if boxes[small]['h'] <= gap_h \
                            and boxes[small]['h'] <= available_h \
                            and boxes[small]['isuse'] is False:
                        # 零件small的外接矩形的4个顶点和中心点
                        move_s = boxes[small]['bl_point'] - cur_point
                        tmp_x2 = boxes[small]['x1'] + boxes[small]['w'] - move_s[0]
                        tmp_y2 = boxes[small]['y1'] + boxes[small]['h'] - move_s[1]
                        tmp_x1 = boxes[small]['x1'] - move_s[0]
                        tmp_y1 = boxes[small]['y1'] - move_s[1]
                        tmp_mx = boxes[small]['x1'] + boxes[small]['w'] / 2 - move_s[0]
                        tmp_my = boxes[small]['y1'] + boxes[small]['h'] / 2 - move_s[1]
                        tmp_s = []  # 移动后的零件small所有顶点坐标
                        # 判断4个顶点和中心点是否在已排列零件内部
                        if inpolygon(tmp_x2, tmp_y2, boxes[j]['x_list'], boxes[j]['y_list'])[0] + 0 == 0 \
                                and inpolygon(tmp_x1, tmp_y2, boxes[j]['x_list'], boxes[j]['y_list'])[0] + 0 == 0 \
                                and inpolygon(tmp_x1, tmp_y1, boxes[j]['x_list'], boxes[j]['y_list'])[0] + 0 == 0 \
                                and inpolygon(tmp_x2, tmp_y1, boxes[j]['x_list'], boxes[j]['y_list'])[0] + 0 == 0 \
                                and inpolygon(tmp_mx, tmp_my, boxes[j]['x_list'], boxes[j]['y_list'])[0] + 0 == 0:
                            # 计算移动后的多边形顶点坐标
                            for k in range(len(boxes[small]['x_list'])):
                                tmp_s.append(
                                    [boxes[small]['x_list'][k] - move_s[0], boxes[small]['y_list'][k] - move_s[1]])
                            tmp_pols = np.array(tmp_s)
                            # 判断small零件与已排放的boxes[j]是否重叠
                            tmp_polj = tranfer_(boxes[j]['x_list'], boxes[j]['y_list'])
                            pol_s = asPolygon(tmp_pols)
                            pol_j = asPolygon(tmp_polj)
                            if pol_s.disjoint(pol_j):
                                boxes[small]['isuse'] = True
                                move_list[small] = move_s
                                # 存储移动后的多边形顶点坐标
                                for k in range(len(boxes[small]['x_list'])):
                                    boxes[small]['x_list'][k] = boxes[small]['x_list'][k] - move_list[small][0]
                                    boxes[small]['y_list'][k] = boxes[small]['y_list'][k] - move_list[small][1]
                                # 绘制多边形
                                plt.plot(boxes[small]['x_list'], boxes[small]['y_list'], linewidth=0.3)
                                break

                available_h = available_h - boxes[j]['h']
                # 零件排列到布料顶端的情况
                if boxes[j]['h'] > available_h and flag == 0:
                    flag = 1  # 标记用于只需判断1次布料顶端能否排列下小零件（small)
                    # 计算gap
                    gap_h = mianliao_h - (boxes[j]['y_left'][1] - move[1])
                    # print(gap_h)
                    # 搜索放置在gap间的小零件
                    for small in range(j + 1, boxes_len):
                        if boxes[small]['h'] <= gap_h \
                                and boxes[small]['isuse'] is False:
                            # 零件small的外接矩形的4个顶点和中心点
                            move_s = boxes[small]['bl_point'] + np.array([0.0, boxes[small]['h'] - 5]) - np.array(
                                [cur_point[0], mianliao_h])
                            tmp_x2 = boxes[small]['x1'] + boxes[small]['w'] - move_s[0]
                            tmp_y2 = boxes[small]['y1'] + boxes[small]['h'] - move_s[1]
                            tmp_x1 = boxes[small]['x1'] - move_s[0]
                            tmp_y1 = boxes[small]['y1'] - move_s[1]
                            tmp_mx = boxes[small]['x1'] + boxes[small]['w'] / 2 - move_s[0]
                            tmp_my = boxes[small]['y1'] + boxes[small]['h'] / 2 - move_s[1]
                            # 判断4个顶点和中心点是否在已排列零件内部
                            tmp_s = []
                            if inpolygon(tmp_x2, tmp_y2, boxes[j]['x_list'], boxes[j]['y_list'])[0] + 0 == 0 \
                                    and inpolygon(tmp_x1, tmp_y2, boxes[j]['x_list'], boxes[j]['y_list'])[0] + 0 == 0 \
                                    and inpolygon(tmp_x1, tmp_y1, boxes[j]['x_list'], boxes[j]['y_list'])[0] + 0 == 0 \
                                    and inpolygon(tmp_x2, tmp_y1, boxes[j]['x_list'], boxes[j]['y_list'])[0] + 0 == 0 \
                                    and inpolygon(tmp_mx, tmp_my, boxes[j]['x_list'], boxes[j]['y_list'])[0] + 0 == 0:
                                # 计算移动后的多边形顶点坐标
                                for k in range(len(boxes[small]['x_list'])):
                                    tmp_s.append(
                                        [boxes[small]['x_list'][k] - move_s[0], boxes[small]['y_list'][k] - move_s[1]])
                                tmp_pols = np.array(tmp_s)
                                # 判断small零件与已排放的boxes[j]是否重叠
                                tmp_polj = tranfer_(boxes[j]['x_list'], boxes[j]['y_list'])
                                pol_s = asPolygon(tmp_pols)
                                pol_j = asPolygon(tmp_polj)
                                if pol_s.disjoint(pol_j):
                                    boxes[small]['isuse'] = True
                                    move_list[small] = move_s
                                    # 存储移动后的多边形顶点坐标
                                    for k in range(len(boxes[small]['x_list'])):
                                        boxes[small]['x_list'][k] = boxes[small]['x_list'][k] - move_list[small][0]
                                        boxes[small]['y_list'][k] = boxes[small]['y_list'][k] - move_list[small][1]
                                    # 绘制多边形
                                    plt.plot(boxes[small]['x_list'], boxes[small]['y_list'], linewidth=0.3)

                                    break
                    break
                last_box = boxes[j]
                cur_point = cur_point + np.array([0.0, boxes[j]['h']])
        init_point = init_point + np.array([boxes[i]['w'], 0.0])
        flag = 0
    # 计算布料利用率
    area_box = 0
    width = init_point[0]
    for box in boxes:
        area_box += PolyArea(box['x_list'], box['y_list'])
    rate = area_box/(width*mianliao_h)
    # 保存排样图片
    path_name = path.split("\\")[-1].split('.')[0]
    path_plot = path_name + '.png'
    plt.title(r'%s——%0.4f' % (path_name, rate))
    plt.savefig(path_plot, dpi=600)
    # plt.ylim((0, 1600))
    plt.show()

    # 保存csv文件
    path_csv = path_name + '.csv'
    headers = ['下料批次号', '零件号', '面料号', '零件外轮廓线坐标']
    values = []

    for box in boxes:
        values.append([box['batch'],
                       box['id'],
                       box['cloth'],
                       str(tranfer_(box['x_list'].round(1), box['y_list'].round(1)).tolist())])
    csv_file = open(path_csv, "w", encoding='utf-8', newline='')
    # csv按行写入
    writer = csv.writer(csv_file)
    writer.writerow(headers)
    writer.writerows(values)
    csv_file.close()

    return rate


if __name__ == "__main__":
    # First_path = r'.\L0002_lingjian.csv'
    Second_path = r'.\L0003_lingjian.csv'
    mianliao_height = 1600.0
    # rate1 = get_rate(First_path, mianliao_height)
    rate2 = get_rate(Second_path, mianliao_height)
    # print(round(rate1/2 + rate2/2, 3))
    print(round(rate2, 4))










