import sched

import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
from glob import glob

import tcpClient
from align import AlignPoints
from clientdemo import Conf, HttpHelper
from clientdemo.DataModel import PoseStatus, DetailPoseInfo, PoseInfo, RoomInfo
from src import model
from src import util
from src.body import Body
import torch.nn as nn
import time
import threading
import datetime


def get_time_now():
    return datetime.datetime.now().strftime('%m-%d %H:%M:%S')


class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 4)
        self.drop = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def prepare_posreg_multiperson(corr, subset):
    multi_person = []
    for persion_idx in range(len(subset)):
        pos_idx_tmp = [int(j) for j in subset[persion_idx]]
        pos_idx_tmp = pos_idx_tmp[:18]
        multi_person.append(corr[pos_idx_tmp])
    pos = np.array(multi_person)[:, :18, :3]
    return pos


def reg_infer(pos):
    conf = pos[:, :, -1:]
    pos = aligner.align_points(pos)
    pos = (pos - 129) / 255
    pos = np.concatenate([pos, conf], -1)
    pos = pos.reshape([pos.shape[0], -1])
    pos = torch.Tensor(pos).cuda()
    out = model(pos)
    _, classidx = torch.max(out, 1)
    return classidx


def aged_status_reset(aged):
    aged.timesit = 0
    aged.timelie = 0
    aged.timestand = 0
    aged.timedown = 0
    aged.timeother = 0
    aged.timein = None


def aged_status_sync(aged):
    '''
    读取数据库中该用户的数据记录赋值为初始值
    :param aged: PoseInfo对象实例
    :return: None
    '''
    # 拼接url，参考接口文档
    aged_today_status__url = Conf.Urls.PoseInfoUrl + "/" + str(aged.agesinfoid)
    print(f'get {aged_today_status__url}')

    try:
        aged_status_today = HttpHelper.get_items(aged_today_status__url)
    except Exception:  # 还没有数据记录
        return
    aged.timesit = aged_status_today.timeSit
    aged.timelie = aged_status_today.timeLie
    aged.timestand = aged_status_today.timeStand
    aged.timedown = aged_status_today.timeDown
    aged.timeother = aged_status_today.timeOther


def pose_detect_with_video(aged_id, classidx, human_box, parse_pose_demo_instance):
    use_aged = ages[aged_id]
    # detect if new day
    now_date = time.strftime('%Y-%m-%dT00:00:00', time.localtime())
    if not now_date == use_aged.date:  # a new day
        aged_status_reset(use_aged)
        use_aged.is_first_frame = True
        use_aged.date = now_date

    if use_aged.is_first_frame:  # 第一帧，开始计时
        # 从服务器获取当天的状态记录信息，进行本地值的更新，防止状态计时被重置
        aged_status_sync(use_aged)
        use_aged.is_first_frame = False
    else:
        last_pose_time = time.time() - use_aged.last_time  # 上一个状态至今的时间差，单位为s
        if use_aged.status == PoseStatus.Sit.value:
            use_aged.timesit += last_pose_time
        elif use_aged.status == PoseStatus.Down.value:
            use_aged.timedown += last_pose_time
        elif use_aged.status == PoseStatus.Lie.value:
            use_aged.timelie += last_pose_time
        elif use_aged.status == PoseStatus.Stand.value:
            use_aged.timestand += last_pose_time
        else:
            use_aged.timeother += last_pose_time

    use_aged.last_time = time.time()

    if classidx == 0:
        now_status = PoseStatus.Sit.value
    elif classidx == 1:
        now_status = PoseStatus.Lie.value
    elif classidx == 2:
        now_status = PoseStatus.Stand.value
    elif classidx == 3:
        now_status = PoseStatus.Lie.value
    else:
        now_status = PoseStatus.Other.value

    now_date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    if not now_status == use_aged.status:  # 新的行为发生
        if not now_date_time == use_aged.datetime:  # 根据实际情况，每秒只记录一次状态更改操作
            temp_detail_pose_info = DetailPoseInfo(agesInfoId=aged_id, dateTime=now_date_time, status=now_status)
            # 写数据记录到数据库表DetaiPoseInfo
            detail_pose_url = Conf.Urls.DetailPoseInfoUrl
            http_result = HttpHelper.create_item(detail_pose_url, temp_detail_pose_info)
    use_aged.datetime = now_date_time

    if parse_pose_demo_instance.camera.isUseSafeRegion:
        is_outer_chuang = False  # 临时变量，指示是否在安全区外
        is_outer_chair = False # 临时变量，指示是否在椅子安全区外
        #  因为床的矩形坐标是在原图压缩1/2之后的值，所以下面的值也需要压缩1/2
        xmin, ymin, xmax, ymax = int(human_box[0]), int(human_box[1]), int(human_box[2]), int(
            human_box[3])
        x_core = int(xmin+(xmax-xmin)/2)
        y_core = int(ymin+(ymax-ymin)/2)

        if x_core > 472\
            or y_core > 270 \
            or x_core < 288 \
            or y_core < 184:
            is_outer_chair = True

        if x_core > parse_pose_demo_instance.camera.rightBottomPointX \
                or y_core > parse_pose_demo_instance.camera.rightBottomPointY \
                or x_core < parse_pose_demo_instance.camera.leftTopPointX \
                or y_core < parse_pose_demo_instance.camera.leftTopPointY:
            is_outer_chuang = True

        use_aged.isalarm = False
        #rooms[parse_pose_demo_instance.camera.roomInfoId].isalarm = False
        # 判断当前状态是否需求报警
        if is_outer_chuang and is_outer_chair:
            if now_status == PoseStatus.Lie.value:
                use_aged.isalarm = True
                #rooms[parse_pose_demo_instance.camera.roomInfoId].isalarm = True
                now_status = PoseStatus.Down.value
    else:
        if now_status == PoseStatus.Down.value:  # TODO：这里的给值是不对的，需要赋予识别服务的对应的需要报警的状态值
            use_aged.isalarm = True

    use_aged.status = now_status


class ParsePoseCore:
    def __init__(self, camera, pose_model, use_body_estimation, tcp_client):
        self.camera = camera
        self.pose_model = pose_model
        self.use_body_estimation = use_body_estimation
        self.tcp_client = tcp_client
        self.is_stop = True
        self.stream = None
        self.is_first_frame = True

    def start(self):
        self.is_stop = False
        self.stream = cv2.VideoCapture(self.camera.videoAddress)
        print(f'videoAddress {self.camera.videoAddress}')
        t = threading.Thread(target=self.parse, args=())
        t.daemon = True
        t.start()

    def stop(self):
        self.is_stop = True

    def parse(self):
        frame_num = 0
        while not self.is_stop:
            (grabbed, frame) = self.stream.read()
            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file or we disconnect

            if grabbed:
                #  because it is so slow to detect, so detect once each 20 frames
                if frame_num < 2:
                    frame_num += 1
                else:
                    frame_num = 0
                    h, w, c = frame.shape
                    # 将原图片尺寸压缩，宽高都压缩到原来的1/2.
                    oriImg = cv2.resize(frame, (int(w / 2), int(h / 2)), interpolation=cv2.INTER_CUBIC)
                    corrs, subset = self.use_body_estimation(oriImg)
                    if corrs.any() or not subset.shape[0] == 0:
                        subset_vis = subset.copy()
                        oriImg = util.draw_bodypose(oriImg, corrs, subset_vis)
                        if self.tcp_client.is_room_video_send:
                            # oriImg = cv2.resize(frame, (int(w / 2), int(h / 2)), interpolation=cv2.INTER_CUBIC)
                            self.tcp_client.send_img(oriImg)
                            # print(f'{get_time_now()} send img')
                        try:
                            corrs = prepare_posreg_multiperson(corrs, subset)
                        except Exception:
                            continue
                        preds = reg_infer(corrs)
                        preds = preds.cpu().numpy()

                        for aged in self.camera.roomInfo.agesInfos:
                            if not aged.id in ages.keys():
                                ages[aged.id] = PoseInfo(agesInfoId=aged.id,
                                                         date=time.strftime('%Y-%m-%dT00:00:00', time.localtime()),
                                                         dateTime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                         timeStand=0,
                                                         timeSit=0,
                                                         timeLie=0,
                                                         timeDown=0,
                                                         timeOther=0,
                                                         last_position=0
                                                         )
                        if self.is_first_frame:
                            if corrs.shape[0] == len(self.camera.roomInfo.agesInfos):
                                self.is_first_frame = False
                                for i in range(corrs.shape[0]):
                                    xmin, ymin, xmax, ymax = np.min(corrs[i, :, 0]), np.min(corrs[i, :, 1]), np.max(
                                        corrs[i, :, 0]), np.max(corrs[i, :, 1])
                                    aged = ages[self.camera.roomInfo.agesInfos[i].id]
                                    aged.last_position = ymax
                            else:
                                continue

                        for i in range(corrs.shape[0]):
                            if i <= len(self.camera.roomInfo.agesInfos) - 1:  # 避免人的列表溢出
                                xmin, ymin, xmax, ymax = np.min(corrs[i, :, 0]), np.min(corrs[i, :, 1]), np.max(
                                    corrs[i, :, 0]), np.max(corrs[i, :, 1])

                                # 比较最小偏移来进行人的简单跟踪
                                min_aged_offset = abs(ymax - ages[self.camera.roomInfo.agesInfos[0].id].last_position)
                                min_aged_id = self.camera.roomInfo.agesInfos[0].id
                                for aged in self.camera.roomInfo.agesInfos:
                                    poseinfo = ages[aged.id]
                                    if abs(ymax - poseinfo.last_position) < min_aged_offset:
                                        min_aged_offset = abs(ymax - poseinfo.last_position)
                                        min_aged_id = aged.id
                                pose_detect_with_video(min_aged_id, preds[i],
                                                       (xmin, ymin, xmax, ymax), self)
            else:
                #  reconnect the video stream
                self.stream = cv2.VideoCapture(self.camera.videoAddress)
                print(f'{get_time_now()} restart videoCapture')

        self.stream.release()


def write_database():
    """
    每1秒更新一次数据库表PoseInfo的记录信息
    :return:
    """
    while not is_stop:
        pose_url = Conf.Urls.PoseInfoUrl + '/UpdateOrCreatePoseInfo'

        for aged in ages.values():
            temp_pose_info = PoseInfo(agesInfoId=aged.agesinfoid,
                                      date=aged.date,
                                      dateTime=aged.datetime,
                                      timeStand=int(float(aged.timestand)),
                                      timeSit=int(float(aged.timesit)),
                                      timeLie=int(float(aged.timelie)),
                                      timeDown=int(float(aged.timedown)),
                                      timeOther=int(float(aged.timeother)),
                                      isAlarm=aged.isalarm,
                                      status=aged.status
                                      )
            http_result = HttpHelper.create_item(pose_url, temp_pose_info)

        # 更新房间状态信息
        room_url = Conf.Urls.RoomInfoUrl
        
        for camera in current_server.cameraInfos:  # 遍历本服务器需要处理的摄像头
            temp_room_isalarm = False
            for aged in camera.roomInfo.agesInfos:
                if not aged.id in ages.keys():
                    break
                poseinfo = ages[aged.id]
                if poseinfo.isalarm:
                    temp_room_isalarm = True
                    break
            if temp_room_isalarm:
                rooms[camera.roomInfoId].isalarm = True
            else:
                rooms[camera.roomInfoId].isalarm = False
            
        for room in rooms.values():
            result= HttpHelper.update_item_post(room_url, room.id, room)

        time.sleep(1)


def wait_for_stop_cmd():
    global is_stop
    input("\n\n input ENTER to exit")
    is_stop = True

    for tcp_client in tcp_clients:
        tcp_client.stop()

    for pose_parse_instance in pose_parse_instances:
        pose_parse_instance.stop()


ages = {}  # 老人字典
rooms = {}  # 本服务器监控的房间字典
cameras = {}  # 摄像头字典
tcp_clients = []
pose_parse_instances = []
# 获取或设置本机IP地址信息
local_ip = '192.168.1.60'
tcp_server_ip = '127.0.0.1'
tcp_server_port = 8008
is_stop = False

print(f"Torch device: {torch.cuda.get_device_name()}")

if __name__ == "__main__":
    model = NeuralNet(18 * 3).cuda()
    model.load_state_dict(torch.load('./49_model.ckpt'))
    body_estimation = Body('model/body_pose_model.pth')
    aligner = AlignPoints()

    current_server_url = Conf.Urls.ServerInfoUrl + "/GetServerInfo?ip=" + local_ip
    print(f'get {current_server_url}')

    current_server = HttpHelper.get_items(current_server_url)
    print(f'current_server.camera_count: {len(current_server.cameraInfos)}')

    for camera in current_server.cameraInfos:  # 遍历本服务器需要处理的摄像头
        temp_tcp_client = tcpClient.TcpClient(tcp_server_ip, tcp_server_port, camera.id, camera.roomInfoId)
        temp_tcp_client.start()
        tcp_clients.append(temp_tcp_client)

        temp_ai_instance = ParsePoseCore(camera, model, body_estimation, temp_tcp_client)
        temp_ai_instance.start()
        pose_parse_instances.append(temp_ai_instance)

        rooms[camera.roomInfoId] = RoomInfo(id=camera.roomInfo.id,
                                            name=camera.roomInfo.name,
                                            isAlarm=camera.roomInfo.isAlarm,
                                            roomSize=camera.roomInfo.roomSize)

    # 定时调度来更新数据库
    t = threading.Thread(target=write_database, args=())
    t.daemon = True
    t.start()

    wait_for_stop_cmd()
