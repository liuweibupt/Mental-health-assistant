# 疲劳识别综合部分
# 包含视频眼部闭眼时间占比和哈欠识别部分
# 眼部疲劳检测采用闭眼次数比值作为疲劳阈值，占90
# 哈欠超出阈值后，1-1/(x+1) *10 进行评分
import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from scipy.spatial import distance

# 定义常数
# 眼睛长宽比阈值
EYE_AR_THRESH = 0.2
# 打哈欠长宽比阈值
MAR_THRESH = 0.5
# 哈欠个数阈值
MOUTH_FRAMES = 3
# 初始化帧计数器和眨眼总数
COUNTER = 0
TOTAL = 0
# 初始化帧计数器和打哈欠总数
mCOUNTER = 0
mTOTAL = 0
# 帧数
counter_x = 0
counter_eye = 0


# 计算眼部精神程度（张开比例）
def eye_open_percent(eye):
    e1 = distance.euclidean(eye[1], eye[5])
    e2 = distance.euclidean(eye[2], eye[4])
    # 计算水平之间的距离
    dist_eye = distance.euclidean(eye[0], eye[3])
    e_open = (e1 + e2) / (2.0 * dist_eye)
    return e_open


# 计算嘴部张开比例
def mouth_open_percent(mouth):  # 嘴部
    m1 = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    m2 = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    m3 = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    m = (m1 + m2) / (2.0 * m3)
    return m


# 初始化Dlib的人脸检测器（HOG），然后创建面部标志物预测
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')

# 分别获取左右眼面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# 打开cv2 本地摄像头
cap = cv2.VideoCapture(0)

# 从视频流循环帧
while True:
    counter_x += 1
    temp_k_flag = 0
    # 进行循环，读取图片，并对图片做维度扩大，并进灰度化
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用detector(gray, 0) 进行脸部位置检测
    rects = detector(gray, 0)

    # 循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
    for rect in rects:
        shape = predictor(gray, rect)
        # 将脸部特征信息转换为数组array的格式
        shape = face_utils.shape_to_np(shape)
        # 提取左眼和右眼坐标
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # 嘴巴坐标
        mouth = shape[mStart:mEnd]

        # 构造函数计算左右眼的OPEN值，使用平均值作为最终的OPEN
        leftEAR = eye_open_percent(leftEye)
        rightEAR = eye_open_percent(rightEye)
        open_e = (leftEAR + rightEAR) / 2.0
        # 打哈欠
        mar = mouth_open_percent(mouth)

        # 进行画图操作，用矩形框标注人脸
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)

        if open_e < EYE_AR_THRESH:
            counter_eye += 1

        # 进行画图操作，同时使用cv2.putText将眨眼次数进行显示
        cv2.putText(frame, "Faces: {}".format(len(rects)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "eye degree: {:.2f}".format(100*counter_eye/counter_x), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0),2)

        # 计算张嘴评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示打了一次哈欠，同一次哈欠大约在3帧
        # 同理，判断是否打哈欠
        if mar > MAR_THRESH:  # 张嘴阈值0.5
            mCOUNTER += 1
        else:
            # 如果连续3次都小于阈值，则表示打了一次哈欠
            if mCOUNTER >= MOUTH_FRAMES:  # 阈值：3
                mTOTAL += 1
            # 重置嘴帧计数器
            mCOUNTER = 0

        cv2.putText(frame, "Yawning: {}".format(mTOTAL), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(frame, "Tired degree: {:.2f}".format(90*counter_eye/counter_x+30*(1-1/(1+mTOTAL))), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 按q退出
    cv2.putText(frame, "Press 'q': Quit", (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 2)
    # 窗口显示 show with opencv
    cv2.imshow("Frame", frame)

    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头 release camera
cap.release()
# do a bit of cleanup
cv2.destroyAllWindows()
