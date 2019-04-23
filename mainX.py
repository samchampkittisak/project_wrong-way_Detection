import cv2
import numpy as np
import vehiclesX
import time
import os
import requests
import shutil
from threading import Thread
from imageai.Prediction import ImagePrediction
'''
# set time Run Program
while True:
    print(time.strftime("%H:%M:%S", time.localtime()))
    time.sleep(1)
    if "07:30" == time.strftime("%H:%M", time.localtime()):
        break
'''
# Alert Line
url = 'https://notify-api.line.me/api/notify'
token = 'CJPhtEYFbF1l53D8itWCetndYPk0cWafKQWDSdjJlWJ'
headers = {'content-type': 'application/x-www-form-urlencoded', 'Authorization': 'Bearer ' + token}

# open cctv
cap = cv2.VideoCapture()
cap.open("outpyRight.mp4")
cap1 = cv2.VideoCapture()
cap1.open("...")

# pathdir file
pathsetfile = "E:\\projectdetectlistImageandVideo"

# Number of days to delete directory

Delete_day = 5
liststr_dir = []


def notifyFile(filename, idcar, nametype, per, tokens):
    file = {'imageFile': open(filename, 'rb')}
    payload = {'message': 'ID%d is %s percent = %s' % (idcar, nametype, per)}
    return _lineNotify(tokens, payload, file)


def _lineNotify(tokens, payload, file=None):
    url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': 'Bearer ' + tokens}
    return requests.post(url, headers=headers, data=payload, files=file)


def alert_function(cnt, Time, path, Name, per, tokens):
    msg = 'ขณะนี้จับรถย้อนศรได้ ID%d วันที่ %s' % (cnt, Time)
    r = requests.post(url, headers=headers, data={'message': msg})
    notifyFile(path, cnt, Name, per, tokens)
    print(r.text)


def PredictImageCheck(pathset, path, NameImage):
    readImg = cv2.imread(os.path.join(pathset, path, NameImage))
    if readImg is not None:
        predictions, probabilities = prediction.predictImage(os.path.join(pathset, path, NameImage), result_count=1)
        for eachPrediction, eachProbability in zip(predictions, probabilities):
            print(eachPrediction, " : ", eachProbability)
        if eachPrediction in Modelcheck:
            return True, eachPrediction, eachProbability
        else:
            return False, "None", 0.00
    else:
        return False, "None", 0.00


def PredictImageTrack(pathset, pathtr, NameImagetr):
    readImg = cv2.imread(os.path.join(pathset, pathtr, NameImagetr))
    if readImg is not None:
        predic, proba = prediction.predictImage(os.path.join(pathset, pathtr, NameImagetr), result_count=1)
        for eachPredict, eachProba in zip(predic, proba):
            print(eachPredict)
        return eachPredict, eachProba
    else:
        return "None", 0.00


def createDirectory(pathset, dirNameLeft, dirNameRight, dirNameLeftCrop, dirNameRightCrop, dirNameVideo, dirDay):
    pathdir = pathset + "\\%s" % dirDay
    if not os.path.exists(pathdir):
        os.mkdir(pathdir)
        print("Directory ", dirDay, " Created ")
    else:
        print("Directory ", dirDay, " already directory")

    print(pathdir)
    pathdirNameLeft = pathdir + "\\%s" % dirNameLeft
    pathdirNameRight = pathdir + "\\%s" % dirNameRight
    pathdirNameLeftCrop = pathdir + "\\%s" % dirNameLeftCrop
    pathdirNameRightCrop = pathdir + "\\%s" % dirNameRightCrop
    pathdirNameVideo = pathdir + "\\%s" % dirNameVideo
    if not os.path.exists(pathdirNameLeft):
        os.mkdir(pathdirNameLeft)
        print("Directory ", dirNameLeft, " Created ")
    else:
        print("Directory ", dirNameLeft, " already directory")

    if not os.path.exists(pathdirNameRight):
        os.makedirs(pathdirNameRight)
        print("Directory ", dirNameRight, " Created ")
    else:
        print("Directory ", dirNameRight, " already directory")

    if not os.path.exists(pathdirNameLeftCrop):
        os.makedirs(pathdirNameLeftCrop)
        print("Directory ", dirNameLeftCrop, " Created ")
    else:
        print("Directory ", dirNameLeftCrop, " already directory")

    if not os.path.exists(pathdirNameRightCrop):
        os.makedirs(pathdirNameRightCrop)
        print("Directory ", dirNameRightCrop, " Created ")
    else:
        print("Directory ", dirNameRightCrop, " already directory")

    if not os.path.exists(pathdirNameVideo):
        os.makedirs(pathdirNameVideo)
        print("Directory ", dirNameVideo, " Created ")
    else:
        print("Directory ", dirNameVideo, " already directory")


cnt_right = 0
cnt_left = 0
str_predict = "None"


NameLeft = 'PictureLeftFrame'
NameRight = 'PictureRightFrame'
NameLeftCrop = 'PictureLeftCrop'
NameRightCrop = 'PictureRightCrop'
NameVideo = 'videoRight'
Thispath = os.getcwd()
Nameday = time.strftime("day%d.%b.%Y", time.localtime())
createDirectory(pathsetfile, NameLeft, NameRight, NameLeftCrop, NameRightCrop, NameVideo, Nameday)
liststr_dir.append(Nameday)
print(liststr_dir[0])

# Get width and height of video
w1 = cap.get(3)
h1 = cap.get(4)
print(w1)
print(h1)
W = 1300
imgScale = float(W / w1)
ret, framevideo = cap.read()
newX, newY = framevideo.shape[1] * imgScale, framevideo.shape[0] * imgScale
frameArea = newX * newY
areaTH = frameArea / 200

# Lines
left_limit = int(6 * (newX / 10))
line_left = int(7 * (newX / 10))
line_right = int(7 * (newX / 10))
right_limit = int(9 * (newX / 10))
up_limit = int(2.5 * (newY / 10))
down_limit = int(9 * (newY / 10))

# print pixel in line
print("Right line x:", str(line_right))
print("Left line x:", str(line_left))

# color line
line_right_color = (80, 127, 255)
line_left_color = (0, 255, 255)

# showtrackX
deletelistX = 80
deletelistY = 20

pt1 = [line_right, 0]
pt2 = [line_right, newY]
pts_L1 = np.array([pt1, pt2], np.int32)
pts_L1 = pts_L1.reshape((-1, 1, 2))
pt3 = [line_left, 0]
pt4 = [line_left, newY]
pts_L2 = np.array([pt3, pt4], np.int32)
pts_L2 = pts_L2.reshape((-1, 1, 2))

pt5 = [left_limit, 0]
pt6 = [left_limit, newY]
pts_L3 = np.array([pt5, pt6], np.int32)
pts_L3 = pts_L3.reshape((-1, 1, 2))
pt5A = [left_limit - deletelistX, 0]
pt6A = [left_limit - deletelistX, newY]
pts_L3L = np.array([pt5A, pt6A], np.int32)
pts_L3L = pts_L3L.reshape((-1, 1, 2))

pt7 = [right_limit, 0]
pt8 = [right_limit, newY]
pts_L4 = np.array([pt7, pt8], np.int32)
pts_L4 = pts_L4.reshape((-1, 1, 2))
pt7A = [right_limit + deletelistX, 0]
pt8A = [right_limit + deletelistX, newY]
pts_L4R = np.array([pt7A, pt8A], np.int32)
pts_L4R = pts_L4R.reshape((-1, 1, 2))

pt9 = [0, up_limit]
pt10 = [newX, up_limit]
pts_L5 = np.array([pt9, pt10], np.int32)
pts_L5 = pts_L5.reshape((-1, 1, 2))
pt9A = [0, up_limit - deletelistY]
pt10A = [newX, up_limit - deletelistY]
pts_L5U = np.array([pt9A, pt10A], np.int32)
pts_L5U = pts_L5U.reshape((-1, 1, 2))

pt11 = [0, down_limit]
pt12 = [newX, down_limit]
pts_L6 = np.array([pt11, pt12], np.int32)
pts_L6 = pts_L6.reshape((-1, 1, 2))
pt11A = [0, down_limit + deletelistY]
pt12A = [newX, down_limit + deletelistY]
pts_L6D = np.array([pt11A, pt12A], np.int32)
pts_L6D = pts_L6D.reshape((-1, 1, 2))

# Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=100, detectShadows=False)

# Kernals
kernalOp = np.ones((3, 3), np.uint8)
kernalOp2 = np.ones((5, 5), np.uint8)
kernalCl = np.ones((20, 20), np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
cars = []
cars2 = []
max_p_age = 5
pid = 1
alert = 0
croptime = "ERROR"
croptime1 = "ERROR"

# path
pathCropLeft = pathsetfile + '\\' + NameLeftCrop
pathCropRight = pathsetfile + '\\' + NameLeftCrop

# imageAIsetPathModel
prediction = ImagePrediction()
prediction.setModelTypeAsInceptionV3()
prediction.setModelPath(os.path.join(os.getcwd(), "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
prediction.loadModel()

# imageAItrackRealTime
Trackpredic = ImagePrediction()
Trackpredic.setModelTypeAsInceptionV3()
Trackpredic.setModelPath(os.path.join(os.getcwd(), "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
Trackpredic.loadModel(prediction_speed="faster")

# NameModelfortoggle
Modelcheck = ["motor_scooter", "golfcart", "moped",
              "convertible", "sports_car", "pickup",
              "minivan", "bus", "tricycle",
              "go-kart", "cab", "limousine",
              "minibus", "moving_van"]

# ready write video
waitTime = 0
cntframe = 0
Videoout = None
videoTime = 10
fps = 30
TotalFramevideo = 10*fps


while (cap.isOpened()):
    ret, framevideo = cap.read()
    ret1, framevideo1 = cap1.read()
    frame = cv2.resize(framevideo, (int(newX), int(newY)))
    for i in cars:
        i.age_one()
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)
    #cv2.imshow("fgmask", fgmask)
    if waitTime == 1:
        Videoout.write(framevideo1)
        cntframe += 1
        if cntframe <= TotalFramevideo:
            waitTime = 1
        else:
            waitTime = 0
            cntframe = 0
            Videoout.release()
    if ret is True:

        # Binarization
        ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        rets, imBin2 = cv2.threshold(fgmask2, 200, 255, cv2.THRESH_BINARY)
        # OPening i.e First Erode the dilate
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp2)
        mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_CLOSE, kernalOp2)
        # Closing i.e First Dilate then Erode
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernalCl)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernalCl)

        # Find Contours
        countours0, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in countours0:
            area = cv2.contourArea(cnt)

            if area > areaTH:
                ####Tracking######
                m = cv2.moments(cnt)
                cx = int(m['m10'] / m['m00'])
                cy = int(m['m01'] / m['m00'])
                x, y, w, h = cv2.boundingRect(cnt)

                new = True
                if cx in range(left_limit - deletelistX, right_limit + deletelistX) and cy in range(
                        up_limit + deletelistY, down_limit + deletelistY):
                    for i in cars:
                        if abs(x - i.getX()) <= w + 20 and abs(y - i.getY()) <= h + 20:
                            new = False
                            i.updateCoords(cx, cy)
                            if i.going_RIGHT(line_left, line_right) is True:
                                cnt_right += 1
                                alert = 1
                                print("ID:", i.getId(), 'crossed going Right at', time.strftime("%c"))
                                crop_carRight = frame[y - 20:y + h + 20, x - 20:x + w + 20]
                                croptime = time.strftime("day%d%b%Y time%Hh%Mm%Ss", time.localtime())
                                pathdirRight = pathsetfile + '\\%s\\' % Nameday + NameRightCrop + "\\%s ID%d.jpg" % (
                                    croptime, cnt_right)
                                pathdirRightframe = pathsetfile + '\\%s\\' % Nameday + NameRight + "\\%s ID%d.jpg" % (
                                    croptime, cnt_right)
                                pathdirRightVideo = pathsetfile + '\\%s\\' % Nameday + NameVideo + "\\%s ID%d.mp4" % (
                                    croptime, cnt_right)
                                namepictureRight = "%s ID%d.jpg" % (croptime, cnt_right)
                                cv2.imwrite(pathdirRight, crop_carRight)
                                cv2.imwrite(pathdirRightframe, frame)
                            elif i.going_LEFT(line_left, line_right) is True:
                                cnt_left += 1
                                print("ID:", i.getId(), 'crossed going LEFT at', time.strftime("%c"))
                                crop_carLeft = frame[y - 20:y + h + 20, x - 20:x + w + 20]
                                croptime1 = time.strftime("day%d%b%Y time%Hh%Mm%Ss", time.localtime())
                                pathdirLeft = pathsetfile + '\\%s\\' % Nameday + NameLeftCrop + "\\%s ID%d.jpg" % (
                                    croptime1, cnt_left)
                                pathdirLeftframe = pathsetfile + '\\%s\\' % Nameday + NameLeft + "\\%s ID%d.jpg" % (
                                    croptime1, cnt_left)
                                pathdirLeftVideo = pathsetfile + '\\%s\\' % Nameday + NameVideo + "\\%s ID%d.mp4" % (
                                    croptime1, cnt_left)
                                namepictureLeft = "%s ID%d.jpg" % (croptime1, cnt_left)
                                cv2.imwrite(pathdirLeft, crop_carLeft)
                                cv2.imwrite(pathdirLeftframe, frame)
                                AI, percent = PredictImageTrack(pathset=pathsetfile+"\\", pathtr=Nameday + "\\" + NameLeftCrop,
                                                                NameImagetr=namepictureLeft)
                                str_predict = "LEFT ID %d : " % cnt_left + AI + " %.2f percent" % percent

                        if alert == 1:

                            check, AI, percent = PredictImageCheck(pathset=pathsetfile + "\\", path=Nameday + "\\" + NameRightCrop,
                                                                   NameImage=namepictureRight)
                            str_predict = "RIGHT ID %d : " % cnt_right + AI + " %.2f percent" % percent
                            if check:
                                thread_Token = Thread(target=alert_function,
                                                      args=(cnt_right, croptime, pathdirRightframe, AI, percent, token))
                                thread_Token.start()
                                if waitTime == 0:
                                    Videoout = cv2.VideoWriter(pathdirRightVideo, 0x00000021, fps, (int(cap1.get(3)), int(cap1.get(4))))
                                    startTime = time.time()
                                    waitTime = 1
                            alert = 0

                        if i.getState() == '1':
                            if i.getDir() == 'right' and (
                                    i.getX() > right_limit or i.getY() < up_limit or i.getY() > down_limit):
                                i.setDone()
                            elif i.getDir() == 'left' and (
                                    i.getX() < left_limit or i.getY() < up_limit or i.getY() > down_limit):
                                i.setDone()
                        if i.timedOut():
                            index = cars.index(i)
                            cars.pop(index)
                            del i

                    if new is True and left_limit < cx < right_limit and up_limit < cy < down_limit:  # If nothing is detected,create new
                        p = vehiclesX.Car(pid, cx, cy, max_p_age)
                        cars.append(p)
                        pid += 1

                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for i in cars:
            cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), font, 3, i.getRGB(), 3, cv2.LINE_AA)
            cv2.line(frame, (i.getX(), i.getY()), (i.last_x, i.last_y), (0, 255, 255), 3)
            i.setlastX(i.getX())
            i.setlastY(i.getY())

        if Nameday != time.strftime("day%d.%b.%Y", time.localtime()):
            print("create directory Next day")
            Nameday = time.strftime("day%d.%b.%Y", time.localtime())
            createDirectory(pathsetfile, NameLeft, NameRight, NameLeftCrop, NameRightCrop, NameVideo, Nameday)
            liststr_dir.append(Nameday)
            if len(liststr_dir) > Delete_day:
                shutil.rmtree(pathsetfile+'\\'+liststr_dir[0])
                del liststr_dir[0]
            pid = 0
            cnt_right = 0
            cnt_left = 0


        str_up = 'Right: ' + str(cnt_right)
        str_down = 'left: ' + str(cnt_left)
        frame = cv2.polylines(frame, [pts_L1], False, line_right_color, thickness=2)
        frame = cv2.polylines(frame, [pts_L2], False, line_left_color, thickness=2)
        frame = cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
        frame = cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)
        frame = cv2.polylines(frame, [pts_L5], False, (255, 255, 255), thickness=1)
        frame = cv2.polylines(frame, [pts_L6], False, (255, 255, 255), thickness=1)
        frame = cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
        frame = cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)

        cv2.putText(frame, str_up, (10, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_up, (10, 40), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 90), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 90), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str_predict, (10, 140), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_predict, (10, 140), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (left_limit - deletelistX, up_limit - deletelistY),
                      (right_limit + deletelistX, down_limit + deletelistY), (0, 150, 0), 2)
        cv2.putText(frame, "AreaList", (left_limit - 20, up_limit - 20), font, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, "AreaList", (left_limit - 20, up_limit - 20), font, 1, (150, 150, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (left_limit, up_limit), (right_limit, down_limit), (150, 150, 0), 5)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xff is ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
