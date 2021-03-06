import cv2
import pygame as pg


RES = 1200, 900
RES = 960, 720
# main_capture = cv2.VideoCapture('img/gg.mp4')
main_capture = cv2.VideoCapture(0)
# back_captuer = cv2.VideoCapture('img/wdwd.mkv')
back_captuer = cv2.VideoCapture('img/wdw.mkv')
subtractor = cv2.createBackgroundSubtractorKNN()



while True:
    frame = main_capture.read()[1]
    frame = cv2.resize(frame, RES, interpolation=cv2.INTER_AREA)

    bg = back_captuer.read()[1]
    bg = cv2.resize(bg, RES, interpolation=cv2.INTER_AREA)

    mask = subtractor.apply(frame, 1)
    bitwice = cv2.bitwise_and(bg, bg, mask=mask)
    # cv2.imshow("", frame)
    cv2.imshow("", bitwice)
    video = cv2.VideoWriter("output/pneo_artw.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25, RES)



    if cv2.waitKey(1) == 27:
        break

video.write(bitwice)
cv2.destroyAllWindows()
video.release()

