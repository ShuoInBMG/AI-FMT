import cv2
import os
import matplotlib.pyplot as plt

def save_img():
    video_path = r"D:/inuse"
    videos = os.listdir(video_path)
    for video_name in videos:
        file_name = video_name.split('.')[0]
        folder_name = video_path + file_name
        os.makedirs(folder_name, exist_ok=True)
        print(video_path+'/'+video_name)
        vc = cv2.VideoCapture(video_path+'/'+video_name) 
        c=0
        rval=vc.isOpened()

        while rval:   
            c = c + 1
            rval, frame = vc.read()
            pic_path = folder_name+'/'
            # region of interest
            image = frame[960:1160,234:434]
            filename = pic_path + str(c) + '.png'
            if c < 390:
                pass            
            elif rval and (c >= 390) and (c <= 3990):
                cv2.imwrite(filename, image) 
                print(pic_path + str(c) + '.png')
            else:
                break
        vc.release()
        print('save_success')
        print(folder_name)
save_img()