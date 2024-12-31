# importing openCV librabry
import cv2

def read_video(video_path):
    '''this function will read video from the given video path
    Each video consist of different images(frames) 24fps(frame per second)'''
    cap = cv2.VideoCapture(video_path)
    frames = []
    # we loop on each frame and read each frame
    # if ret is False means for the last frame then we break the loop else we append it in the frame list
    while True:
        ret, frame = cap.read() 
        if not ret:
            break
        frames.append(frame)
    # return the frames list
    return frames

def save_video(ouput_video_frames,output_video_path):
    '''This function save the video which takes output frames and the output video path where the video will be saved'''
    # this is specifying the video format
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # specifying output video by giving path,format,fps,(width,height)
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()