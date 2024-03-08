from mtcnn.mtcnn import MTCNN
import cv2
import pickle

def real_time_face_recognition(save_vid=False):
    '''
    A demo of running real-time face recognition using the webcam
    
    Parameters:
    save_vid (bool): Whether to save the video or not. Default is False.
    '''
    
    detector = MTCNN()
    cap = cv2.VideoCapture(0)

    cap.set(3,640) # set Width
    cap.set(4,480) # set Height

    if save_vid:
        size = (int(cap.get(3)), int(cap.get(4)))
        result = cv2.VideoWriter('demo.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

    while(True):
        ret, frame = cap.read()

        if not ret:
            break

        # Flip camera vertically
        frame = cv2.flip(frame, 1) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detect_faces(frame)
        for face in faces:
            x, y, width, height = face['box']
            c = (0, 0, 255)
            name = 'Unknown'
            
            try:
                face_roi = gray[int((y+height)/2)-31:int((y+height)/2)+31, int((x+width)/2-23.5):int((x+width)/2+23.5)] 
                x_test = reduce_dimension(face_roi.reshape(1, -1))
                name = face_identification_svm(x_test)
            except:
                pass

            font_scale = width / 300
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.rectangle(frame, (x, y), (x+width, y+height), c, 2)
            cv2.rectangle(frame, (x, y+height-int(height/10)), (x+width, y+height), c, -1)
            cv2.putText(frame, str(name), (x+2, y+height-1), font, font_scale, (255, 255, 255), 1)

        cv2.imshow('Face Recognition', frame)

        if save_vid:
            result.write(frame)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break
    cap.release()
    cv2.destroyAllWindows()

def face_identification_svm(face):
    '''Return the name of the person in the photo using SVM classification'''
    with open('svc.pkl', 'rb') as file:
        svm = pickle.load(file)
    name = svm.predict(face)
    return name[0]

def reduce_dimension(face):
    '''Reduce the dimensions of the face encodings using PCA'''
    with open('pca.pkl', 'rb') as file:
        pca = pickle.load(file)
    return pca.transform(face)

if __name__ == "__main__":
    real_time_face_recognition()