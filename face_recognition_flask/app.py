import cv2
import os
from flask import Flask, request, render_template, jsonify
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)

# Maximum number of images to capture
nimgs = 10
MAX_IMAGES = 20  # Safety limit

# Maximum time for image capture (in seconds)
MAX_CAPTURE_TIME = 30

imgBackground=cv2.imread("background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%I:%M %p")
    
    # Allow multiple entries for the same person with different timestamps
    with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
        f.write(f'\n{username},{userid},{current_time}')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    
    # Create window and set it to be always on top
    cv2.namedWindow('Attendance', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Attendance', cv2.WND_PROP_TOPMOST, 1)
    
    # Keep track of marked attendances in this session
    marked_people = []
    
    frame_height = 480
    frame_width = 640
    
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (frame_width, frame_height))
        faces = extract_faces(frame)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            
            # Visual feedback
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, f'{identified_person}', (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
            
            # Mark attendance if not already marked
            if identified_person not in [person['name'] for person in marked_people]:
                add_attendance(identified_person)
                marked_people.append({
                    'name': identified_person,
                    'time': datetime.now().strftime("%I:%M %p")
                })
                cv2.putText(frame, 'New Attendance Marked!', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show list of marked attendances
        y_offset = 60
        cv2.putText(frame, 'Marked Attendances:', (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        for person in marked_people:
            cv2.putText(frame, f"{person['name']} at {person['time']}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30
        
        # Show exit instruction at the bottom
        cv2.putText(frame, 'Press SPACEBAR to exit', (10, frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Place the frame in the correct position on the background
        imgBackground[160:160 + frame_height, 50:50 + frame_width] = frame
        cv2.imshow('Attendance', imgBackground)
        
        # Only exit on spacebar press
        key = cv2.waitKey(1)
        if key == 32:  # 32 is the ASCII code for spacebar
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Get updated attendance after closing camera
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)



@app.route('/add', methods=['GET', 'POST'])
def add():
    try:
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
        
        # Check if user already exists
        if os.path.isdir(userimagefolder):
            return render_template('home.html', mess='User already exists! Please use a different name or ID.')
        
        os.makedirs(userimagefolder)
        
        frame_height = 480
        frame_width = 640
        
        cv2.namedWindow('Adding new User', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Adding new User', cv2.WND_PROP_TOPMOST, 1)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return render_template('home.html', mess='Could not access webcam! Please check your camera.')
        
        i = 0  # Image counter
        start_time = datetime.now()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Check for timeout
            if (datetime.now() - start_time).seconds > MAX_CAPTURE_TIME:
                cap.release()
                cv2.destroyAllWindows()
                # Clean up incomplete registration
                if os.path.exists(userimagefolder):
                    import shutil
                    shutil.rmtree(userimagefolder)
                return render_template('home.html', mess='Registration timed out! Please try again.')
            
            frame = cv2.resize(frame, (frame_width, frame_height))
            faces = extract_faces(frame)
            
            if len(faces) == 0:
                cv2.putText(frame, 'No face detected', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif len(faces) > 1:
                cv2.putText(frame, 'Multiple faces detected', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                    cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                    
                    # Only capture every 5th frame and if we haven't exceeded the limit
                    if i < nimgs:
                        name = f'{newusername}_{i}.jpg'
                        face_img = frame[y:y+h, x:x+w]
                        cv2.imwrite(os.path.join(userimagefolder, name), face_img)
                        i += 1
                    
                    if i >= nimgs:
                        # Show completion message with countdown
                        remaining_time = 3
                        while remaining_time > 0:
                            completion_frame = frame.copy()
                            cv2.putText(completion_frame, 'Registration Complete!', (30, 60), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(completion_frame, f'Closing in {remaining_time}s...', (30, 100),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            imgBackground[160:160 + frame_height, 50:50 + frame_width] = completion_frame
                            cv2.imshow('Adding new User', imgBackground)
                            cv2.waitKey(1000)  # Wait for 1 second
                            remaining_time -= 1
                        break
            
            imgBackground[160:160 + frame_height, 50:50 + frame_width] = frame
            cv2.imshow('Adding new User', imgBackground)
            
            key = cv2.waitKey(1)
            if key == 27 or i >= nimgs:  # Exit on ESC or when enough images captured
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Verify we have enough images
        captured_images = len(os.listdir(userimagefolder))
        if captured_images < nimgs:
            # Clean up incomplete registration
            import shutil
            shutil.rmtree(userimagefolder)
            return render_template('home.html', mess=f'Registration failed! Only captured {captured_images} images. Please try again.')
        
        print('Training Model')
        train_model()
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)
        
    except Exception as e:
        # Clean up on error
        if 'userimagefolder' in locals() and os.path.exists(userimagefolder):
            import shutil
            shutil.rmtree(userimagefolder)
        return render_template('home.html', mess=f'An error occurred: {str(e)}')

# Add a new route for real-time attendance updates
@app.route('/get_attendance')
def get_attendance():
    names, rolls, times, l = extract_attendance()
    return jsonify({
        'names': names.tolist(),
        'rolls': rolls.tolist(),
        'times': times.tolist(),
        'length': l
    })

# Add this new route after other routes
@app.route('/clear_attendance', methods=['POST'])
def clear_attendance():
    try:
        # Create a new empty attendance file for today
        with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
            f.write('Name,Roll,Time')
        
        # Return updated empty attendance
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', 
                             names=names, 
                             rolls=rolls, 
                             times=times, 
                             l=l, 
                             totalreg=totalreg(), 
                             datetoday2=datetoday2,
                             mess='Attendance cleared successfully!')
    except Exception as e:
        return render_template('home.html', 
                             mess=f'Error clearing attendance: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
