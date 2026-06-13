import os
import cv2
import torch
import joblib
import pandas as pd
import numpy as np
import torch.nn as nn
import tensorflow as tf
from datetime import date
from datetime import datetime
import torchvision.transforms as transforms
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from flask import Flask, request, render_template, Response, redirect, url_for

app = Flask(__name__)

nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Define the discriminator model
discriminator = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    nn.Flatten(),
    nn.Sigmoid()
)

# Load the saved discriminator model onto CPU
discriminator.load_state_dict(torch.load(r'C:\Users\Ajaz\Pictures\FYP\SPOOFIFY_FYP\D2.pth', map_location=torch.device('cpu')))
discriminator.eval()  # Set discriminator to evaluation mode

# Define a function to preprocess frames from the camera feed
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    return transform(frame)

# Load pre-trained face detector model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function to perform spoofing detection
def is_spoofed(frame, save_path=None):
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    input_tensor = preprocessed_frame.unsqueeze(0)

    # Pass the preprocessed frame through the discriminator
    with torch.no_grad():
        output = discriminator(input_tensor)

    probability_real = output.item()
    is_real = probability_real < 0.5

    if is_real and save_path:
        cv2.imwrite(save_path, frame)

    return is_real

# Define the video frame generator function
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Perform spoofing detection
        is_real = is_spoofed(frame)

        # Display the classification result on the frame
        if is_real:
            cv2.putText(frame, "Real", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Spoofed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

# Helper function to create pairs of images
def create_pairs(faces, labels):
    pairs = []
    pair_labels = []
    
    unique_labels = np.unique(labels)
    label_to_images = {label: [] for label in unique_labels}
    
    # Group images by label
    for img, label in zip(faces, labels):
        label_to_images[label].append(img)
    
    for label in unique_labels:
        positive_images = label_to_images[label]
        negative_labels = [l for l in unique_labels if l != label]
        
        # Create positive pairs
        for i in range(len(positive_images) - 1):
            pairs.append([positive_images[i], positive_images[i + 1]])
            pair_labels.append(1)  # Same person
        
        # Create negative pairs
        for i in range(len(positive_images)):
            neg_label = np.random.choice(negative_labels)
            negative_image = np.random.choice(label_to_images[neg_label])
            pairs.append([positive_images[i], negative_image])
            pair_labels.append(0)  # Different person
    
    return np.array(pairs), np.array(pair_labels)

# Define the Siamese network
def build_siamese_model(input_shape):
    # Base network
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(128)(x)
    
    base_network = Model(inputs, outputs)
    
    # Inputs for the pairs
    input_a = layers.Input(input_shape)
    input_b = layers.Input(input_shape)
    
    # Process the pairs with the base network
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # L2 distance between embeddings
    distance = layers.Lambda(lambda tensors: tf.math.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
    
    # Final prediction layer
    outputs = layers.Dense(1, activation='sigmoid')(distance)
    
    siamese_net = Model([input_a, input_b], outputs)
    
    return siamese_net

# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# Training function for Siamese network
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face)
            labels.append(user)
    
    faces = np.array(faces)
    labels = np.array(labels)
    
    # Create pairs for training
    pairs, pair_labels = create_pairs(faces, labels)
    
    # Split into training and validation sets
    (train_pairs, val_pairs, train_labels, val_labels) = train_test_split(pairs, pair_labels, test_size=0.2)
    
    # Build the Siamese network
    input_shape = (50, 50, 3)
    siamese_net = build_siamese_model(input_shape)
    
    # Compile the model
    siamese_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    siamese_net.fit(
        [train_pairs[:, 0], train_pairs[:, 1]],
        train_labels,
        validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
        epochs=10,
        batch_size=32
    )
    
    # Save the model
    siamese_net.save('static/face_recognition_siamese_model.h5')



# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


## A function to get names and rol numbers of all users
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


## A function to delete a user folder 
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)

# Route for the spoofing checker page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_spoofing', methods=['GET', 'POST'])
def detect_spoofing():
    if request.method == 'POST':
        # Handle the POST request
        # Capture a frame
        cap = cv2.VideoCapture(0)
        success, frame = cap.read()
        cap.release()

        save_path = 'static/spoof_checks/captured_face.jpg'
        
        # Perform spoofing detection on the captured frame
        is_real = is_spoofed(frame, save_path=save_path)

        print("Is Real:", is_real)  # Debugging statement

        if is_real:
            # Redirect to the home page if real
            names, rolls, times, l = extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)
        else:
            # Render the spoofing alert page if spoofed
            return render_template('alert.html')
    else:
        # Handle the GET request
        return redirect(url_for('index'))  # Redirect to the index page or another appropriate page

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='The database is empty. Kindly add a new user.')

    # Load the saved image from the spoof check
    saved_image_path = 'static/spoof_checks/captured_face.jpg'
    saved_image = cv2.imread(saved_image_path)
    
    if saved_image is None:
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='No captured image from spoofing check. Please retry.')

    # Extract face from the saved image
    faces = extract_faces(saved_image)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = cv2.resize(saved_image[y:y+h, x:x+w], (50, 50))
        identified_person = identify_face(face.reshape(1, -1))

        if identified_person:
            identified_person = identified_person[0]
            add_attendance(identified_person)
            result_message = f'User {identified_person} is a registered user.'
        else:
            result_message = 'Face not recognized. You are not a registered user.'
    else:
        result_message = 'No face detected in the saved image.'

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=result_message)

# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

if __name__ == '__main__':
    app.run(debug=True)