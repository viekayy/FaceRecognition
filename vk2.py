#!/usr/bin/env python
# coding: utf-8

# In[6]:


pip install matplotlib


# In[7]:


get_ipython().system('pip install tensorflow-gpu ')


# In[8]:


pip install opencv-python 


# In[1]:


import cv2
import os
import random
import numpy as np

from tkinter import *
from tkinter import ttk
import tkinter.messagebox


import cv2
import numpy as np
import os


# from PIL import ImageGrab
# import the modules
import os
from os import listdir
from tkinter import filedialog
# from PIL import ImageGrab
# import the modules
import os
from os import listdir
import face_recognition as model
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
print("done")


# In[204]:


#import tensorflow dependency - functional apis ko include kiya
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense , MaxPooling2D , Input , Flatten
import tensorflow as tf
print('done')


# In[3]:


Model(inputs=[inputimage, verificationimage] ,outputs=[1,0])


# In[205]:


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)


# In[206]:


for gpu in gpus:
    print(gpus)


# In[207]:


gpus


# In[208]:


POS_PATH = os.path.join('data','positive')
NEG_PATH = os.path.join('data','negative')
ANC_PATH = os.path.join('data','anchor')
ANC_PATH


# In[209]:


NEG_PATH


# In[4]:


VK_PATH = os.path.join('data','VK')
VK_PATH


# In[5]:


POS_PATH


# In[58]:


#making directories
os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)


# In[106]:


os.makedirs(VK_PATH)


# In[14]:


#http://vis-www.cs.umass.edu/lfw/
#uncompress 
get_ipython().system('tar -xf lfw.tgz')


# In[15]:


#move ifw images to following repo data/negative
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw',directory)):
        EX_PATH = os.path.join('lfw',directory,file)
        NEW_PATH = os.path.join(NEG_PATH,file)
        os.replace(EX_PATH,NEW_PATH)


# In[18]:


for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw',directory)):
        print(os.path.join('lfw',directory,file))
        print(os.path.join(NEG_PATH,file))
print('hello')


# In[210]:


#uniform unique identity uuid unique names
import uuid
uuid


# In[211]:


os.path.join(ANC_PATH,'{}.jpg'.format(uuid.uuid1()))


# In[99]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    
    ret, frame = cap.read()
    #cutting frame into 250 X 250 pixel
    frame =  frame[120:120+250,200:200+250,:]
  
    #collect anchors
    if cv2.waitKey(1) & 0XFF == ord('a'):
     #creating path
        imgname = os.path.join(ANC_PATH,'{}.jpg'.format(uuid.uuid1()))
        #write in anchor image
        cv2.imwrite(imgname, frame)
        
    #collect positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
         #creating path
        imgname = os.path.join(POS_PATH,'{}.jpg'.format(uuid.uuid1()))
        #write in positive image
        cv2.imwrite(imgname, frame)
        
    if cv2.waitKey(1) & 0XFF == ord('k'):
         #creating path
        imgname = os.path.join(VK_PATH,'{}.jpg'.format(uuid.uuid1()))
        #write in positive image
        cv2.imwrite(imgname, frame)
        
    cv2.imshow('Image Collection',frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[212]:


#pipeline
anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(300)
print("done")


# In[213]:


dir_test=anchor.as_numpy_iterator()
print(dir_test.next())


# In[214]:


def preprocess(file_path):
    #image read kr rahe hai
    byte_img=tf.io.read_file(file_path)
    
    #load kr rahe hhai image mai 
    img = tf.io.decode_jpeg(byte_img)
    
    #preprocessing ho rahi hai
    img = tf.image.resize(img,(100,100))
    img = img/255.0
    return img


# In[215]:


dataset.map(preprocess)


# In[216]:


img = preprocess('data\\anchor\\57aec983-3ffa-11ed-985e-287fcfe14550.jpg')


# In[218]:


plt.imshow(img)


# In[219]:


img.numpy().max()


# In[220]:


class_labels = tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))


# In[221]:


iterator_labs = class_labels.as_numpy_iterator()


# In[222]:


iterator_labs.next()


# In[223]:


positives = tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)


# In[224]:


samples = data.as_numpy_iterator()
example = samples.next()
example


# In[225]:


def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)


# In[226]:


res = preprocess_twin(*example)
print("done")


# In[227]:


len(res)


# In[228]:


plt.imshow(res[1])


# In[229]:


#data load krne ja rahe hai pipeline mai
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)


# In[444]:


samples = data.as_numpy_iterator()

samp = samples.next()

plt.imshow(samp[0])


# In[463]:


samples = data.as_numpy_iterator()

samp = samples.next()
plt.imshow(samp[1])


# In[418]:


samp[2]


# In[233]:


#training partition building
train_data=data.take(round(len(data)*.7))
train_data=train_data.batch(16)
train_data=train_data.prefetch(8)


# In[423]:



#train_samples = train_data.as_numpy_iterator()
#train_sample = train_samples.next()
#len(train_sample[0])


# In[234]:


#testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


# In[235]:


data


# In[236]:


#inp = Input(shape=(105,105,3))
#inp


# In[238]:


#c1 = Conv2D(64,(10,10), activation='relu')(inp)
#c1


# In[239]:


# m1 = MaxPooling2D(64,(2,2),padding='same')(c1)
#m1


# In[240]:


#second block
   # c2 = Conv2D(128,(7,7),activation = 'relu')(m1)
   # m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
   # m2


# In[241]:


#c3 = Conv2D(128,(4,4), activation='relu')(m2)
#m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
#m3


# In[242]:


#c4 = Conv2D(256, (4,4), activation='relu')(m3)
#f1 = Flatten()(c4)
#d1 = Dense(4096, activation='sigmoid')(f1)
#m3
#c4


# In[243]:


#mod = Model(inputs=[inp],outputs=[d1], name='embedding')


# In[244]:


#mod.summary()


# In[474]:


def make_embedding():
    inp = Input(shape=(100,100,3) , name = 'input image')
    
    #first block
    c1 = Conv2D(64,(10,10),activation='relu')(inp)
    m1 = MaxPooling2D(64,(2,2),padding='same')(c1)
    
    #second block
    c2 = Conv2D(128,(7,7),activation = 'relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    #Third block
    c3 = Conv2D(128,(4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    #forth final embedding block
    c4 = Conv2D(256, (6,6), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    return Model(inputs=[inp],outputs=[d1], name='embedding')


# In[475]:


Model(inputs=[inp],outputs=[d1], name='embedding')


# In[476]:


embedding = make_embedding()


# In[477]:


embedding.summary()


# In[478]:


#Saimese L1 Distance Class
class L1Dist(Layer):
    #init methof - inheritance
    def __init__(self, **kwargs):
        super().__init__()
    #tagdi calculation - similaity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


# In[250]:


l1 = L1Dist()
l1(anchor_embedding, validation_embedding)


# In[479]:


input_image = Input(name='input_img' , shape=(100,100,3))


# In[480]:


validation_image = Input(name='validation_img', shape=(100,100,3))
   


# In[481]:


inp_embedding = embedding(input_image)
val_embedding = embedding(validation_image)


# In[482]:


val_embedding


# In[483]:


siamese_layer = L1Dist()


# In[484]:


distances = siamese_layer(inp_embedding, val_embedding)


# In[485]:


classifier = Dense(1, activation ='sigmoid')(distances)


# In[486]:


classifier


# In[487]:


siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


# In[488]:


siamese_network.summary()


# In[489]:


def make_siamese_model():
    # Handle inputs
    #anchor image in network
    input_image = Input(name='input_img' , shape=(100,100,3))
    
    #validation image in network
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    #Combine saimese distance components
    saimese_layer = L1Dist()
    saimese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    #classification layer
    classifier = Dense(1, activation ='sigmoid')(distances)
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


# In[490]:


siamese_model  = make_siamese_model()


# In[491]:


siamese_model.summary()


# In[492]:


#Setup Loss and Optimizer
binary_cross_loss = tf.losses.BinaryCrossentropy()


# In[493]:


opt=tf.keras.optimizers.Adam(1e-4) # 0.0001


# In[494]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


# In[495]:


test_batch = train_data.as_numpy_iterator()


# In[496]:


batch_1 = test_batch.next()


# In[497]:


X = batch_1[:2]
X


# In[498]:


y = batch_1[2]
y


# In[499]:


np.array(X).shape


# In[500]:


@tf.function
def train_step(batch):
    
    #record operations
    with tf.GradientTape() as tape:
        
        #Get anchor and positive/negative image
        X = batch[:2]
        
        #Get label
        
        y = batch[2]
        #Forward pass
        yhat = siamese_model(X, training=True)
        #calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)
    #calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    #calculate  updated weights and apply to siameese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    return loss
#print('done')


# In[501]:


def train(data, EPOCHS):
    #loops through ephos
    for epoch in range (1,EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        for idx, batch in enumerate(data):
            train_step(batch)
            progbar.update(idx+1)
        
            #save checkpoint
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
    #loop through each batch
    #run train step here


# In[502]:


EPOCHS = 50


# In[503]:


train(train_data,EPOCHS)


# train(train_data,EPOCHS)

# In[504]:


#Import metric calculations
from tensorflow.keras.metrics import Precision, Recall


# In[505]:


test_input, test_val, y_true = test_data.as_numpy_iterator().next()


# In[506]:


y_hat = siamese_model.predict([test_inpu/t, test_val])
y_hat


# In[507]:


#post processing the ressults
[1 if prediction > 0.5 else 0 for prediction in y_hat]


# In[508]:


y_true


# In[509]:


m = Recall()

m.update_state(y_true, y_hat)

m.result().numpy()
#if k  > 0.5:
#        print("1")
#else:
#        print("0")


# In[510]:


m = Precision()

m.update_state(y_true, y_hat)

m.result().numpy()


# In[511]:


plt.figure(figsize=(10,8))

plt.subplot(1,2,1)
plt.imshow(test_input[4])

plt.subplot(1,2,2)
plt.imshow(test_val[1])
plt.show()


# In[512]:


siamese_model.save('siamesemodel.h5')


# In[513]:


L1Dist


# In[514]:


model = tf.keras.models.load_model('siamesemodel.h5',
                            custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})


# In[319]:


model.predict([test_input, test_val])


# In[320]:


model.summary()


# In[358]:


for image in os.listdir(os.path.join('application_data','verification_images')):
    validation_img = preprocess(os.path.join('application_data', 'verification_images',image))
    print(validation_img)


# In[515]:


##verification method
def verify(model , detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(os.path.join('application_data','verification_images')):
        input_img = preprocess(os.path.join('application_data','input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images',image))
        
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
        
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold
    return results, verified
    #test positive = +50% , detection threshold
    #verification threshold +ve/total


# In[1]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame =  frame[120:120+250,200:200+250,:]
    cv2.imshow('Verification', frame)
    if cv2.waitKey(10) & 0xFF == ord('v'):
        #save image input folder
        cv2.imwrite(os.path.join('application_data','input_image', 'input_image.jpg'), frame)
        results, verified = verify(model, 0.5, 0.5)
        print(verified)
        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[419]:


np.sum(np.squeeze(results) > 0.5)


# In[ ]:


from tkinter import *
from tkinter import ttk
import tkinter.messagebox
from tkinter import filedialog
Model=r"C:\Users\91705\Desktop\capstone_2\App\siamesemodel.h5"

# get the path or directory
# folder_dir = "C:\Users\91705\Desktop\vk\videshwar\videshwar\Training_images"
# for images in os.listdir(folder_dir):

# 	# check if the image ends with png or jpg or jpeg
# 	if (images.endswith(".png") or images.endswith(".jpg")\
# 		or images.endswith(".jpeg")):
# 		# display
# 		print(images)



path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
# print(myList)





for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
# print("Trained successfully classNames")


def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = model.face_encodings(img)[0]
        encodeList.append(encode)
        # break;
    return encodeList




def isfound(name,faceDis,matchIndex):
    
    # print('The person is identified as :- ',name, " " , matchIndex)
    
    cv2.imshow(name, images[matchIndex])
    
    cv2.waitKey(1)
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Starting please wait')

# cap = cv2.VideoCapture(0)
flag = True
import tkinter
root = Tk()


li = []
def browseFiles():
    
    
    filenamee = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("all files",
                                                        "*.*"),
                                                       ("all files",
                                                        "*.*")))
    location  = filenamee
    li.append(location)
#     li.append("hiii")
  #  print("selected 1" , location)
  #  print("list", li)
# print("selected" , location)
  




def ans():
    flag = True
    cnt=0
    
    while flag:
        if cnt==len(images):
            tkinter.messagebox.showinfo("USER NOT FOUND")
            flag=False
            continue
        cnt+=1
        var = li[len(li)-1]+""
#         print(var)
    #     success
    #     img = cap.read()
    # img = captureScreen()
    
        # print(li)
        img = cv2.imread(var,cv2.IMREAD_COLOR)
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = model.face_locations(imgS)
        encodesCurFrame = model.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = model.compare_faces(encodeListKnown, encodeFace)
            faceDis = model.face_distance(encodeListKnown, encodeFace)
           # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
               # print('The Sketch matching process was successfull and it was matched with ',matchIndex)
                isfound(name,faceDis,matchIndex)
                flag = False

# def teacher():
    # pass
    # frame.quit
#     deleteAndCrt()
    # tecr()


# def res():
#     # pass
# #     result()

def showmain():

    root.title('Main Window ')
    root.geometry("800x600")
    root.resizable(width=False,height=False)
    root.title("sketch face recognition")
    # f0 = Frame(root,width= 100,height=100)
    # f0.pack()
    # f0.place(anchor=CENTER,relx=5,rely=5)
    # img = ImageTK.PhotoImage(Image.open("icon.png"))
    # lab = Label(f0,image=img)
    # lab.pack()
    f1 = Frame(root,bg="black", borderwidth=4, relief=SUNKEN)
    f1.pack(side=BOTTOM, fill="y")
    value=StringVar()
    bg = PhotoImage(file = "background.png",height= 600, width=800)
    label1 = Label( root,image = bg)
    label1.place(x=0,y=0)
    p1 = PhotoImage(file = 'icon.png')
    root.iconphoto(False, p1)
    lbl = ttk.Label(root, text=" SKETCH FACE RECOGNITION ", font="comicsansms 13 bold", padding='10')
    lbl.pack()
    b1=Button(text="Pick Sketch",command=browseFiles,height= 2, width=100,bg="white", borderwidth=2)    
    b2=Button(text="Check Result",command=ans,height= 2, width=100,bg="green", borderwidth=2)
    b2.pack(side= BOTTOM)
    b1.pack(side = BOTTOM)
    
    
    
    
    
    root.mainloop()


showmain()


# In[13]:





# In[23]:


#import modules
 
from tkinter import *
import os
 
# Designing window for registration
 
def register():
    global register_screen
    register_screen = Toplevel(main_screen)
    register_screen.title("Register")
    register_screen.geometry("300x250")
 
    global username
    global password
    global username_entry
    global password_entry
    username = StringVar()
    password = StringVar()
 
    Label(register_screen, text="Please enter details below", bg="blue").pack()
    Label(register_screen, text="").pack()
    username_lable = Label(register_screen, text="Username * ")
    username_lable.pack()
    username_entry = Entry(register_screen, textvariable=username)
    username_entry.pack()
    password_lable = Label(register_screen, text="Password * ")
    password_lable.pack()
    password_entry = Entry(register_screen, textvariable=password, show='*')
    password_entry.pack()
    Label(register_screen, text="").pack()
    Button(register_screen, text="Register", width=10, height=1, bg="blue", command = register_user).pack()
 
 
# Designing window for login 
 
def login():
    global login_screen
    login_screen = Toplevel(main_screen)
    login_screen.title("Login")
    login_screen.geometry("300x250")
    Label(login_screen, text="Please enter details below to login").pack()
    Label(login_screen, text="").pack()
 
    global username_verify
    global password_verify
 
    username_verify = StringVar()
    password_verify = StringVar()
 
    global username_login_entry
    global password_login_entry
 
    Label(login_screen, text="Username * ").pack()
    username_login_entry = Entry(login_screen, textvariable=username_verify)
    username_login_entry.pack()
    Label(login_screen, text="").pack()
    Label(login_screen, text="Password * ").pack()
    password_login_entry = Entry(login_screen, textvariable=password_verify, show= '*')
    password_login_entry.pack()
    Label(login_screen, text="").pack()
    Button(login_screen, text="Login", width=10, height=1, command = login_verify).pack()
 
# Implementing event on register button
 
def register_user():
 
    username_info = username.get()
    password_info = password.get()
 
    file = open(username_info, "w")
    file.write(username_info + "\n")
    file.write(password_info)
    file.close()
 
    username_entry.delete(0, END)
    password_entry.delete(0, END)
 
    Label(register_screen, text="Registration Success", fg="green", font=("calibri", 11)).pack()
 
# Implementing event on login button 
 
def login_verify():
    username1 = username_verify.get()
    password1 = password_verify.get()
    username_login_entry.delete(0, END)
    password_login_entry.delete(0, END)
 
    list_of_files = os.listdir()
    if username1 in list_of_files:
        file1 = open(username1, "r")
        verify = file1.read().splitlines()
        if password1 in verify:
            login_sucess()
 
        else:
            password_not_recognised()
 
    else:
        user_not_found()
 
# Designing popup for login success
 
def login_sucess():
    global login_success_screen
    login_success_screen = Toplevel(login_screen)
    login_success_screen.title("Success")
    login_success_screen.geometry("150x100")
    Label(login_success_screen, text="Login Success").pack()
    Button(login_success_screen, text="OK", command=delete_login_success).pack()
 
# Designing popup for login invalid password
 
def password_not_recognised():
    global password_not_recog_screen
    password_not_recog_screen = Toplevel(login_screen)
    password_not_recog_screen.title("Success")
    password_not_recog_screen.geometry("150x100")
    Label(password_not_recog_screen, text="Invalid Password ").pack()
    Button(password_not_recog_screen, text="OK", command=delete_password_not_recognised).pack()
 
# Designing popup for user not found
 
def user_not_found():
    global user_not_found_screen
    user_not_found_screen = Toplevel(login_screen)
    user_not_found_screen.title("Success")
    user_not_found_screen.geometry("150x100")
    Label(user_not_found_screen, text="User Not Found").pack()
    Button(user_not_found_screen, text="OK", command=delete_user_not_found_screen).pack()
 
# Deleting popups
 
def delete_login_success():
    login_success_screen.destroy()
 
 
def delete_password_not_recognised():
    password_not_recog_screen.destroy()
 
 
def delete_user_not_found_screen():
    user_not_found_screen.destroy()
 
 
# Designing Main(first) window
 
def main_account_screen():
    global main_screen
    main_screen = Tk()
    main_screen.geometry("300x250")
    main_screen.title("Account Login")
    Label(text="Select Your Choice", bg="blue", width="300", height="2", font=("Calibri", 13)).pack()
    Label(text="").pack()
    Button(text="Login", height="2", width="30", command = login).pack()
    Label(text="").pack()
    Button(text="Register", height="2", width="30", command=register).pack()
 
    main_screen.mainloop()
 
 
main_account_screen()


# In[ ]:




