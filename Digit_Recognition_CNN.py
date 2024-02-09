# import the mnist dataset from keras
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()# This line imports the mnist dataset from keras, it consists of train_images, train_labels, test_images and test_labels

# reshape the train images to include an additional dimension for a single channel
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)

print(f"Training Images\n{train_images}")
# This line reshape the train images to include an additional dimension for a single channel,
# the first parameter is the number of images, and the last parameter is the number of channels
# the values of 28 are the size of the image 28x28

# reshape the test images to include an additional dimension for a single channel
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1) # This line reshape the test images to include an additional dimension for a single channel, the first parameter is the number of images, and the last parameter is the number of channels
print(f"Test Images\n{test_images}")

# return the modified images
print("Train Images shape: ",train_images.shape)
print("Test Images shape: ",test_images.shape)

ef preprocess_images(imgs): # should work for both a single image and multiple images
    # check if the shape of the input image is 2D, and if so, assign it to `sample_img`
    sample_img = imgs if len(imgs.shape) == 2 else imgs[0]
    print(type(imgs.shape))
    print(sample_img.shape)
    # assert that the shape of the image is 28x28 and single-channel (grayscale)
    assert sample_img.shape in [(28, 28, 1), (28, 28)], sample_img.shape
    # normalize the image by dividing each pixel by 255.0
    return imgs / 255.0

# preprocess the train images
train_images = preprocess_images(train_images)
print(f"Training Images\n{train_images}")

# preprocess the test images
test_images = preprocess_images(test_images)
print(f"Test Images\n{test_images}")

# create a figure with a specific size
plt.figure(figsize=(20,4))
# loop through the first 10 images in the train dataset
for i in range(10):
    # add a subplot to the figure
    plt.subplot(1,10,i+1)
    # remove the x-axis ticks
    plt.xticks([])
    # remove the y-axis ticks
    plt.yticks([])
    # remove grid
    plt.grid(False)
    # display the image
    plt.imshow(train_images[i].reshape(28, 28), cmap=plt.cm.binary)# This line displays the image on the subplot, reshaping it to 28 by 28 pixels and using the 'binary' colormap
    # add the label of the image to the x-axis
    plt.xlabel(train_labels[i])
	
# create a sequential model
model = keras.Sequential()

# add a 2D convolution layer with 32 filters of size 3x3 and relu activation
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
# add a 2D convolution layer with 64 filters of size 3x3 and relu activation
model.add(Conv2D(64, (3, 3), activation='relu'))
# add a 2D pooling layer to choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# add a dropout layer to randomly turn neurons on and off to improve convergence
# model.add(Dropout(0.25))
# flatten the data since too many dimensions, we only want a classification output
model.add(Flatten())
# add a fully connected layer with 128 neurons and relu activation
model.add(Dense(128, activation='relu'))
# add a dropout layer to randomly turn neurons on and off to improve convergence
# model.add(Dropout(0.5))
# add a final output layer with 10 neurons and softmax activation
model.add(Dense(10, activation='softmax'))

# print the summary of the model
model.summary()

#compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              # set the optimizer as Adam with a learning rate of 0.1
              loss='sparse_categorical_crossentropy',
              # use sparse categorical crossentropy as the loss function
              metrics=['accuracy'])
              # track the accuracy metric during training

#extra code to use tensorboard if wanted
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
################################

# train the model
history = model.fit(train_images, train_labels, epochs=1, batch_size=64, callbacks=[tensorboard_callback])
# set the number of epochs to 1 and the batch size to 64


#print the shape of test images
print(test_images.shape)

#evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)

#print the test accuracy
print('Test accuracy:', test_acc)


# Define the path to the sample video file
mnist_dream_path = 'https://github.com/strathpaulkirkland/DM996/raw/master/mnist_dream.mp4'
# Define the path to save the predicted video file
mnist_prediction_path = 'mnist_dream_predicted.mp4'

# Check if the file already exists, if not download the sample video
if not os.path.isfile(mnist_dream_path):
    print('downloading the sample video...')
    vid_url = mnist_dream_path
    # Download the video file from the url and save it to the defined path
    mnist_dream_path = urllib.request.urlretrieve(vid_url)[0]

# Define a function to display an image using IPython library
def cv2_imshow(img):
    # Encode the image to .png format
    ret = cv2.imencode('.png', img)[1].tobytes()
    # Create an IPython image object
    img_ip = IPython.display.Image(data=ret)
    # Display the image
    IPython.display.display(img_ip)

# Open a video capture object using the video file path
cap = cv2.VideoCapture(mnist_dream_path)
# Initialize the variable vw (used for debugging if needed)
vw = None
# Initialize the variable frame to -1 (used for debugging if needed)
frame = -1


######################################

while True: # This line creates an infinite loop that will continue to execute the code within it until a specific condition is met (in this case, a break statement is used later in the code)
    frame += 1 # This line increments the value of the "frame" variable by 1 on each iteration of the loop
    ret, img = cap.read() # This line reads a frame from the video capture object "cap" and assigns the returned values to the variables "ret" and "img"
    if not ret: break # This line checks the value of the "ret" variable and, if it is False, breaks out of the loop (ending the infinite loop)
    assert img.shape[0] == img.shape[1] # This line asserts that the height and width of the image are equal (i.e. the image is a square)
    if img.shape[0] != 720: # This line checks if the height of the image is not equal to 720
        img = cv2.resize(img, (720, 720)) # If the height is not equal to 720, the image is resized to a height and width of 720 using the "cv2.resize" function

#preprocess the image for prediction
    img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # This line converts the image from BGR color space to grayscale using the "cv2.cvtColor" function
    img_proc = cv2.resize(img_proc, (28, 28)) # This line resizes the image to a width and height of 28 pixels using the "cv2.resize" function
    img_proc = preprocess_images(img_proc) #This line applies preprocessing function on the image
    img_proc = 1 - img_proc # This line inverses the image, since the training dataset is white text with black background
    net_in = np.expand_dims(img_proc, axis=0) # This line adds an extra dimension to the image array to specify a batch size of 1 using numpy expand_dims function
    net_in = np.expand_dims(net_in, axis=3) # This line adds an extra dimension to the image array to specify the number of channels using numpy expand_dims function
    preds = model.predict(net_in)[0] # This line uses the trained model to make a prediction on the preprocessed image
    guess = np.argmax(preds) # This line assigns the index of the highest predicted probability to the "guess" variable
    perc = np.rint(preds * 100).astype(int) # This line rounds the predicted probabilities to the nearest integer and converts them to integers


    img = 255 - img # this line inverts the color of the image by subtracting all pixel values from 255
    pad_color = 0 # this line sets the padding color to be black (0)
    img = np.pad(img, ((0,0), (0,1280-720), (0,0)), mode='constant', constant_values=(pad_color))
    # this line pads the image with the pad_color (black) on the right side to make the image width 1280 pixels

    line_type = cv2.LINE_AA # this line sets the line type for the text to be anti-aliased
    font_face = cv2.FONT_HERSHEY_SIMPLEX # this line sets the font face to be simplex
    font_scale = 1.3 # this line sets the font scale to be 1.3
    thickness = 2 # this line sets the thickness of the text to be 2
    x, y = 740, 60 # this line sets the x and y coordinates for the text to be placed at (740, 60)
    color = (255, 255, 255) # this line sets the color of the text to be white

    text = "Neural Network Output:" # this line sets the text to be written as "Neural Network Output:"
    cv2.putText(img, text=text, org=(x, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
                        color=color, lineType=line_type)
    # this line writes the text "Neural Network Output:" on the image at the specified x and y coordinates with the specified font, font scale, thickness, color and line type.

    text = "Input:" # this line sets the text to be written as "Input:"
    cv2.putText(img, text=text, org=(30, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
                        color=color, lineType=line_type)
    # this line writes the text "Input:" on the image at the specified x, y coordinates with the specified font, font scale, thickness, color and line type.

    y = 130 # this line sets the y coordinate for the next set of text and rectangles to be written at 130
    for i, p in enumerate(perc):
        if i == guess: color = (255, 218, 158)
        else: color = (100, 100, 100)

        rect_width = 0
        if p > 0: rect_width = int(p * 3.3)

        rect_start = 180
        cv2.rectangle(img, (x+rect_start, y-5), (x+rect_start+rect_width, y-20), color, -1)
        # this line draws a rectangle on the image starting at (x+rect_start, y-5) and ending at (x+rect_start+rect_width, y-20) with the specified color and filled in.

        text = '{}: {:>3}%'.format(i, int(p))
        cv2.putText(img, text=text, org=(x, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
                        color=color, lineType=line_type)
        # this line writes the text '{}: {:>3}%'.format(i, int(p)) on the image at the specified x, y coordinates with the specified font, font scale, thickness, color and line type.
        y += 60
      # this line increments the y coordinate by 60 for the next set of text and rectangles

    # if you don't want to save the output as a video, set this to False
    save_video = True

    if save_video:
        if vw is None:
            codec = cv2.VideoWriter_fourcc(*'MP4V')
            vid_width_height = img.shape[1], img.shape[0]
            vw = cv2.VideoWriter('mnist_dream_predicted.mp4', codec, 30, vid_width_height)
        # 15 fps above doesn't work robustly so we right frame twice at 30 fps
        vw.write(img)
        vw.write(img)

    # scale down image for display
    img_disp = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    cv2_imshow(img_disp)
    IPython.display.clear_output(wait=True)

        #releasing all the stored information such as the frame to allow the process to move on
cap.release()
if vw is not None:
    vw.release()
	
#Videos in Collab are a pain to play but this is a workaround

from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = "/content/mnist_dream_predicted.mp4"

# Compressed video path
compressed_path = "/content/mnist_dream_predicted_C.mp4"

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)


