import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2


WINDOW_SIZE_X = 500
WINDOW_SIZE_Y = 500

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)

BOUNDARY = 7
IMAGESV = False

# Check command-line arguments
if len(sys.argv) != 2:
    sys.exit("Usage: python recognition.py model")

# load model
model = tf.keras.models.load_model(sys.argv[1])

# for digit identification
LABELS = {
    0: "ZERO (0)",
    1: "ONE (1)",
    2: "TWO (2)",
    3: "THREE (3)",
    4: "FOUR (4)",
    5: "FIVE (5)",
    6: "SIX (6)",
    7: "SEVEN (7)",
    8: "EIGHT (8)",
    9: "NINE (9)"
}

# initialize our pygame
pygame.init()

SCREEN = pygame.display.set_mode((WINDOW_SIZE_X, WINDOW_SIZE_Y))
pygame.display.set_caption("Handwritten Digit Recognition")

iswriting = False

# append the coordinates where the writing is done on screen
number_x = []
number_y = []

imageCount = 1

PREDICT = True

while True:
    # to run the gui till exited
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    # to be able to draw on SCREEN
    if event.type == MOUSEMOTION and iswriting:
        X_coor, Y_coor = event.pos
        # last 2 parameters: thickness, brightness
        pygame.draw.circle(SCREEN, WHITE, (X_coor, Y_coor), 3, 0)
        number_x.append(X_coor)
        number_y.append(Y_coor)

    # start writing
    if event.type == MOUSEBUTTONDOWN:
        iswriting = True

    # to send the drawing of digit for recognition "enter" to perform recognition
    if event.type == KEYDOWN and event.unicode == 13:
        iswriting = False
        #sorting list
        number_x = sorted(number_x)
        number_y = sorted(number_y)

        # to draw rectangle around
        rect_minx, rect_maxx = max(number_x[0] - BOUNDARY, 0), min(number_x[-1] + BOUNDARY, WINDOW_SIZE_X)
        rect_miny, rect_maxy = max(number_y[0] - BOUNDARY, 0), min(number_y[-1] + BOUNDARY, WINDOW_SIZE_Y)

        # reinit the number coordinate array now since we are done for one digit
        number_x = []
        number_y = []

        # extract the written digit from the rectangle - pixels in the rect
        # take transpose of img_arr in float type
        img_arr = np.array(pygame.PixelArray(SCREEN))[rect_minx:rect_maxx, rect_miny: rect_maxy].T.astype(np.float32)

        if IMAGESV:
            cv2.imwrite("image.png")
            imageCount += 1 # to have a batch size

        # incorporate with the ML
        if PREDICT:
            # model trained with (28, 28)
            image = cv2.resize(img_arr (28, 28))
            # add padding to avoid losing on any information by chance
            image = np.pad(10,10, 'constant', constant_values = 0)
            # white ink value : 255
            image = cv2.resize(image, (28, 28))/255


            # prediction
            # index with highest probability distribution
            label = str(LABELS(np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))))

            textSurface = FONT.render(label, True, YELLOW, WHITE)
            textRect = testing.get_rect()
            # providing coordinates
            textRect.left, textRect.bottom = rect_minx, rect_maxy

            # blit for display
            SCREEN.blit(textSurface, textRect)

            # clear screen
            if event.type == KEYDOWN:
                if event.unicode == "c":
                    SCREEN.fill(BLACK)

    pygame.update()
