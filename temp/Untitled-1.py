import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def segment_and_plot_characters(image_path: str):
    #Reading image
    image = cv2.imread(image_path)
  
    y_start = 31
    y_end = 171
    x_start = 122
    x_end = 758
    image = image[y_start:y_end, x_start:x_end]


    #Pre-process cropped image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    #Finding contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Filtering contours and extract characters
    char_boxes = []
    image_height = image.shape[0]

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        
        if h == 0:
            continue

        aspect_ratio = w / h
        
        is_character_height = h > 0.4 * image_height
        is_character_aspect_ratio = aspect_ratio < 1.2 # A character is not very wide

        if is_character_height and is_character_aspect_ratio:
            char_boxes.append((x, y, w, h))

    #Sorting characters from left to right
    char_boxes = sorted(char_boxes, key=lambda box: box[0])

    #Ploting segmented characters
    num_chars = len(char_boxes)
    print(num_chars)
    
    fig, axes = plt.subplots(1, num_chars, figsize=(num_chars * 1.5, 1.5))
    
    if num_chars == 1:
        axes = [axes]

    #Saving segmented characters into ideal folder... (optional)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'ideal')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving images to: {output_dir}")

    for i, box in enumerate(char_boxes):
        x, y, w, h = box
        char_image = gray[y:y+h, x:x+w]

        save_path = os.path.join(output_dir, f"{i}.jpg")
        cv2.imwrite(save_path, char_image)

        ax = axes[i]
        ax.imshow(char_image, cmap='gray')
        ax.axis('off')

    plt.suptitle("Segmented Characters")
    plt.show()



script_dir = os.path.dirname(os.path.abspath(__file__))
image_file_path = os.path.join(script_dir, 'ideal/p1.jpg')

segment_and_plot_characters(image_file_path)