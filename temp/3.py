import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# SECTION 1: CORE FUNCTIONS (WITH DYNAMIC PADDING)
# ==============================================================================

def segment_characters(image, scale_factor=1.0):
    """
    Segments characters from a license plate image object.
    **UPDATED:** Padding is now scaled dynamically based on the scale_factor.
    """
    if image is None: return []
    
    # --- Dynamic Padding Calculation ---
    # The original padding for a full-resolution image is 5 pixels.
    # This value is scaled down, with a minimum of 1 pixel.
    original_padding = 5
    scaled_padding = int(max(1, original_padding * scale_factor))

    # --- The rest of the function remains similar ---
    orig_y_start, orig_y_end = 31, 141
    orig_x_start, orig_x_end = 115, 727
    y_start, y_end = int(orig_y_start * scale_factor), int(orig_y_end * scale_factor)
    x_start, x_end = int(orig_x_start * scale_factor), int(orig_x_end * scale_factor)
    h_img, w_img = image.shape[:2]
    y_end, x_end = min(y_end, h_img), min(x_end, w_img)
    cropped_image = image[y_start:y_end, x_start:x_end]
    if cropped_image.size == 0: return []
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_boxes = []
    image_height = cropped_image.shape[0]
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if h == 0: continue
        aspect_ratio = w / h
        if (h > 0.4 * image_height) and (aspect_ratio < 1.2):
            char_boxes.append((x, y, w, h))
    char_boxes = sorted(char_boxes, key=lambda box: box[0])
    character_images = []
    for box in char_boxes:
        x, y, w, h = box
        char_thresh_tight = thresh[y:y+h, x:x+w]
        # Use the new scaled_padding value here.
        char_thresh_padded = cv2.copyMakeBorder(char_thresh_tight, scaled_padding, scaled_padding, scaled_padding, scaled_padding, cv2.BORDER_CONSTANT, value=0)
        character_images.append({'thresh': char_thresh_padded})
    return character_images

# Other functions (recognize_plate, visualize_specific_comparison, etc.) remain unchanged.
# For completeness, the full script is provided below.

def recognize_plate(character_images, template_dir='numbers'):
    """Recognizes a full license plate from segmented characters."""
    if not character_images: return "", 0.0
    templates = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_template_dir = os.path.join(script_dir, template_dir)
    for filename in os.listdir(full_template_dir):
        if filename.endswith('.jpg'):
            char_name = os.path.splitext(filename)[0]
            template_path = os.path.join(full_template_dir, filename)
            template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            _, template_binary = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY_INV)
            templates[char_name] = template_binary
    final_plate_text = ""
    correlation_scores = []
    for images in character_images:
        char_image = images['thresh']
        best_match_char, max_corr_value = None, -1.0
        h_char, w_char = char_image.shape
        for char_name, template_img in templates.items():
            resized_template = cv2.resize(template_img, (w_char, h_char))
            result = cv2.matchTemplate(char_image, resized_template, cv2.TM_CCOEFF_NORMED)
            corr_value = result[0][0]
            if corr_value > max_corr_value:
                max_corr_value = corr_value
                best_match_char = char_name
        if best_match_char:
            final_plate_text += best_match_char
            correlation_scores.append(max_corr_value)
    avg_correlation = np.mean(correlation_scores) if correlation_scores else 0.0
    return final_plate_text, avg_correlation

def visualize_specific_comparison(original_image, failing_rate, ground_truth_plate, template_dir='numbers'):
    """Visualizes the template matching for a specific character at the point of failure."""
    print("\nStep 4: Visualizing the pixel-level comparison at the failure point...")
    h, w = original_image.shape[:2]
    new_dims = (int(w * failing_rate), int(h * failing_rate))
    failing_image = cv2.resize(original_image, new_dims, interpolation=cv2.INTER_AREA)
    segmented_chars = segment_characters(failing_image, scale_factor=failing_rate)
    if not segmented_chars:
        print("-> Could not segment characters for visualization at the failing rate.")
        return
    char_from_plate = segmented_chars[6]['thresh']
    correct_char_name = ground_truth_plate[6]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, template_dir, f"{correct_char_name}.jpg")
    template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template_gray is None:
        print(f"-> Could not load template for character '{correct_char_name}'.")
        return
    _, high_res_template_binary = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY_INV)
    h_char, w_char = char_from_plate.shape
    resized_template = cv2.resize(high_res_template_binary, (w_char, h_char))
    result = cv2.matchTemplate(char_from_plate, resized_template, cv2.TM_CCOEFF_NORMED)
    correlation = result[0][0]
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(char_from_plate, cmap='gray')
    plt.title(f"Character from Failed Plate\n(Rate: {failing_rate:.2f})")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(resized_template, cmap='gray')
    plt.title(f"Resized High-Res Template ('{correct_char_name}')")
    plt.axis('off')
    plt.suptitle(f"Pixel-Level Comparison for Character '{correct_char_name}'\nCorrelation: {correlation:.4f}")
    plt.show()

def analyze_downsampling_effect():
    """Main function to analyze the effect of spatial downsampling on recognition accuracy."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_file_path = os.path.join(script_dir, 'ideal/p4.jpg')
    original_image = cv2.imread(image_file_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_file_path}. Exiting.")
        return

    print("Step 1: Establishing ground truth from original image (rate = 1.0)...")
    full_res_chars = segment_characters(original_image, scale_factor=1.0)
    ground_truth_plate, _ = recognize_plate(full_res_chars)
    if not ground_truth_plate:
        print("Fatal Error: Could not recognize the plate from the original image.")
        return
    print(f"-> Ground Truth plate established as: '{ground_truth_plate}'\n")

    print("Step 2: Analyzing effects of downsampling across different rates...")
    downsampling_rates = np.arange(1.0, 0.04, -0.05)
    results = []
    for rate in downsampling_rates:
        h, w = original_image.shape[:2]
        new_dims = (int(w * rate), int(h * rate))
        downsampled_image = cv2.resize(original_image, new_dims, interpolation=cv2.INTER_AREA)
        segmented_chars = segment_characters(downsampled_image, scale_factor=rate)
        recognized_text, avg_corr = recognize_plate(segmented_chars)
        is_correct = 1 if recognized_text == ground_truth_plate else 0
        results.append({
            "rate": rate, "recognized_text": recognized_text, "is_correct": is_correct, "avg_correlation": avg_corr
        })

    print("\nStep 3: Finding the point of failure for visualization...")
    first_failing_rate = None
    sorted_results_desc = sorted(results, key=lambda x: x['rate'], reverse=True)
    for i, res in enumerate(sorted_results_desc):
        if not res['is_correct'] and i > 0 and sorted_results_desc[i-1]['is_correct']:
            first_failing_rate = res['rate']
            print(f"-> Failure point identified at rate: {first_failing_rate:.2f}")
            break
            
    if first_failing_rate:
        visualize_specific_comparison(original_image, first_failing_rate, ground_truth_plate)
    else:
        print("\n-> No failure point found. The system was successful at all tested rates.")

    print("\n--- Analysis Results Table ---")
    print(f"{'Rate':<10} | {'Recognized Plate':<20} | {'Avg. Correlation':<20} | {'Correct?':<10}")
    print("-" * 75)
    for res in sorted_results_desc:
        correct_str = "Yes" if res['is_correct'] else "No"
        recognized_text = res['recognized_text'] if res['recognized_text'] else "N/A"
        print(f"{res['rate']:.2f}{'':<6} | {recognized_text:<20} | {res['avg_correlation']:.4f}{'':<14} | {correct_str:<10}")

    rates = [r['rate'] for r in results]
    accuracies = [r['is_correct'] for r in results]
    correlations = [r['avg_correlation'] for r in results]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.title('Effect of Downsampling on License Plate Recognition Accuracy')
    ax1.set_xlabel('Downsampling Rate (Image Scale Factor)')
    ax1.set_ylabel('Recognition Accuracy (1 = Correct, 0 = Incorrect)', color='tab:blue')
    ax1.plot(rates, accuracies, 'o-', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(-0.1, 1.1)
    ax1.invert_xaxis()
    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Character Correlation Score', color='tab:red')
    ax2.plot(rates, correlations, 's--', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.tight_layout()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    analyze_downsampling_effect()