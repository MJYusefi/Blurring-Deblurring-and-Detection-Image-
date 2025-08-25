import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# SECTION 1: HELPER FUNCTIONS
# These are unchanged and are required for the script to run.
# ==============================================================================
def segment_characters(image, scale_factor=1.0):
    if image is None: return []
    original_padding = 5
    scaled_padding = int(max(1, original_padding * scale_factor))
    orig_y_start, orig_y_end, orig_x_start, orig_x_end = 31, 141, 115, 727
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
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if h != 0 and (h > 0.4 * image_height) and ((w/h) < 1.2):
            char_boxes.append((x, y, w, h))
    char_boxes = sorted(char_boxes, key=lambda box: box[0])
    character_images = []
    for box in char_boxes:
        x, y, w, h = box
        char_thresh_tight = thresh[y:y+h, x:x+w]
        char_thresh_padded = cv2.copyMakeBorder(char_thresh_tight, scaled_padding, scaled_padding, scaled_padding, scaled_padding, cv2.BORDER_CONSTANT, value=0)
        character_images.append({'thresh': char_thresh_padded})
    return character_images

def recognize_plate(character_images, template_dir='numbers'):
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
    final_plate_text, correlation_scores = "", []
    for images in character_images:
        char_image = images['thresh']
        best_match_char, max_corr_value = None, -1.0
        h_char, w_char = char_image.shape
        for char_name, template_img in templates.items():
            resized_template = cv2.resize(template_img, (w_char, h_char))
            result = cv2.matchTemplate(char_image, resized_template, cv2.TM_CCOEFF_NORMED)
            if result[0][0] > max_corr_value:
                max_corr_value, best_match_char = result[0][0], char_name
        if best_match_char:
            final_plate_text += best_match_char
            correlation_scores.append(max_corr_value)
    return final_plate_text, (np.mean(correlation_scores) if correlation_scores else 0.0)

def visualize_pixel_comparison(original_image, failing_rate, ground_truth_plate, template_dir='numbers'):
    """Visualizes the pixel-level comparison at the point of failure."""
    print("\nStep 4: Visualizing the pixel-level comparison at the failure point...")
    subsampled_image = original_image[::failing_rate, ::failing_rate]
    scale_factor = 1.0 / failing_rate
    segmented_chars = segment_characters(subsampled_image, scale_factor=scale_factor)
    if not segmented_chars:
        print("-> Could not segment characters for visualization at the failing rate.")
        return
    char_from_plate = segmented_chars[0]['thresh']
    correct_char_name = ground_truth_plate[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, template_dir, f"{correct_char_name}.jpg")
    template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template_gray is None: return
    _, high_res_template_binary = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY_INV)
    h_char, w_char = char_from_plate.shape
    resized_template = cv2.resize(high_res_template_binary, (w_char, h_char))
    correlation = cv2.matchTemplate(char_from_plate, resized_template, cv2.TM_CCOEFF_NORMED)[0][0]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(char_from_plate, cmap='gray')
    plt.title(f"Character from Failed Plate\n(Subsampled 1 in {failing_rate})")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(resized_template, cmap='gray')
    plt.title(f"Resized High-Res Template ('{correct_char_name}')")
    plt.axis('off')
    plt.suptitle(f"Pixel-Level Comparison for Character '{correct_char_name}'\nCorrelation: {correlation:.4f}")
    plt.show()

# ==============================================================================
# SECTION 2: MAIN ANALYSIS SCRIPT
# ==============================================================================

def analyze_subsampling_effect():
    """Analyzes character recognition by subsampling with full-featured reporting."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_file_path = os.path.join(script_dir, 'ideal/p4.jpg')
    original_image = cv2.imread(image_file_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_file_path}.")
        return

    # Step 1: Establish Ground Truth
    print("Step 1: Establishing ground truth from original image...")
    full_res_chars = segment_characters(original_image, scale_factor=1.0)
    ground_truth_plate, _ = recognize_plate(full_res_chars)
    if not ground_truth_plate:
        print("Fatal Error: Could not recognize the plate from the original image.")
        return
    print(f"-> Ground Truth plate established as: '{ground_truth_plate}'\n")

    # Step 2: Perform Subsampling Analysis
    print("Step 2: Analyzing effects of subsampling...")
    subsampling_rates = range(1, 21)
    results = []
    for rate in subsampling_rates:
        subsampled_image = original_image[::rate, ::rate]
        scale_factor = 1.0 / rate
        segmented_chars = segment_characters(subsampled_image, scale_factor=scale_factor)
        recognized_text, avg_corr = recognize_plate(segmented_chars)
        is_correct = 1 if recognized_text == ground_truth_plate else 0
        results.append({"rate": rate, "recognized_text": recognized_text, "is_correct": is_correct, "avg_correlation": avg_corr})

    # Step 3: Visualize the Point of Failure
    print("\nStep 3: Finding the point of failure for visualization...")
    last_successful_rate, first_failing_rate = None, None
    for i, res in enumerate(results):
        if not res['is_correct'] and i > 0 and results[i-1]['is_correct']:
            last_successful_rate, first_failing_rate = results[i-1]['rate'], res['rate']
            print(f"-> Failure point identified. Last success at 1-in-{last_successful_rate}, first failure at 1-in-{first_failing_rate}.")
            break
    
    if last_successful_rate:
        last_success_img = original_image[::last_successful_rate, ::last_successful_rate]
        first_fail_img = original_image[::first_failing_rate, ::first_failing_rate]
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(last_success_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Last Successful Image (1 in {last_successful_rate} pixels)")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(first_fail_img, cv2.COLOR_BGR2RGB))
        plt.title(f"First Failing Image (1 in {first_failing_rate} pixels)")
        plt.axis('off')
        plt.suptitle("Comparison at the Point of Failure")
        plt.show()

    # Step 4: Visualize Pixel-Level Mismatch
    if first_failing_rate:
        visualize_pixel_comparison(original_image, first_failing_rate, ground_truth_plate)
    
    # Step 5: Report Full Results in a Table
    print("\n--- Subsampling Analysis Results Table ---")
    print(f"{'Rate (1 in N)':<15} | {'Recognized Plate':<20} | {'Avg. Correlation':<20} | {'Correct?':<10}")
    print("-" * 75)
    for res in results:
        correct_str = "Yes" if res['is_correct'] else "No"
        recognized_text = res['recognized_text'] if res['recognized_text'] else "N/A"
        print(f"{res['rate']:<15} | {recognized_text:<20} | {res['avg_correlation']:.4f}{'':<14} | {correct_str:<10}")

    successful_rates = [r['rate'] for r in results if r['is_correct']]
    max_successful_rate = max(successful_rates) if successful_rates else None
    print("-" * 75)
    if max_successful_rate:
        print(f"\n✅ Maximum successful subsampling rate: 1 in {max_successful_rate} pixels.")
    else:
        print("\n❌ Recognition failed for all tested subsampling rates.")

    # Step 6: Plot Overall Performance Graph (Dual Axis)
    rates = [r['rate'] for r in results]
    accuracies = [r['is_correct'] for r in results]
    correlations = [r['avg_correlation'] for r in results]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.title('Recognition Performance vs. Subsampling Rate')
    
    ax1.set_xlabel('Subsampling Rate (N, where 1 in N pixels is taken)')
    ax1.set_ylabel('Recognition Accuracy (1 = Correct, 0 = Incorrect)', color='tab:blue')
    ax1.plot(rates, accuracies, 'o-', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(-0.1, 1.1)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Character Correlation Score', color='tab:red')
    ax2.plot(rates, correlations, 's--', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.xticks(rates)
    plt.grid(True, linestyle='--')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_subsampling_effect()