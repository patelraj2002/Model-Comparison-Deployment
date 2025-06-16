#!/usr/bin/env python3
"""
Object Detection Model Comparison Tool
Compares YOLOv5n vs YOLOv5s on the same set of images
"""

import os
import time
import torch
import pandas as pd
from pathlib import Path
from PIL import Image # <<< CHANGE HERE: Import Image for saving
import argparse

# <<< CHANGE HERE: No changes to setup_models, it's already good
def setup_models():
    """Load YOLOv5n and YOLOv5s models"""
    print("Loading models...")
    model_n = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    model_s = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model_n.eval()
    model_s.eval()
    return model_n, model_s

# <<< CHANGE HERE: We will return the full 'results' object to be efficient
def run_inference(model, image_path):
    """Run inference on a single image and return detailed results."""
    start_time = time.time()
    
    # Run inference
    results = model(image_path, size=640)
    
    inference_time = time.time() - start_time
    
    # Parse results
    detections = results.pandas().xyxy[0]
    detection_count = len(detections)
    unique_classes = detections['name'].nunique() if not detections.empty else 0
    
    return {
        'inference_time': inference_time,
        'detection_count': detection_count,
        'unique_classes': unique_classes,
        'results_object': results  # <<< CHANGE HERE: Return the whole object
    }

def compare_models(image_folder, output_folder):
    """Compare YOLOv5n vs YOLOv5s on images"""
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    model_n, model_s = setup_models()
    
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [f for f in os.listdir(image_folder) 
                   if Path(f).suffix.lower() in image_extensions][:10]
    
    if not image_files:
        print(f"Error: No images found in {image_folder}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    results_list = []
    
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing image {i}/{len(image_files)}: {image_file}")
        
        # Run inference with both models (only once per model)
        result_n = run_inference(model_n, image_path)
        result_s = run_inference(model_s, image_path)
        
        results_list.append({
            'Image': image_file,
            'YOLOv5n_Time': result_n['inference_time'],
            'YOLOv5n_Detections': result_n['detection_count'],
            'YOLOv5n_Classes': result_n['unique_classes'],
            'YOLOv5s_Time': result_s['inference_time'],
            'YOLOv5s_Detections': result_s['detection_count'],
            'YOLOv5s_Classes': result_s['unique_classes']
        })
        
        # <<< CHANGE HERE: New, reliable way to save annotated images
        # Get the rendered image (image with boxes drawn on it)
        rendered_image_n = result_n['results_object'].render()[0]
        rendered_image_s = result_s['results_object'].render()[0]
        
        # Convert from BGR (OpenCV format) to RGB (Pillow format)
        img_n_pil = Image.fromarray(rendered_image_n[:, :, ::-1])
        img_s_pil = Image.fromarray(rendered_image_s[:, :, ::-1])
        
        # Save the images with the correct name in the output folder
        img_n_pil.save(os.path.join(output_folder, f"yolov5n_{image_file}"))
        img_s_pil.save(os.path.join(output_folder, f"yolov5s_{image_file}"))

    # --- The rest of your script for creating and printing the tables is perfect ---
    # --- No changes needed below this line, except for a clearer print message ---
    
    df = pd.DataFrame(results_list)
    avg_results = {
        'Metric': ['Average Inference Time (s)', 'Average Detection Count', 'Average Class Diversity'],
        'YOLOv5n': [df['YOLOv5n_Time'].mean(), df['YOLOv5n_Detections'].mean(), df['YOLOv5n_Classes'].mean()],
        'YOLOv5s': [df['YOLOv5s_Time'].mean(), df['YOLOv5s_Detections'].mean(), df['YOLOv5s_Classes'].mean()]
    }
    avg_df = pd.DataFrame(avg_results)
    avg_df['YOLOv5s/YOLOv5n Ratio'] = avg_df['YOLOv5s'] / avg_df['YOLOv5n']
    
    # Save CSV reports to the output folder
    df.to_csv(os.path.join(output_folder, "detailed_comparison.csv"), index=False)
    avg_df.to_csv(os.path.join(output_folder, "average_comparison.csv"), index=False)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60 + "\nDetailed Results:")
    print(df.to_string(index=False))
    print("\nAverage Comparison:")
    print(avg_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    time_ratio = avg_df.loc[0, 'YOLOv5s/YOLOv5n Ratio']
    detection_ratio = avg_df.loc[1, 'YOLOv5s/YOLOv5n Ratio'] 
    class_ratio = avg_df.loc[2, 'YOLOv5s/YOLOv5n Ratio']
    print(f"• YOLOv5s is {time_ratio:.2f}x slower than YOLOv5n")
    print(f"• YOLOv5s detects {detection_ratio:.2f}x more objects than YOLOv5n")
    print(f"• YOLOv5s has {class_ratio:.2f}x more class diversity than YOLOv5n")
    if detection_ratio > time_ratio:
        print("• YOLOv5s offers better detection performance relative to speed cost")
    else:
        print("• YOLOv5n offers better speed relative to detection performance")
    
    # <<< CHANGE HERE: Clearer final message
    print(f"\nAll annotated images and CSV reports have been saved to: '{output_folder}'")

# --- No changes to the main() function needed, it's perfect ---
def main():
    parser = argparse.ArgumentParser(description='Compare YOLOv5n vs YOLOv5s object detection models')
    parser.add_argument('--input', '-i', required=True, help='Input folder containing images')
    parser.add_argument('--output', '-o', default='./comparison_results', help='Output folder for results')
    args = parser.parse_args()
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist!")
        return
    compare_models(args.input, args.output)

if __name__ == "__main__":
    main()