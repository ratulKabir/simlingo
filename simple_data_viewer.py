import os
import gzip
import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_single_sample(data_dir):
    """Load a single sample from the dataset directory."""
    # Find a Town directory that contains data
    print("Looking for data in:", data_dir)
    town_dirs = list(Path(data_dir).glob('data/simlingo/*/*/*/Town*'))
    if not town_dirs:
        print("No town directories found!")
        return
    
    town_dir = town_dirs[0]  # Take the first town directory
    print(f"\nUsing town directory: {town_dir}")
    
    # Get RGB image
    rgb_dir = town_dir / 'rgb'
    if not rgb_dir.exists():
        print("No rgb directory found!")
        return
    
    # Get first image
    rgb_files = sorted(rgb_dir.glob('*.jpg'))
    if not rgb_files:
        print("No RGB images found!")
        return
    
    # Load image
    img_path = rgb_files[313]
    frame_num = img_path.stem  # Get frame number without extension
    print(f"\nLoading frame: {frame_num}")
    
    # Load RGB image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load corresponding measurement file
    measurement_path = town_dir / 'measurements' / f'{frame_num}.json.gz'
    if not measurement_path.exists():
        print("No measurement file found!")
        return
    
    with gzip.open(measurement_path, 'rt') as f:
        measurements = json.load(f)
    
    # Load corresponding boxes file
    boxes_path = town_dir / 'boxes' / f'{frame_num}.json.gz'
    boxes = None
    if boxes_path.exists():
        with gzip.open(boxes_path, 'rt') as f:
            boxes = json.load(f)
    
    # Load commentary file
    commentary_path = str(measurement_path).replace('data/', 'commentary/').replace('measurements', 'commentary')
    commentary = None
    if Path(commentary_path).exists():
        with gzip.open(commentary_path, 'rt') as f:
            commentary = json.load(f)
    
    # Load VQA file
    vqa_path = str(measurement_path).replace('data/', 'drivelmTD/').replace('measurements', 'vqa')
    vqa = None
    if Path(vqa_path).exists():
        with gzip.open(vqa_path, 'rt') as f:
            vqa = json.load(f)
    
    # Display everything
    plt.figure(figsize=(15, 10))
    
    # Add title showing the frame path
    frame_path = str(town_dir)
    frame_title = (f"Town: {frame_path.split('Town')[1].split('/')[0]}\n"
                  f"Route: {frame_path.split('route')[1].split('/')[0]}\n"
                  f"Frame: {frame_num}")
    plt.suptitle(frame_title, fontsize=10, y=0.98)
    
    # Show image
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('RGB Image')
    
    # Print commentary
    plt.subplot(2, 2, 2)
    plt.axis('off')
    text = "Commentary:\n"
    if commentary:
        text += commentary.get('commentary', 'N/A')
    else:
        text += "No commentary available"
    plt.text(0, 0.5, text, fontsize=10, verticalalignment='center', wrap=True)
    
    # Print measurements
    plt.subplot(2, 2, 3)
    plt.axis('off')
    text = "Measurements:\n"
    text += f"Speed: {measurements.get('speed', 'N/A')} m/s\n"
    text += f"Command: {measurements.get('command', 'N/A')}\n"
    text += f"Target Speed: {measurements.get('target_speed', 'N/A')} m/s\n"
    text += f"Brake: {measurements.get('brake', 'N/A')}\n"
    text += f"Throttle: {measurements.get('throttle', 'N/A')}\n"
    text += f"Steer: {measurements.get('steer', 'N/A')}\n"
    
    plt.text(0, 0.5, text, fontsize=10, verticalalignment='center')
    
    # Print VQA
    plt.subplot(2, 2, 4)
    plt.axis('off')
    text = "Visual Q&A:\n"
    if vqa and 'QA' in vqa:
        for qa_type, qa_list in vqa['QA'].items():
            text += f"\n{qa_type}:\n"
            # Only show first 3 QA pairs from each section
            for qa in qa_list[:3]:
                text += f"Q: {qa['Q']}\n"
                text += f"A: {qa['A']}\n"
            if len(qa_list) > 3:
                text += f"... ({len(qa_list)-3} more)\n"
    else:
        text += "No VQA data available"
    plt.text(0, 0.5, text, fontsize=8, verticalalignment='center')
    
    plt.tight_layout()
    plt.show()
    
    # Save visualization
    plt.savefig('sample_visualization.png')
    print("\nSaved visualization to: sample_visualization.png")
    
    return {
        'image': img,
        'measurements': measurements,
        'boxes': boxes,
        'commentary': commentary,
        'vqa': vqa
    }

if __name__ == "__main__":
    # Path to your dataset
    data_dir = "database/simlingo_v2_2025_05_24"
    
    print("This script will load and display a single sample from your dataset.")
    print("It will show you the RGB image and all associated data (measurements, boxes, commentary, and VQA).")
    
    sample = load_single_sample(data_dir)
    
    if sample:
        print("\nAvailable measurements:", list(sample['measurements'].keys()))
        if sample['boxes']:
            print("\nTypes of objects detected:", set(box['class'] for box in sample['boxes'] if 'class' in box))
        if sample['commentary']:
            print("\nCommentary available:", bool(sample['commentary']))
        if sample['vqa']:
            print("\nVQA data available:", bool(sample['vqa'])) 