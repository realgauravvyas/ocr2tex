# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 01:40:52 2026

@author: Gaurav
"""

#!/usr/bin/env python3
import cv2, numpy as np, math, sys
from pathlib import Path
from tqdm import tqdm

def get_skew_angle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 1. Connect text into horizontal stripes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # 2. Detect lines
    edges = cv2.Canny(eroded, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=int(img.shape[1]*0.15), maxLineGap=10)
    
    if lines is None: 
        return 0.0
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2-y1, x2-x1))
        
        # Normalize to -90..90
        if angle < -90: angle += 180
        if angle > 90: angle -= 180
        
        # CRITICAL: Only keep nearly horizontal text lines
        if -8 <= angle <= 8:
            angles.append(angle)
            
    if not angles: 
        return 0.0
    
    # Median rejects diagram outliers
    skew = float(np.median(angles))
    
    # Safety: If model detects >5°, it's likely a diagram, not page skew
    if abs(skew) > 5.0:
        return 0.0
        
    return skew

def rotate_to_straight(img, angle, bg_color=(255,255,255)):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    
    # NEGATE the angle: if page is tilted +2°, we rotate -2° to fix it
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    # Expand canvas to prevent clipping
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    return cv2.warpAffine(img, M, (new_w, new_h), borderValue=bg_color)

def main():
    print("📐 Fixed Deskewer (Sign-Corrected + Safe Rotation)")
    
    src_input = input("📁 Source folder: ").strip().strip('"').strip("'")
    dst_input = input("📁 Output folder: ").strip().strip('"').strip("'")
    
    src_path = Path(src_input)
    dst_path = Path(dst_input)
    
    if not src_path.exists(): print("❌ Source not found."); sys.exit(1)
    dst_path.mkdir(parents=True, exist_ok=True)
    
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    files = []
    for ext in extensions:
        files.extend(list(src_path.rglob(ext)))
    files = sorted(list(set(files)))
    
    if not files:
        print("❌ No images found.")
        sys.exit(1)
        
    print(f"\n🔍 Found {len(files)} images. Starting...\n")
    
    rotated_count = 0
    for i, f in tqdm(enumerate(files, 1), total=len(files), desc="Deskewing"):
        try:
            img = cv2.imread(str(f))
            if img is None: continue
            
            bg_color = (int(np.mean(img[:,:,2])), int(np.mean(img[:,:,1])), int(np.mean(img[:,:,0])))
            angle = get_skew_angle(img)
            
            if abs(angle) >= 0.8:
                img = rotate_to_straight(img, angle, bg_color)
                rotated_count += 1
                status = f"✅ Rotated by {-angle:.2f}°"
            else:
                status = "✅ Already straight"
                
            cv2.imwrite(str(dst_path / f.name), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            print(f"[{i}/{len(files)}] {status}: {f.name}")
            
        except Exception as e:
            print(f"[{i}/{len(files)}] ❌ Error {f.name}: {e}")
            
    print(f"\n✨ Done! Rotated {rotated_count}/{len(files)} pages.")
    print(f"💾 Saved to: {dst_path.absolute()}")

if __name__ == '__main__':
    main()