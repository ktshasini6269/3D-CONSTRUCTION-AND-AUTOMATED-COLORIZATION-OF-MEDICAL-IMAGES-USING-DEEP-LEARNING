# stack_colorized_slices.py
import numpy as np
import os
import cv2
import SimpleITK as sitk

colorized_dir = "output/"
stacked_volume_output_path = "colorized_3d_volume.nrrd"

# Read all colorized images
colorized_slices = []

for filename in sorted(os.listdir(colorized_dir)):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(colorized_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        colorized_slices.append(img)

# Stack into 3D volume
volume_np = np.stack(colorized_slices, axis=0)  # Shape: (depth, height, width, 3)

print(f"Stacked volume shape: {volume_np.shape}")

# Save as a 3D NRRD
volume_sitk = sitk.GetImageFromArray(volume_np, isVector=True)  # Important: isVector=True for 3-channel RGB
sitk.WriteImage(volume_sitk, stacked_volume_output_path)

print(f"✅ 3D colorized volume saved to: {stacked_volume_output_path}")