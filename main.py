import os
import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
import napari
import shutil
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QScrollArea, QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap

# ====== CONFIG ======
import os

while True:
    INPUT_NRRD_PATH = input("Enter path to input NRRD file (e.g. input/R-002-1.nrrd): ")
    if os.path.isfile(INPUT_NRRD_PATH):
        break
    else:
        print("❌ File not found. Please try again.")

OUTPUT_NRRD_PATH = "output/colorized_3d_volume.nrrd"
TMP_SLICE_DIR = "temp_slices/"
TMP_COLORIZED_DIR = "temp_colorized_slices/"
os.makedirs("output", exist_ok=True)
os.makedirs(TMP_SLICE_DIR, exist_ok=True)
os.makedirs(TMP_COLORIZED_DIR, exist_ok=True)

# ====== LOAD MODEL ======
prototxt_path = "models/colorization_deploy_v2.prototxt"
model_path = "models/colorization_release_v2.caffemodel"
cluster_path = "models/pts_in_hull.npy"

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
pts = np.load(cluster_path)
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# ====== STEP 1: Loading image Slices from NRRD ======
print("🔵 Loading Image...")
volume = sitk.ReadImage(INPUT_NRRD_PATH)
volume_np = sitk.GetArrayFromImage(volume)  # shape: (Z, H, W)
for idx, slice_ in enumerate(volume_np):
    path = os.path.join(TMP_SLICE_DIR, f"slice_{idx:03d}.png")
    # plt.imsave(path, slice_, cmap='gray')
    plt.imsave(path, slice_, cmap='gray')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(path, rgb)

# ====== STEP 2: Colorizing  ======
print("🟠 Colorizing...")
for file in sorted(os.listdir(TMP_SLICE_DIR)):
    img = cv2.imread(os.path.join(TMP_SLICE_DIR, file))
    if img is None: continue
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    resized_L = cv2.resize(L, (224, 224)) - 50
    net.setInput(cv2.dnn.blobFromImage(resized_L))
    ab_dec = net.forward()[0].transpose(1, 2, 0)
    ab_up = cv2.resize(ab_dec, (img.shape[1], img.shape[0]))
    Lab_out = np.concatenate((L[:, :, np.newaxis], ab_up), axis=2)
    bgr = cv2.cvtColor(Lab_out.astype("float32"), cv2.COLOR_LAB2BGR)
    bgr = np.clip(bgr, 0, 1)
    out_path = os.path.join(TMP_COLORIZED_DIR, f"colorized_{file}")
    cv2.imwrite(out_path, (bgr * 255).astype("uint8"))

# ====== STEP 3: Stack colorized slices into volume ======
print("🟢 Rebuilding volume...")
slices = []
for file in sorted(os.listdir(TMP_COLORIZED_DIR)):
    img = cv2.imread(os.path.join(TMP_COLORIZED_DIR, file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    slices.append(img)
vol_color_np = np.stack(slices, axis=0)
sitk.WriteImage(sitk.GetImageFromArray(vol_color_np, isVector=True), OUTPUT_NRRD_PATH)
print(f"✅ Saved colorized volume: {OUTPUT_NRRD_PATH}")

# ====== STEP 4: Show Input GUI (Grayscale Viewer) ======
def show_input_gui():
    app = QApplication(sys.argv)
    win = QWidget()
    layout = QVBoxLayout()

    first_file = sorted(os.listdir(TMP_SLICE_DIR))[0]
    img_path = os.path.join(TMP_SLICE_DIR, first_file)
    print("📸 Showing image from:", img_path)

    pixmap = QPixmap(img_path)

    if pixmap.isNull():
        print("❌ Failed to load image!")
        error_label = QLabel("Image failed to load.")
        layout.addWidget(error_label)
    else:
        img_label = QLabel()
        img_label.setPixmap(pixmap.scaledToWidth(400))
        layout.addWidget(img_label)

   
    win.setLayout(layout)
    win.resize(450, 500)
    win.show()
    app.exec_()
show_input_gui()

# ====== STEP 5: Show 3d Output Viewer ======
print("👁️ Launching 3d Output Viewer...")
image = sitk.ReadImage(OUTPUT_NRRD_PATH)
array = sitk.GetArrayFromImage(image)
viewer = napari.view_image(array, name="Colorized Volume")
viewer.window._qt_viewer.dockLayerControls.setVisible(False)
viewer.window._qt_viewer.dockConsole.setVisible(False)
viewer.layers.selection.active = None
napari.run()

# ====== STEP 6: Cleanup ======
shutil.rmtree(TMP_SLICE_DIR)
shutil.rmtree(TMP_COLORIZED_DIR)
print("🧹 Cleaned up temp folders.")