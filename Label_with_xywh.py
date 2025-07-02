import os
import random
import shutil
import cv2

# === CONFIG ===
image_folder = r"C:\Users\WSrisook\Downloads\1906\6Entegris"   # โฟลเดอร์รูปภาพ
labels_folder = r"C:\Users\WSrisook\Downloads\1906\labels"       # โฟลเดอร์ปลายทางไฟล์ .txt
file_label_folder = r"C:\Users\WSrisook\Downloads\1906\file_label"  # โฟลเดอร์รูปที่ตีกรอบแล้ว
classes = ["Canister", "Foam", "Ring", "Tyvek", "Wafer"]

# === SET bounding box values ===
x = 0.49954802858976477
y = 0.5574033282796372
w = 0.43050458369337075
h = 0.6787368486225077
   
# === สร้าง folder ถ้ายังไม่มี ===
os.makedirs(labels_folder, exist_ok=True)
os.makedirs(file_label_folder, exist_ok=True)
i = 1
j = 1
# === PROCESS each image ===
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        class_index = None
        for idx, class_name in enumerate(classes):
            if class_name.lower() in filename.lower():
                class_index = idx
                break

        if class_index is not None:
            # Create YOLO format line
            line = f"{class_index} {x} {y} {w} {h}\n"

            # Write to .txt file with the same name as image, in output folder
            basename = os.path.splitext(filename)[0]
            txt_path = os.path.join(labels_folder, f"{basename}.txt")

            with open(txt_path, "w") as f:
                f.write(line)

            print(f"{i}")
            i += 1
        else:
            print(f"No matching class found for {filename}, skipping.")

image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# === สุ่มเลือก 10% ของไฟล์ ===
num_select = max(1, int(len(image_files) * 0.1))
selected_files = random.sample(image_files, num_select)

for filename in selected_files:
    img_path = os.path.join(image_folder, filename)
    label_name = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(labels_folder, label_name)

    # === อ่านรูป ===
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        continue

    h_img, w_img, _ = img.shape

    # === อ่าน label file ===
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = parts
                x_center = float(x_center)
                y_center = float(y_center)
                width = float(width)
                height = float(height)

                # แปลงจาก normalized เป็น pixel
                x_c = x_center * w_img
                y_c = y_center * h_img
                w_box = width * w_img
                h_box = height * h_img

                x1 = int(x_c - w_box / 2)
                y1 = int(y_c - h_box / 2)
                x2 = int(x_c + w_box / 2)
                y2 = int(y_c + h_box / 2)

                # วาด rectangle (สีเขียว)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                print(f"Invalid label format in {label_path}")

        # === Save output image ===
        output_path = os.path.join(file_label_folder, filename)
        cv2.imwrite(output_path, img)
        print(f"Sample {j}")
        j += 1

    else:
        print(f"No label file for {filename}, skipping.")