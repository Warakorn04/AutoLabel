import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

# === CONFIG ===
classes = ["Canister", "Foam", "Ring", "Tyvek", "Wafer"]

bbox_presets = {
    "6 inch": (0.49954802858976477, 0.5574033282796372, 0.43050458369337075, 0.6787368486225077),
    "8 inch": (0.4936826531694092, 0.5458452479416916, 0.5703962017917926, 0.8867322545523849)
}

class AutoLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Labeling")
        self.root.geometry("500x600")
        self.root.configure(bg="white")

        self.image_folder = ""
        self.output_folder = ""
        self.bbox_size = tk.StringVar(value="6 inch")

        self.setup_ui()

    def setup_ui(self):
        font_title = ("Arial", 16, "bold")
        font_label = ("Arial", 12, "bold")
        font_btn = ("Arial", 10, "bold")

        tk.Label(self.root, text="Auto Labeling", font=font_title, bg="white").pack(pady=10)

        # Input folder
        tk.Label(self.root, text="Image input folder", font=font_label, bg="white").place(x=30, y=60)
        tk.Button(self.root, text="Browse", font=font_btn, command=self.select_input).place(x=400, y=55)

        # Output folder
        tk.Label(self.root, text="Image & Label Output folder", font=font_label, bg="white").place(x=30, y=100)
        tk.Button(self.root, text="Browse", font=font_btn, command=self.select_output).place(x=400, y=95)

        # Size selection
        tk.Label(self.root, text="Size:", font=font_label, bg="white").place(x=30, y=150)
        tk.Radiobutton(self.root, text="6 inch", font=font_btn, variable=self.bbox_size, value="6 inch", bg="white").place(x=100, y=150)
        tk.Radiobutton(self.root, text="8 inch", font=font_btn, variable=self.bbox_size, value="8 inch", bg="white").place(x=180, y=150)

        # Preview area
        tk.Label(self.root, text="Preview", font=font_label, bg="white").place(x=30, y=200)
        self.canvas = tk.Canvas(self.root, width=400, height=250, bg="white", highlightbackground="black")
        self.canvas.place(x=50, y=230)

        # Button
        tk.Button(self.root, text="Run Auto Labeling", font=font_btn, command=self.run_labeling).place(x=180, y=510)

    def select_input(self):
        path = filedialog.askdirectory()
        if path:
            self.image_folder = path
            self.show_preview()

    def select_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_folder = path

    def run_labeling(self):
        if not self.image_folder or not self.output_folder:
            messagebox.showwarning("Missing folder", "Please select both input and output folders.")
            return

        os.makedirs(self.output_folder, exist_ok=True)
        label_folder = os.path.join(self.output_folder, "labels")
        os.makedirs(label_folder, exist_ok=True)

        x, y, w, h = bbox_presets[self.bbox_size.get()]

        for filename in os.listdir(self.image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                class_index = None
                for idx, cname in enumerate(classes):
                    if cname.lower() in filename.lower():
                        class_index = idx
                        break
                if class_index is not None:
                    line = f"{class_index} {x} {y} {w} {h}\n"
                    txt_path = os.path.join(label_folder, os.path.splitext(filename)[0] + ".txt")
                    with open(txt_path, "w") as f:
                        f.write(line)

        # === วาดกรอบและเซฟทุกภาพ ===
        file_label_folder = os.path.join(self.output_folder, "image_label")
        os.makedirs(file_label_folder, exist_ok=True)
        image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        for filename in image_files:
            img_path = os.path.join(self.image_folder, filename)
            label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + ".txt")
            img = cv2.imread(img_path)
            if img is None or not os.path.exists(label_path):
                continue

            h_img, w_img = img.shape[:2]
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, x_center, y_center, width, height = map(float, parts)
                    x_c = x_center * w_img
                    y_c = y_center * h_img
                    w_box = width * w_img
                    h_box = height * h_img
                    x1, y1 = int(x_c - w_box / 2), int(y_c - h_box / 2)
                    x2, y2 = int(x_c + w_box / 2), int(y_c + h_box / 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imwrite(os.path.join(file_label_folder, filename), img)

        self.show_preview()
        messagebox.showinfo("Success", "Auto labeling completed successfully!")

    def show_preview(self):
        if not self.image_folder:
            return

        files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        if not files:
            return

        selected = random.choice(files)
        img_path = os.path.join(self.image_folder, selected)
        img = cv2.imread(img_path)

        if img is None:
            return

        h_img, w_img = img.shape[:2]
        x, y, w, h = bbox_presets[self.bbox_size.get()]
        x_c = x * w_img
        y_c = y * h_img
        w_box = w * w_img
        h_box = h * h_img

        x1, y1 = int(x_c - w_box / 2), int(y_c - h_box / 2)
        x2, y2 = int(x_c + w_box / 2), int(y_c + h_box / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        pil_img.thumbnail((400, 250))
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(200, 125, image=self.tk_img)

# === RUN ===
if __name__ == "__main__":
    root = tk.Tk()
    app = AutoLabelingApp(root)
    root.mainloop()
