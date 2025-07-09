import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import threading
import time
from tqdm import tqdm

# === CONFIG ===
classes = ["Canister", "Foam", "Ring", "Tyvek", "Wafer"]

bbox_presets = {
    "6 inch": (0.49954802858976477, 0.5574033282796372, 0.43050458369337075, 0.6787368486225077),
    "8 inch": (0.4936826531694092, 0.5458452479416916, 0.5703962017917926, 0.8867322545523849)
}

class ProgressDialog:
    def __init__(self, parent, title="Processing", max_value=100):
        self.parent = parent
        self.max_value = max_value
        self.is_cancelled = False
        self.current_step = 0
        self.total_files = 0
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x200")
        self.dialog.resizable(False, False)
        self.dialog.grab_set()  # Make dialog modal
        
        # Center the dialog
        self.dialog.transient(parent)
        self.dialog.geometry("+{}+{}".format(
            parent.winfo_rootx() + 50,
            parent.winfo_rooty() + 50
        ))
        
        # Variables
        self.progress_var = tk.DoubleVar()
        self.current_file_var = tk.StringVar()
        self.counter_var = tk.StringVar()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame with border
        main_frame = tk.Frame(self.dialog, relief='ridge', bd=2, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title label
        title_label = tk.Label(main_frame, text="Auto Labeling Progress", 
                              font=('Arial', 11, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=(10, 5))
        
        # Counter label (1/10 style)
        self.counter_label = tk.Label(main_frame, textvariable=self.counter_var,
                                     font=('Arial', 10, 'bold'), bg='#f0f0f0', fg='#333333')
        self.counter_label.pack(pady=(0, 5))
        
        # Current file label
        self.file_label = tk.Label(main_frame, textvariable=self.current_file_var,
                                  font=('Arial', 9), bg='#f0f0f0', fg='#666666')
        self.file_label.pack(pady=(0, 10))
        
        # Progress frame
        progress_frame = tk.Frame(main_frame, bg='#f0f0f0')
        progress_frame.pack(pady=10, padx=20, fill='x')
        
        # Percentage label
        self.percent_label = tk.Label(progress_frame, text="0%", 
                                     font=('Arial', 10, 'bold'), bg='#f0f0f0')
        self.percent_label.pack(pady=(0, 8))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(progress_frame, 
                                           variable=self.progress_var,
                                           maximum=self.max_value,
                                           length=320,
                                           mode='determinate')
        self.progress_bar.pack(pady=(0, 15))
        
        # Button frame
        button_frame = tk.Frame(progress_frame, bg='#f0f0f0')
        button_frame.pack()
        
        # Cancel button
        self.cancel_button = tk.Button(button_frame, text="Cancel", 
                                      command=self.cancel_operation,
                                      width=10, height=1)
        self.cancel_button.pack(side='left', padx=5)
        
        # Close button (initially hidden)
        self.close_button = tk.Button(button_frame, text="Close", 
                                     command=self.close_dialog,
                                     width=10, height=1)
        
    def set_total_files(self, total_files):
        """Set the total number of files to process"""
        self.total_files = total_files
        self.counter_var.set(f"0/{total_files}")
        
    def update_progress(self, value, current_file="", file_number=0):
        """Update progress bar, current file, and counter"""
        self.progress_var.set(value)
        percentage = int((value / self.max_value) * 100)
        self.percent_label.config(text=f"{percentage}%")
        
        # Update counter
        if file_number > 0:
            self.counter_var.set(f"{file_number}/{self.total_files}")
        
        # Update current file display
        if current_file:
            # Truncate filename if too long
            if len(current_file) > 40:
                current_file = current_file[:37] + "..."
            self.current_file_var.set(f"Processing: {current_file}")
        
        self.dialog.update_idletasks()
        
    def cancel_operation(self):
        """Cancel the operation"""
        self.is_cancelled = True
        self.cancel_button.config(state='disabled', text="Cancelling...")
        
    def complete_operation(self):
        """Called when operation completes successfully"""
        self.cancel_button.pack_forget()
        self.close_button.pack(side='left', padx=5)
        self.current_file_var.set("Operation completed successfully!")
        self.counter_var.set(f"{self.total_files}/{self.total_files}")
        
        # Print final status to terminal
        print(f"\n✅ Auto labeling completed successfully! ({self.total_files}/{self.total_files})")
        print("="*60)
        
    def close_dialog(self):
        """Close the dialog"""
        self.dialog.destroy()
        
    def show(self):
        """Show the dialog"""
        self.dialog.focus_set()
        return self.dialog

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
        self.run_button = tk.Button(self.root, text="Run Auto Labeling", font=font_btn, command=self.run_labeling)
        self.run_button.place(x=180, y=510)

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

        # Get list of image files
        image_files = [f for f in os.listdir(self.image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not image_files:
            messagebox.showwarning("No images", "No image files found in the selected folder.")
            return

        # Disable the run button
        self.run_button.config(state='disabled')
        
        # Create progress dialog
        total_steps = len(image_files) * 2  # 2 steps per image (label + draw)
        self.progress_dialog = ProgressDialog(self.root, "Auto Labeling", total_steps)
        self.progress_dialog.set_total_files(len(image_files))
        self.progress_dialog.show()
        
        # Print start message to terminal
        print("="*60)
        print(f"🚀 Starting Auto Labeling Process...")
        print(f"📁 Input folder: {self.image_folder}")
        print(f"📁 Output folder: {self.output_folder}")
        print(f"📏 Bbox size: {self.bbox_size.get()}")
        print(f"🖼️  Total images: {len(image_files)}")
        print("="*60)
        
        # Start processing in a separate thread
        threading.Thread(target=self.process_images, args=(image_files,), daemon=True).start()

    def process_images(self, image_files):
        try:
            # Create output directories
            os.makedirs(self.output_folder, exist_ok=True)
            label_folder = os.path.join(self.output_folder, "labels")
            os.makedirs(label_folder, exist_ok=True)
            
            x, y, w, h = bbox_presets[self.bbox_size.get()]
            current_step = 0
            
            print("\n📝 Phase 1: Generating label files...")
            
            # Step 1: Generate label files with tqdm progress bar
            with tqdm(total=len(image_files), desc="Labeling", unit="file", 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
                
                for i, filename in enumerate(image_files, 1):
                    if self.progress_dialog.is_cancelled:
                        break
                        
                    # Update GUI progress
                    current_step += 1
                    self.progress_dialog.update_progress(current_step, f"Labeling: {filename}", i)
                    
                    # Update terminal progress
                    pbar.set_postfix_str(f"Processing: {filename}")
                    
                    # Generate label
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
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Small delay to show progress
                    time.sleep(0.01)
            
            if self.progress_dialog.is_cancelled:
                print("\n❌ Operation cancelled by user")
                self.cleanup_and_enable()
                return
            
            print("\n🎨 Phase 2: Drawing bounding boxes and saving images...")
            
            # Step 2: Draw bounding boxes and save images with tqdm progress bar
            file_label_folder = os.path.join(self.output_folder, "image_label")
            os.makedirs(file_label_folder, exist_ok=True)
            
            with tqdm(total=len(image_files), desc="Drawing", unit="file",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
                
                for i, filename in enumerate(image_files, 1):
                    if self.progress_dialog.is_cancelled:
                        break
                        
                    # Update GUI progress
                    current_step += 1
                    self.progress_dialog.update_progress(current_step, f"Drawing: {filename}", i)
                    
                    # Update terminal progress
                    pbar.set_postfix_str(f"Processing: {filename}")
                    
                    # Process image
                    img_path = os.path.join(self.image_folder, filename)
                    label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + ".txt")
                    img = cv2.imread(img_path)
                    
                    if img is None or not os.path.exists(label_path):
                        pbar.update(1)
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
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Small delay to show progress
                    time.sleep(0.01)
            
            # Complete or cancelled
            if self.progress_dialog.is_cancelled:
                print("\n❌ Operation cancelled by user")
                self.cleanup_and_enable()
                messagebox.showinfo("Cancelled", "Auto labeling was cancelled.")
            else:
                self.progress_dialog.complete_operation()
                self.show_preview()
                messagebox.showinfo("Success", "Auto labeling completed successfully!")
                
        except Exception as e:
            print(f"\n❌ Error occurred: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.cleanup_and_enable()

    def cleanup_and_enable(self):
        """Re-enable the run button"""
        self.root.after(0, lambda: self.run_button.config(state='normal'))

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