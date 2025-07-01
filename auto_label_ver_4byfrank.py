import cv2
from ultralytics import YOLO
from tkinter import messagebox
from tkinter import ttk
import tkinter as tk
from tkinter.colorchooser import *
import tkinter.filedialog
import customtkinter as ctk
import os
import shutil
from tqdm import tqdm
import random

class program :
    def save_random_preview(self, img_detection, filename, output_path, sample_rate=0.2):
        if random.random() < sample_rate:  # 20% sampling
            preview_path = os.path.join(output_path, "preview")
            os.makedirs(preview_path, exist_ok=True)
            cv2.imwrite(f"{preview_path}/{filename}.jpg", img_detection)
    
    def __init__(self) :
        self.class_list = ["canister", "foam", "ring", "tyvek", "wafer"]
        self.color_map = {
            0: (0, 255, 255),    # canister
            1: (255, 0, 255),    # foam
            2: (0, 255, 0),    # ring
            3: (0, 0, 255),  # tyvek
            4: (255, 0, 0),  # wafer
        }

    def detect_class_id_from_filename(self, filename):
        filename_lower = filename.lower()
        for i, cls_name in enumerate(self.class_list):
            if filename_lower.startswith(cls_name + "_"):
                return i  # return index id
        return 0  # ถ้าไม่เจอ class ให้ default เป็น 0
    
    def set_model(self,model) :
        self.model = YOLO(model)
    
    def set_class(self, class_name=0) :
        self.class_name = class_name
    
    def set_input_path(self, input_path) :
        self.input_path = input_path

    def set_output_path(self, output_path) :
        self.output_path = output_path

    def draw_boxes(self,frame, detections):
        for det in detections:
            box = det.xyxy[0].cpu().numpy()  # xyxy format for bounding box coordinates
            label = self.model.names[int(det.cls[0])]  # class label
            confidence = det.conf[0].item()  # confidence score
            # Draw bounding box
            #cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            # Draw label and confidence
            #cv2.putText(frame, f'face {confidence:.2f}', (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame
    
    def create_folder(self,folder_path):
        try:
            # Create a new folder
            os.makedirs(folder_path, exist_ok=True)
            print(f"Folder created: {folder_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def list_files_in_folder(self,folder_path):
        list_name_file = []
        # List all files in the specified folder
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file():
                    list_name_file.append(entry.name)
        return list_name_file

    def find_file_name(self,file_path) :
        # Get the file name with extension
        file_name_with_extension = os.path.basename(file_path)

        # Split the file name and extension
        file_name, file_extension = os.path.splitext(file_name_with_extension)

        return file_name

    def create_output_folder(self, output_path) :
        detection_path = "{path}/output/detections".format(path = output_path)
        no_detection_path = "{path}/output/no detections".format(path = output_path)
        label_path = "{path}/output/labels".format(path = output_path)
        image_path = "{path}/output/images".format(path = output_path)
        self.no_detection_label_path = "{path}/output/no detections label".format(path = output_path)
        self.create_folder(detection_path)
        self.create_folder(no_detection_path)
        self.create_folder(label_path)
        self.create_folder(image_path)
        self.create_folder(self.no_detection_label_path)
        self.detection_path = detection_path
        self.label_path = label_path
        self.image_path = image_path
        self.no_detection_path = no_detection_path

    def create_report_folder(self, output_path,no_detection_file = True, more_than_two_detection = True) :
        path = r"{path}/report".format(path = output_path)
        self.create_folder(path)
        self.no_detection_file = no_detection_file
        self.more_than_two_detection = more_than_two_detection
        self.report_path = path
        # if no_detection_file :
        #     open("{f}.txt".format(f = "no_detection"),"w")
        # if more_than_two_detection :
        #     open("{f}.txt".format(f = "more_than_two_detection"),"w")

    def copy_files(self,source_path, output_path) :
        shutil.copy(source_path, output_path)
        #print(f"File copied from {source_path} to {output_path}")
        return
    
    def check_w_h_range(self, w, h):
        w_ranges = [(0.35, 0.45), (0.49, 0.59)]
        w_valid = any(lower <= w <= upper for (lower, upper) in w_ranges)
        
        h_ranges = [(0.57, 0.73), (0.77, 0.93)]
        h_valid = any(lower <= h <= upper for (lower, upper) in h_ranges)

        if w_valid and h_valid:
            return True
        else:
            return False
        
    def check_single_detection(self, detections):
        return detections.xywh.shape[0] == 1

    def run(self) :
        count = 0
        model = self.model
        #input path
        input_path = self.input_path
        #output path
        output_path = self.output_path
        label_path = self.label_path
        image_path = self.image_path
        detection_path = self.detection_path
        no_detection_path = self.no_detection_path
        #report path
        report_path = self.report_path
        #class name
        class_name = self.class_name
        #set datalog
        no_detection_file = self.no_detection_file
        more_than_two_detection = self.more_than_two_detection
        #list of input file name
        list_file_name = self.list_files_in_folder(input_path)
        
        for i in tqdm(range((len(list_file_name))), desc="Processing") :
            #read image
            img = cv2.imread("{path}/{f}".format(path = input_path,f = list_file_name[i])) 
            #get h and w of image
            height, width = img.shape[:2]
            #prediction
            results = model(img, conf = 0.5, verbose = False)
            detections = results[0].boxes
            cls_t = detections.cls
            if not self.check_single_detection(detections):
                # ถ้ามีมากกว่า 1 detection → จัดเป็น no detection
                no_detection_file = open("{path}/{f}.txt".format(path = report_path,f = "no_detection_more_than_one"),"a")
                no_detection_file.write("{f} \n".format(f = self.find_file_name(list_file_name[i])))
                no_detection_file.close()
                self.copy_files("{path}/{f}".format(path = input_path, f = list_file_name[i]),"{path}/{f}".format(path = no_detection_path , f = list_file_name[i]))
                continue  # skip การประมวลผลภาพนี้ทันที
            if no_detection_file :
                if cls_t.numel() == 0 :
                    #create file log-no detection
                    no_detection_file = open("{path}/{f}.txt".format(path = report_path,f = "no_detection"),"a")
                    no_detection_file.write("{f} \n".format(f = self.find_file_name(list_file_name[i])))
                    self.copy_files("{path}/{f}".format(path = input_path, f = list_file_name[i]),"{path}/{f}".format(path = no_detection_path , f = list_file_name[i]))
                    no_detection_file.close()
                    #create label
                    with open("{no_detection_label_path}/{f}.txt".format(no_detection_label_path = self.no_detection_label_path, f = self.find_file_name(list_file_name[i])), "w") as file:
                        file.write("")
                    #print("no detection")
                    #print("the path of this file is {path}\{f}".format(path = folder_path, f = list_file_name[i]))
            if cls_t.numel() == 0 :
                pass
            else :
                x_t,y_t,w_t,h_t = detections.xywh[0]
                #print(detections.xywh)
                # Get the dimensions of the tensor
                rows, cols = detections.xywh.shape
                # labeling
                for j in range(rows) :
                    x_t,y_t,w_t,h_t = detections.xywh[j]
                    x = x_t.item()/width
                    y = y_t.item()/height
                    w = w_t.item()/width
                    h = h_t.item()/height

                    if not self.check_w_h_range(w, h):
                        # ถ้าไม่อยู่ในช่วง → จัดไฟล์เป็น no detection
                        no_detection_file = open("{path}/{f}.txt".format(path = report_path,f = "no_detection_out_of_range"),"a")
                        no_detection_file.write("{f} \n".format(f = self.find_file_name(list_file_name[i])))
                        no_detection_file.close()
                        self.copy_files("{path}/{f}".format(path = input_path, f = list_file_name[i]),"{path}/{f}".format(path = no_detection_path , f = list_file_name[i]))
                        continue  # ข้ามการสร้าง label สำหรับ bounding box นี้
                    # detection
                    #strat
                    start_x = round(x_t.item()) - round(w_t.item()//2) 
                    start_y = round(y_t.item()) - round(h_t.item()//2)
                    #print(start_x,start_y)
                    #end
                    end_x = round(start_x + w_t.item()) 
                    end_y = round(start_y + h_t.item()) 
                    #print(end_x,end_y) 
                    img_detection = img

                    #cv2.rectangle(img_detection,(start_x,start_y),(end_x,end_y),(255,0,0),2)

                    # ดึง class id จาก filename (ตามระบบคุณ)
                    class_id = self.detect_class_id_from_filename(list_file_name[i])
                    # ดึงสีของ class จาก color_map
                    color = self.color_map[class_id]
                    # วาดกรอบด้วยสีของแต่ละ class
                    cv2.rectangle(img_detection, (start_x, start_y), (end_x, end_y), color, 2)
                    # ใส่ชื่อ class + confidence (สมมุติ confidence ไม่มีในตอนนี้ใส่ class name ก่อน)
                    #label_text = f"{self.class_list[class_id]}"
                    #cv2.putText(img_detection, label_text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # ดึง confidence ของแต่ละ detection (สำคัญมาก ตรงนี้เพิ่มใหม่)
                    confidence = detections.conf[j].item()

                    # แสดงทั้ง class name และ confidence
                    label_text = f"{self.class_list[class_id]} {confidence:.2f}"
                    cv2.putText(img_detection, label_text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    cv2.imwrite("{path}/{f}.jpg".format(path = detection_path,f=self.find_file_name(list_file_name[i])),img_detection)
                    #create file
                    file = open("{path}/{f}.txt".format(path = label_path , f = self.find_file_name(list_file_name[i])), "a")
                    #write file
                    class_id = self.detect_class_id_from_filename(list_file_name[i])
                    file.write("{cls} {x} {y} {w} {h} \n".format(cls = class_id, x = x, y = y, w = w, h = h))
                    #file.write("{cls} {x} {y} {w} {h} \n".format(cls = int(class_name),x = x,y = y,w = w,h = h))
                    #close file
                    file.close()
            rows, cols = detections.xywh.shape
            if more_than_two_detection :
                if rows >= 2 : 
                    more_than_two_detection_file = open("{path}/{f}.txt".format(path = report_path, f = "more_than_two_detection"),"a") 
                    more_than_two_detection_file.write("{f} \n".format(f =  self.find_file_name(list_file_name[i])))
                    more_than_two_detection_file.close()
            self.copy_files("{path}/{f}".format(path = input_path, f = list_file_name[i]), "{path}/{f}".format(path = image_path , f = list_file_name[i]))
            count += 1
            percent = count/len(list_file_name)
            self.percent = f'{percent*100:.2f}%'
            self.save_random_preview(img_detection, self.find_file_name(list_file_name[i]), output_path)




class app :
    def __init__(self) :
        pass
    
    def var(self) :
        self.model_path = tk.StringVar()
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.detection_path = tk.StringVar()
        self.report_path = tk.StringVar()
        self.percent_var = tk.DoubleVar()
        
    def select_model_folder(self) :
        folder =  tkinter.filedialog.askopenfilename()
        self.model_path.set(folder)
        if folder == "" :
            self.model_path.set("{emtry}")
        elif self.model_path.get() != "{emtry}" :
            self.auto_label.set_model(r"{modelpath}".format(modelpath = self.model_path.get()))
            self.window_app.update_idletasks() 
            self.window_app.geometry(f"{self.window_app.winfo_reqwidth()+10}x{self.window_app.winfo_reqheight()}")

    def select_input_folder(self) :
        folder = tkinter.filedialog.askdirectory()
        self.input_path.set(folder)
        if folder == "" :
            self.input_path.set("{emtry}")
        elif self.input_path.get() != "{emtry}" :
            self.auto_label.set_input_path(r"{input_path}".format(input_path = self.input_path.get()))
            self.window_app.update_idletasks() 
            self.window_app.geometry(f"{self.window_app.winfo_reqwidth()+10}x{self.window_app.winfo_reqheight()}")
            

    def select_output_path(self) :
        folder = tkinter.filedialog.askdirectory()
        self.output_path.set(folder)
        if folder == "" :
            self.output_path.set("{emtry}")
        elif self.output_path.get() != "{emtry}" : 
            self.auto_label.set_output_path(r"{output_path}".format(output_path = self.output_path.get()))
            self.window_app.update_idletasks() 
            self.window_app.geometry(f"{self.window_app.winfo_reqwidth()+10}x{self.window_app.winfo_reqheight()}")
            self.create_detection_folder_name()
            self.create_report_folder_name()

    def create_detection_folder_name(self) :
        folder_name = "{path}/output/{f}".format(path = self.output_path.get(), f = "detection")
        self.detection_path.set(folder_name)
        self.window_app.update_idletasks() 
        self.window_app.geometry(f"{self.window_app.winfo_reqwidth()+10}x{self.window_app.winfo_reqheight()}")
    
    def create_detection_folder(self) :
        folder_name = "{path}/{f}".format(path = self.output_path, f = "detection")
        os.makedirs(folder_name, exist_ok=True)
        self.window_app.update_idletasks() 
        self.window_app.geometry(f"{self.window_app.winfo_reqwidth()+10}x{self.window_app.winfo_reqheight()}")

    def create_report_folder_name(self) :
        folder_name = "{path}/output/{f}".format(path = self.output_path.get(), f = "report")
        self.report_path.set(folder_name)
        self.window_app.update_idletasks() 
        self.window_app.geometry(f"{self.window_app.winfo_reqwidth()+10}x{self.window_app.winfo_reqheight()}")
    
    def start_btn(self) :
        self.auto_label.create_output_folder(self.output_path.get())
        self.auto_label.create_report_folder(self.output_path.get())
        #messagebox.showinfo("Info", "The process in runing plase wait")
        #self.show_custom_message_box()
        self.auto_label.run()
        messagebox.showinfo("Info", "The process is finish")
        

    def about_menu_bar(self) :
        messagebox.showinfo("Info","putttphong.benjachaiwat@digitaldevices.com")


    def version_info(self) :
        messagebox.showinfo("Info","for python 3.10.0 and ultralytics official only")
    
    def window(self) :
        app = ctk.CTk()
        self.window_app = app
        app.title("Auto labeling")
        ctk.set_appearance_mode("light")
        #var
        self.var()
        self.auto_label = program()
        self.auto_label.set_class()
        # Create a menu bar
        style = ttk.Style()
        style.configure('Custom.TMenubar', background='#336699')  # Set background color here
        menubar = tk.Menu(app)
        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(label="developer", command=self.about_menu_bar)
        about_menu.add_command(label="version", command = self.version_info)
        # Add the menus to the menu bar
        
        menubar.add_cascade(label="about", menu=about_menu)

        #Label
        model_label = ctk.CTkLabel(app, text="Model : ", fg_color="transparent")
        input_label = ctk.CTkLabel(app, text="Input Folder path : ", fg_color="transparent")
        output_label = ctk.CTkLabel(app, text="Output Folder path : ", fg_color="transparent")
        report_label = ctk.CTkLabel(app, text="report path ", fg_color="transparent")
        
        #label output
        show_model_label = ctk.CTkLabel(app, textvariable = self.model_path, fg_color="transparent",width=350)
        show_input_label = ctk.CTkLabel(app,  textvariable = self.input_path, fg_color="transparent",width=350)
        show_output_label = ctk.CTkLabel(app,  textvariable = self.output_path, fg_color="transparent",width=350)
        check_detection_path = ctk.CTkLabel(app, textvariable = self.detection_path, fg_color="transparent",width=350)
        report_path = ctk.CTkLabel(app, textvariable = self.report_path, fg_color="transparent",width=350)
        
        #button
        model_path_button = ctk.CTkButton(app, text = "...", width = 10,command=self.select_model_folder)
        input_path_button = ctk.CTkButton(app, text = "...", width = 10, command=self.select_input_folder)
        output_path_button = ctk.CTkButton(app, text = "...", width = 10, command= self.select_output_path)
        start_button = ctk.CTkButton(app, text="start",width = 20,fg_color = "green",command = self.start_btn)
        
        #check box
        rectangle = ctk.IntVar()
        rectangle.set(1)
        rectangle_check_box = ctk.CTkCheckBox(app, text="check detection",variable = rectangle)
        no_detection_report = ctk.IntVar()
        no_detection_report.set(1)
        report_check_box = ctk.CTkCheckBox(app, text = "no detection report", variable = no_detection_report)
        more_than_two = ctk.IntVar()
        more_than_two.set(1)
        more_than_two_check_box = ctk.CTkCheckBox(app, text = " more than two detection report", variable = more_than_two)
        
        #pack
        #model
        model_label.grid(row = 1,column = 0,padx = 5,pady = 5,sticky = 'e')
        show_model_label.grid(row = 1, column = 1,padx = 5, pady = 5)
        model_path_button.grid(row = 1, column = 2, padx = 5, pady = 5,sticky = 'w')
        
        #input
        input_label.grid(row = 2,column = 0,padx = 5,pady = 5,sticky = 'e')
        show_input_label.grid(row = 2, column = 1, padx = 5, pady = 5)
        input_path_button.grid(row = 2, column = 2, padx = 5, pady = 5,sticky = 'w')
        
        #output
        output_label.grid(row = 3, column = 0, padx = 5, pady = 5,sticky = 'e')
        show_output_label.grid(row = 3, column = 1, padx = 5, pady = 5)
        output_path_button.grid(row = 3, column = 2, padx = 5, pady = 5,sticky = 'w')
        
        #report
        report_label.grid(row = 6, column = 0, padx = 5, pady = 5, sticky = 'e')
        report_path.grid(row = 6, column = 1)
        
        #check box
        rectangle_check_box.grid(row = 5, column = 0,padx = 5, pady = 5,sticky = 'e')
        check_detection_path.grid(row = 5, column = 1)
        report_check_box.grid(row = 7, column = 0, sticky = 'w',padx = 5, pady = 5)
        more_than_two_check_box.grid(row = 7,column = 1, columnspan = 2,sticky = 'w',padx = 5, pady = 5)
        
        #start btn
        start_button.grid(row = 8, column = 2, sticky = 'w', pady = 5)
        
        #btn command
        self.model_path.set(r"{emtry}")
        self.input_path.set(r"{emtry}")
        self.output_path.set(r"{emtry}")
        self.detection_path.set(r"{emtry}")
        self.report_path.set(r"{emtry}")

        app.config(menu=menubar)
        app.geometry("570x270+650+200")
        app.mainloop()


auto_label_app = app()
auto_label_app.window()