import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter
import os
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import io
from tkinter.font import Font
import cv2

path = None
segmented_image = None

def showimage():
    global path
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title='Select an Image for Visual Segmentation.', filetypes=(("JPG FILE", "*.jpg"),
                                                                                                  ("PNG file", "*.png"),
                                                                                                  ("All Files", ".")))
    path = fln
    img = Image.open(fln)
    img.thumbnail((700,700))
    img = ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image = img

def seg():
    global segmented_image
    if path:
        seg_img = instanceSegmentation()
        seg_img.load_model("pointrend_resnet50.pkl")
        result = seg_img.segmentImage(path, show_bboxes=True)
        segmented_img_array = result[1]
        segmented_image = Image.fromarray(segmented_img_array)
        segmented_image = segmented_image.resize((680, 680), Image.LANCZOS)
        img_buffer = io.BytesIO()
        segmented_image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        segmented_image = Image.open(img_buffer)
        img_display = ImageTk.PhotoImage(segmented_image)
        lbl2.configure(image=img_display)
        lbl2.image = img_display
    else:
        print("No image file selected")

def fun2():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Could not open video capture.")
        exit()
    segment_video = instanceSegmentation()
    segment_video.load_model("pointrend_resnet50.pkl", confidence=0.7, detection_speed="normal")

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        segmask, output_frame = segment_video.segmentFrame(frame, show_bboxes=True)
        cv2.imshow('Visual Segmentation using DeepLearining', output_frame)
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

ctk.set_appearance_mode("dark") 
ctk.set_default_color_theme("blue")  

root = ctk.CTk()  
root.title("Visual Segmantation.")
root.geometry("1100x850")

font1= ctk.CTkFont(size=50, weight="bold")
font2= ctk.CTkFont(family="Arial",size=30,underline=True,overstrike=False)

label=ctk.CTkLabel( master=root,
                    text="Visual Segmentation",
                    width=180,
                    height=50,
                    fg_color='black',  
                    text_color="white",
                    font=font1,
                    corner_radius=8)
#label.place(x=245,y=10)
label.place(relx=0.5,rely=0.1,anchor=tkinter.CENTER)

btn = ctk.CTkButton(root, text="Browse the Image", command=showimage, font=font2)
btn.place(relx=0.2,rely=0.2,anchor=tkinter.CENTER) 


button = ctk.CTkButton(root, text="Segment the Image", command=seg,font=font2)
button.place(relx=0.5,rely=0.2,anchor=tkinter.CENTER) 

button2 = ctk.CTkButton(root, text="Live Segmentation", command=fun2,font=font2)
button2.place(relx=0.8,rely=0.2,anchor=tkinter.CENTER) 

frm = ctk.CTkFrame(root, width=600, height=250,border_width=5)
#frm.place(x=100,y=120)
frm.place(relx=0.5,rely=0.6,anchor=tkinter.CENTER)

lbl = ctk.CTkLabel(frm, text="")
lbl.grid(row=0, column=0, padx=10, pady=10)

lbl2 = ctk.CTkLabel(frm, text="")
lbl2.grid(row=0, column=1, padx=10, pady=10)

root.mainloop()