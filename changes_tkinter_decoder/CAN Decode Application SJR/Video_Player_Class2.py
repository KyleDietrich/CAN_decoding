# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 16:48:32 2025

@author: sughosh.rao
"""

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
from Plotting_routines import *
import ast


class VideoPlayerApp:
    def __init__(self, root, video_source):
        self.root = root
        self.root.title("Tkinter Video Player")
        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source) #, cv2.ROTATE_180)

        if not self.cap.isOpened():
            print("Error: Could not open video file.")
            self.root.destroy()
            return

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(self.total_frames)
        self.delay = int(100 / self.fps) if self.fps > 0 else 30
        self.current_frame = 0
        self.playing = False
        self.plot_data_time = []
        self.plot_data_data = []

        # Create main frames
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Create video and plot frames
        video_frame = ttk.Frame(main_frame)
        video_frame.grid(row=0, column=0, padx=5, pady=5)
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.N,tk.E,tk.W, tk.S))

        # Video Display Canvas
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas = tk.Canvas(video_frame, width=width, height=height, bg="black")
        self.canvas.pack()

        # Control Panel Frame
        control_frame = ttk.Frame(video_frame, padding="5")
        control_frame.pack(fill=tk.X, pady=5)
        
        # CAN Frame
        CAN_frame =ttk.Frame(self.plot_frame, padding="5")
        CAN_frame.pack(fill = tk.X, pady=5)

        # Play/Pause button
        self.play_pause_button = ttk.Button(control_frame, text="▶️", command=self.toggle_playback)
        self.play_pause_button.pack(side=tk.LEFT, padx=5)

        # Video slider
        self.slider = ttk.Scale(
            control_frame,
            from_=0,
            to=self.total_frames - 1,
            orient=tk.HORIZONTAL,
            command=self.slider_moved
        )
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # CAN Id, Bytes, Bits list
        self.can_id = tk.StringVar()
        self.byte_list = tk.StringVar()
        self.bits_list = tk.StringVar()
        
        canlabel = tk.Label(CAN_frame, text='CAN ID')
        canlabel.grid(row=0, column = 0, padx=5)
        self.can = tk.Entry(CAN_frame, textvariable = self.can_id)
        self.can.grid(row =0, column=1, padx=2)
        
        bytelabel = tk.Label(CAN_frame, text= 'Bytes List')
        bytelabel.grid(row = 0, column=2, padx =5)
        self.byt= tk.Entry(CAN_frame, textvariable = self.byte_list)
        self.byt.grid(row=0, column=3, padx=2)
        
        bitslabel = tk.Label(CAN_frame, text= 'Bits List of lists')
        bitslabel.grid(row = 0, column=4, padx =5)
        self.bit= tk.Entry(CAN_frame, textvariable = self.bits_list)
        self.bit.grid(row=0, column=5, padx=2)
        
        plot_button = tk.Button(CAN_frame, text="Plot", command=self.press_plot_button)
        plot_button.grid(row=0,column=6, padx=5)
        
        # Current time label
        self.time_label = ttk.Label(control_frame, text="00:00:00 / 00:00:00")
        self.time_label.pack(side=tk.LEFT, padx=5)
        
        # Matplotlib Plot
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=80)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frame Number")
        self.ax.set_title("Video Playback Progression")
        
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.plot_canvas_widget = self.plot_canvas.get_tk_widget()
        self.plot_canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.update_frame()
        
    def press_plot_button(self):
        canid = self.can_id.get()
        byteslist = self.byte_list.get()
        bitslist= self.bits_list.get()
        self.plot_data_time, self.plot_data_data = get_plotting_data(canid, ast.literal_eval(byteslist),
                                                                     ast.literal_eval(bitslist))
        self.update_frame()
        
    def toggle_playback(self):
        self.playing = not self.playing
        if self.playing:
            self.play_pause_button.config(text="⏸️")
            self.playback_thread = threading.Thread(target=self.play_video)
            self.playback_thread.start()
        else:
            self.play_pause_button.config(text="▶️")

    def play_video(self):
        while self.playing and self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.update_frame()
            time.sleep(1 / (10*self.fps))
            
        if self.current_frame >= self.total_frames - 1:
            self.playing = False
            self.play_pause_button.config(text="▶️")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to start

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to a PhotoImage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            # Update slider position
            self.slider.set(self.current_frame)
            
            # Update time label
            elapsed_seconds = self.current_frame / self.fps
            duration_seconds = self.total_frames / self.fps
            self.time_label.config(text=f"{self.format_time(elapsed_seconds)} / {self.format_time(duration_seconds)}")
            
            # Update plot
            self.update_plot()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video

    def slider_moved(self, value):
        new_frame_index = int(float(value))
        if new_frame_index != self.current_frame:
            self.current_frame = new_frame_index
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.update_frame()

    def update_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Data")
        self.ax.set_title("Video Playback Progression")
        
        # time, data = get_plotting_data(canid, byteslist, bitslist)
        
        # Unpack time and frame number from stored data
        #times, frames = zip(*self.plot_data)
        
        self.ax.plot(self.plot_data_time, self.plot_data_data, color='blue')
        #self.ax.axvline(x=times[-1], color='red', linestyle='--', label='Current Time')
        
        self.ax.axvline(x=self.current_frame / self.fps, color='red', linestyle='--', label='Current Time')
        self.ax.legend()
        self.ax.grid(True)
        
        self.plot_canvas.draw()
    
    def format_time(self, seconds):
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def Vid_main(video_file):
    # Replace 'your_video_file.mp4' with the path to your video file
    
    root = tk.Tk()
    app = VideoPlayerApp(root, video_file)
    root.mainloop()

if __name__ == "__main__":
    video_file = "Lexus/Lexus500_LeftTurnSignal.avi"
    Vid_main(video_file)
