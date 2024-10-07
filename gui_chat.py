import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import asyncio
from PIL import Image, ImageTk  # For loading images
import io  # For handling image data

from agentq.__main__ import run_agent_sync

class ChatApp:
    def __init__(self, master):
        self.master = master
        master.title("Agent Q Chat")

        # Chat history display
        self.chat_history = scrolledtext.ScrolledText(master, state='disabled')
        self.chat_history.pack(fill=tk.BOTH, expand=True)

        # Input field and send button
        self.input_frame = tk.Frame(master)
        self.input_frame.pack(fill=tk.X)

        self.user_input = tk.Entry(self.input_frame)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT)

        # Progress indicator (initially hidden)
        self.progress_label = tk.Label(master, text="Processing...")
        self.progress_canvas = tk.Canvas(self.progress_label, width=20, height=20)
        self.progress_canvas.pack(side=tk.LEFT)
        self.progress_label.pack(fill=tk.X, pady=(5, 0), side=tk.BOTTOM)
        self.progress_label.pack_forget()  # Hide initially

        # Load the rotating circle image
        self.load_progress_image()

        # Animation variables
        self.angle = 0
        self.animation_running = False

    def load_progress_image(self):
        try:
            # Load the image (replace 'loading.png' with your image path)
            image = Image.open("loading.png")
            self.progress_image = ImageTk.PhotoImage(image)
        except Exception as e:
            print(f"Error loading progress image: {e}")
            self.progress_image = None

    def display_message(self, message, sender):
        self.chat_history.config(state='normal')
        self.chat_history.insert(tk.END, f"{sender}: {message}\n")
        self.chat_history.config(state='disabled')
        self.chat_history.see(tk.END)  # Scroll to bottom

    def send_message(self, event=None):
        message = self.user_input.get()
        if message:
            self.display_message(message, "User")
            self.user_input.delete(0, tk.END)
            self.start_agent_thread(message)

    def start_agent_thread(self, message):
        self.progress_label.pack(fill=tk.X, pady=(5, 0), side=tk.BOTTOM)
        #self.start_animation()
        thread = threading.Thread(target=self.run_agent, args=(message,))
        thread.daemon = True  # Allow main thread to exit even if agent is running
        thread.start()

    def run_agent(self, message):
        asyncio.set_event_loop(asyncio.new_event_loop())
        result = run_agent_sync(command=message)
        self.master.after(0, self.process_agent_result, result)

    def process_agent_result(self, result):
        self.display_message(result, "Agent Q")
        self.stop_animation()
        self.progress_label.pack_forget()

    def start_animation(self):
        if not self.animation_running and self.progress_image:
            self.animation_running = True
            self.update_animation()

    def stop_animation(self):
        self.animation_running = False

    def update_animation(self):
        if self.animation_running and self.progress_image:
            self.angle += 10  # Adjust rotation speed
            if self.angle >= 360:
                self.angle = 0
            
            # Rotate the image
            rotated_image = self.progress_image.rotate(self.angle, resample=Image.BICUBIC)
            self.progress_canvas.delete("all")
            self.progress_canvas.create_image(10, 10, anchor=tk.CENTER, image=ImageTk.PhotoImage(rotated_image))

            self.master.after(50, self.update_animation)  # Update every 50ms

root = tk.Tk()
app = ChatApp(root)
root.mainloop()
