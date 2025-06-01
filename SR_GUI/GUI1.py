import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageZoomerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Super Resolution")

        self.canvas = tk.Canvas(self.master, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.image = None
        self.start_x = self.start_y = 0
        self.end_x = self.end_y = 0
        self.selection_box = None

        
        # Bind mouse click events
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.create_text(self.canvas.winfo_reqwidth()/2, self.canvas.winfo_reqheight()/2, text="Welcome to Super Resolution", font=("Helvetica", 16), anchor=tk.CENTER)
        
        # Add buttons
        self.browse_button = tk.Button(self.master, text="Browse Image", command=self.select_image)
        self.browse_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.zoom_2x_button = tk.Button(self.master, text="2.5X", command=lambda: self.perform_zoom(2.5))
        self.zoom_2x_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.zoom_4x_button = tk.Button(self.master, text="4X", command=lambda: self.perform_zoom(4.0))
        self.zoom_4x_button.pack(side=tk.LEFT, padx=5, pady=5)

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])

        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        self.image = Image.open(file_path)
        self.display_image()

    def display_image(self):
        tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        self.canvas.tk_image = tk_image  # To prevent garbage collection

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

    def on_mouse_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        self.canvas.delete("selection_box")
        self.selection_box = self.canvas.create_rectangle(self.start_x, self.start_y, cur_x, cur_y, outline="red", tags="selection_box")

    def on_button_release(self, event):
        self.end_x = self.canvas.canvasx(event.x)
        self.end_y = self.canvas.canvasy(event.y)

    def perform_zoom(self, zoom_factor):
        if self.image and self.selection_box:
            selected_area = self.image.crop((self.start_x, self.start_y, self.end_x, self.end_y))
            print(selected_area.width, selected_area.height)
            zoomed_image = selected_area.resize((int(selected_area.width * zoom_factor), int(selected_area.height * zoom_factor)), Image.BICUBIC)
            print(zoomed_image.width, zoomed_image.height)

            # Remove the selection box after zooming
            self.canvas.delete("selection_box")

            # Display zoomed image in a new window
            top = tk.Toplevel(self.master)
            top.title(f"{zoom_factor}X Zoom")

            tk_image = ImageTk.PhotoImage(zoomed_image)
            label = tk.Label(top, image=tk_image)
            label.image = tk_image
            label.pack()

    def run(self):
        self.master.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageZoomerApp(root)
    app.run()
