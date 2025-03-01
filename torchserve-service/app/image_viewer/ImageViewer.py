import json
import random
import tkinter as tk
from tkinter import Canvas, Scrollbar

from PIL import Image, ImageTk


class ImageViewerApp:
    def __init__(self, root, data):
        self.root = root

        if isinstance(data, str):
            self.data = json.loads(data)
        else:
            self.data = data

        self.index = 0

        # Set up the main window
        self.root.title("Image Viewer with Detections")
        self.root.attributes('-fullscreen', True)

        # Create a frame to hold canvas and scrollbars
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas for displaying images
        self.canvas = Canvas(self.frame, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add scrollbars
        self.v_scrollbar = Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.h_scrollbar = Scrollbar(self.root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        # Navigation buttons
        self.prev_button = tk.Button(self.root, text="Prev", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.next_button = tk.Button(self.root, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.close_button = tk.Button(self.root, text="X", command=self.root.quit, fg="white", bg="red")
        self.close_button.pack(side=tk.TOP, anchor=tk.CENTER, padx=10, pady=10)

        # Bind mouse scroll events
        self.canvas.bind_all("<MouseWheel>", self.on_vertical_scroll)
        self.canvas.bind_all("<Shift-MouseWheel>", self.on_horizontal_scroll)

        if isinstance(self.data, dict):
            self.keys = list(self.data.keys())
            self.display_strategy = self._display_image_new
            self.get_length = lambda: len(self.keys)
        elif isinstance(self.data, list):
            self.display_strategy = self._display_image_old
            self.get_length = lambda: len(self.data)

        # Display the first image
        self.display_strategy()

    def _display_image_new(self):
        # Clear the canvas
        self.canvas.delete("all")

        # Get the current UUID and image data
        current_uuid = self.keys[self.index]
        image_data = self.data[current_uuid]
        image_path = current_uuid  # UUID is the filepath
        detections = image_data['detections']

        self._render_image_and_detections(image_path, detections)

    def _display_image_old(self):
        # Clear the canvas
        self.canvas.delete("all")

        # Get the current image data
        image_data = self.data[self.index]
        image_path = image_data['uuid']
        detections = image_data['detections']

        self._render_image_and_detections(image_path, detections)

    def _render_image_and_detections(self, image_path, detections):
        # Open the image
        image = Image.open(image_path)

        # Convert the image for Tkinter
        self.tk_image = ImageTk.PhotoImage(image)

        # Set canvas scroll region to match image size
        self.canvas.config(scrollregion=(0, 0, image.width, image.height))

        # Align the image to the top-left corner
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Draw detections without scaling
        for detection in detections:
            box = detection['box']
            label = detection['label']
            score = detection['score']

            # Assign a random color for the box
            color = "#%06x" % random.randint(0, 0xFFFFFF)

            # Draw the rectangle and place the label
            self.canvas.create_rectangle(
                box[0], box[1], box[2], box[3], outline=color, width=3
            )
            self.canvas.create_text(
                box[2] - 5, box[1], anchor=tk.NE, text=f"{label} ({score:.2f})", fill=color
            )

    def next_image(self):
        # Go to the next image
        self.index = (self.index + 1) % self.get_length()
        self.display_strategy()

    def prev_image(self):
        # Go to the previous image
        self.index = (self.index - 1) % self.get_length()
        self.display_strategy()

    def on_vertical_scroll(self, event):
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def on_horizontal_scroll(self, event):
        self.canvas.xview_scroll(-1 * (event.delta // 120), "units")


if __name__ == "__main__":
    old_format = [{'uuid': '../shared/data/b/m/001.jpg',
                   'detections': [
                       {'box': [1345, 184, 2085, 2109], 'label': 'person', 'score': 1.0},
                       {'box': [1900, 641, 2444, 1108], 'label': 'potted plant', 'score': 1.0},
                       {'box': [1, 260, 562, 1714], 'label': 'potted plant', 'score': 1.0},
                       {'box': [1322, 1127, 3112, 1993], 'label': 'couch', 'score': 0.99},
                       {'box': [2093, 969, 2260, 1121], 'label': 'vase', 'score': 0.91},
                       {'box': [254, 1131, 1254, 1919], 'label': 'couch', 'score': 0.8}
                   ]}]

    new_format = (
        '{"../shared/data/b/k/8546001.jpg": {"detections": [{"box": [348, 167, 2464, 3998], "label": "person", '
        '"score": 1.0}, {"box": [334, 1662, 977, 3277], "label": "handbag", "score": 0.83}, {"box": [631, 2439, '
        '1888, 3661], "label": "tie", "score": 0.81}], "size": [4000, 2662]}}')

    root = tk.Tk()
    app = ImageViewerApp(root, new_format)
    root.mainloop()
