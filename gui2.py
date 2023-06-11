import tkinter as tk
from PIL import ImageTk, Image

def on_drag_start(event):
    """Start the drag operation."""
    widget = event.widget
    widget.start_drag_pos = (event.x, event.y)

def on_drag_motion(event):
    """Handle the drag motion."""
    widget = event.widget
    x, y = widget.start_drag_pos
    widget.place(x=event.x - x + widget.winfo_x(), y=event.y - y + widget.winfo_y())

def on_drag_end(event):
    """End the drag operation."""
    widget = event.widget
    del widget.start_drag_pos
    
def add_letter(letter):
    # Load and display the image
    filename = image_files.get(letter.upper())
    if filename:
        image = Image.open(filename)
        image = image.resize((200, 200), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        image_widget = tk.Label(root, image=photo)
        image_widget.image = photo
        image_widget.bind("<Button-1>", on_drag_start)
        image_widget.bind("<B1-Motion>", on_drag_motion)
        image_widget.bind("<ButtonRelease-1>", on_drag_end)
        image_widget.pack(side=tk.LEFT, padx=10, pady=10)
        
def on_key_press(event):
    global detect_text
    letter = event.char
    detect_text += letter
    add_letter(letter)
    
root = tk.Tk()

# Dictionary of image filenames
image_files = {'C': "letters_1/C.png", 'I': "letters_1/I.png", 'N': "letters_1/N.png", 'S': "letters_1/S.png", 'U': "letters_1/U.png"}

# Create a list to hold the image widgets
image_widgets = []

detect_text = ""

root.bind("<KeyPress>", on_key_press)
root.mainloop()
