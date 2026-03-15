import os
from PIL import Image, ImageDraw  # pyre-ignore

def create_icon(size, output_path):
    # Create transparent background
    img = Image.new('RGBA', (size, size), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a green circle
    margin = int(size * 0.05)
    draw.ellipse([margin, margin, size - margin, size - margin], fill="#22c55e")
    
    # Draw a white inner shape (simple triangle for sorting/recycling motif)
    w_margin = int(size * 0.3)
    draw.polygon([
        (w_margin, size - w_margin), 
        (size - w_margin, size - w_margin), 
        (size / 2, w_margin)
    ], fill="white")
    
    img.save(output_path)
    print(f"Generated {output_path}")

web_public_dir = os.path.join(os.path.dirname(__file__), '..', 'web', 'public')
os.makedirs(web_public_dir, exist_ok=True)

create_icon(192, os.path.join(web_public_dir, 'icon-192.png'))
create_icon(512, os.path.join(web_public_dir, 'icon-512.png'))
