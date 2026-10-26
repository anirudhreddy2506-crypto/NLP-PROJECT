"""
Utility script to create benchmark sample images for VisionCaption AI demonstrations and testing.
"""

import os
from PIL import Image, ImageDraw


def create_sample_images(output_dir: str = "data/sample_images") -> None:
    """
    Creates synthetic test images representing common captioning scenes:
    1. A sunny outdoor landscape with a house, tree, and sun
    2. A simple indoor still life with a fruit bowl
    """
    os.makedirs(output_dir, exist_ok=True)

    # Image 1: Sunny outdoor landscape
    img1 = Image.new("RGB", (640, 480), color=(135, 206, 235))  # Sky blue
    draw1 = ImageDraw.Draw(img1)
    # Bright sun
    draw1.ellipse([480, 40, 580, 140], fill=(255, 215, 0), outline=(255, 165, 0), width=3)
    # Green grass/hill
    draw1.rectangle([0, 300, 640, 480], fill=(34, 139, 34))
    # Red house body
    draw1.rectangle([150, 240, 320, 380], fill=(178, 34, 34))
    # Triangular brown roof
    draw1.polygon([(130, 240), (235, 150), (340, 240)], fill=(139, 69, 19))
    # Door and window
    draw1.rectangle([210, 300, 260, 380], fill=(245, 245, 220))
    draw1.rectangle([170, 260, 200, 290], fill=(255, 255, 255))
    draw1.rectangle([270, 260, 300, 290], fill=(255, 255, 255))
    # Tree trunk and leaves
    draw1.rectangle([420, 260, 450, 370], fill=(101, 67, 33))
    draw1.ellipse([380, 180, 490, 280], fill=(46, 125, 50))

    img1_path = os.path.join(output_dir, "sunny_house_landscape.jpg")
    img1.save(img1_path, "JPEG", quality=95)
    print(f"Created sample image: {img1_path}")

    # Image 2: Still life with red apple on a desk
    img2 = Image.new("RGB", (640, 480), color=(240, 240, 240))  # Studio wall
    draw2 = ImageDraw.Draw(img2)
    # Wooden table top
    draw2.rectangle([0, 280, 640, 480], fill=(160, 82, 45))
    # Plate
    draw2.ellipse([180, 290, 460, 420], fill=(255, 255, 255), outline=(200, 200, 200), width=4)
    # Red apple
    draw2.ellipse([270, 270, 370, 370], fill=(220, 20, 60))
    # Green stem
    draw2.line([320, 270, 320, 245], fill=(34, 139, 34), width=5)

    img2_path = os.path.join(output_dir, "apple_on_table.png")
    img2.save(img2_path, "PNG")
    print(f"Created sample image: {img2_path}")


if __name__ == "__main__":
    create_sample_images()
