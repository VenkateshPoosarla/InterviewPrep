from rembg import remove
from PIL import Image

# Define input and output paths
input_path = 'input.png'
output_path = 'output.png'

# Open the image
input_image = Image.open(input_path)

# Remove the background
output_image = remove(input_image)

# Save the result
output_image.save(output_path)
