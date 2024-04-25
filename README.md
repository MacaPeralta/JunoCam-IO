# JunoCam-IO
Python scripts for processing images of IO taken from JunoCam


Run Functions in ipython environments.

import libraries and read raw image:
from proj1part1 import *
raw = readit()
stk = jcamstk(raw)
g40, sum40 = contrastdemo(stk)

red_channel, green_channel, blue_channel = split_rgb_channels(raw)
manual_array = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) # fill array with roll values called in np.roll()
adjusted_channel = apply_np_roll_to_bands(red_channel, manual_array)


# Plot Red Channel
axes[0].imshow(red_channel, cmap='Reds', aspect='auto')
axes[0].set_title('Red Channel')
axes[0].axis('off')  # Hide axes for cleaner look
# Plot Green Channel
axes[1].imshow(green_channel, cmap='Greens', aspect='auto')
axes[1].set_title('Green Channel')
axes[1].axis('off')
# Plot Blue Channel
axes[2].imshow(blue_channel, cmap='Blues', aspect='auto')
axes[2].set_title('Blue Channel')
axes[2].axis('off')
plt.show()

# Plot Color Image
plt.imshow(color_image)
plt.title('Color Image')
plt.axis('off')
plt.show()





