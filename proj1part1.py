import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


def readit():

    flistname = 'FLIST'			# 2024.mar.15 5pm
    inf = open(flistname, 'r')		# Open a file of filenames.
    fname1 = inf.readlines()[0][:-1]	# Read the name of the PNG data file.
    raw = np.fliplr(plt.imread(fname1))	# Orient the pixels so that Y increases Up the screen
    #					# and X increases to the Right.
    np.save('raw', raw)			# Save the file in numpy's binary format.
    print(f'{raw.shape=}')		# Describe the array to the user.

    return raw

def jcamstk(raw):

    ny, nx = raw.shape			# Note the dimensions of the raw image.
    nbands = int(ny/128)		# Given that junocam color bands have 128 lines..

    stk = np.zeros((128, nx, nbands))	#..allocate an empty 3-D array.

    for i in range(nbands):		# Stack the bands in the new array.
        stk[:,:,i] = raw[i*128:(i+1)*128, :]

    return stk				# And give it to the user.

def contrastdemo(stk):

    g40 = stk[:,:1600,40]			# Choose a band that's mostly full.
    sum40 = np.sum(g40, axis=0)			# <- Total value of each column in this band.
    for i in range(g40.shape[0]):		# For each row (or line) in this 2-D array..
        g40[i,:] /= sum40			#..divide the entire row by that total.


    fig, ((ax1, ax2)) = plt.subplots(2,1)	# Open a window (a matplotlib "figure")
    ax1.plot(sum40)				# Show the normalization array.
    ax1.set_title(f'Mean values across this band.')	# Label the graph.
    ax2.imshow(np.clip(g40, 0.007, 0.009))	# Display the normalized band, forcing the..
    ax2.set_title(f'"Normalized" brightness.')	#..contrast limits to 

    return g40, sum40

def split_rgb_channels(image_array):
    """
    Splits a raw JunoCam image into red, green, and blue channels, assuming
    the bands are in a repeating pattern.

    Parameters:
        image_array (numpy.ndarray): The image array with bands stacked vertically.

    Returns:
        tuple: A tuple containing the red, green, and blue channel images, each one a stack of its respective color bands.
    """
    # Number of lines per band
    lines_per_band = 128
    
    # Calculate the total number of bands in the image
    total_bands = image_array.shape[0] // lines_per_band
    
    # Initialize empty lists to collect the bands
    red_bands = []
    green_bands = []
    blue_bands = []
    
    # Extract the bands, assuming they are repeating in RGB order
    for i in range(0, total_bands, 3):
        red_bands.append(image_array[i*lines_per_band : (i+1)*lines_per_band])
        green_bands.append(image_array[(i+1)*lines_per_band : (i+2)*lines_per_band])
        blue_bands.append(image_array[(i+2)*lines_per_band : (i+3)*lines_per_band])
  

    # Stack the bands to create three images
    red_channel = np.vstack(red_bands)
    green_channel = np.vstack(green_bands)
    blue_channel = np.vstack(blue_bands)
    
    return red_channel, green_channel, blue_channel

def combine_two_channels(channel1, channel2, 
                         channel1_offset, channel2_offset, 
                         channel1_shift, channel2_shift,
                         channel1_rotate, channel2_rotate):
    """
    Combines two separate image channels into a single two-channel image after rotating and applying 
    vertical and horizontal offsets to each channel. Normalizes the channels to the same scale before combining.
    
    Parameters:
        channel1 (numpy.ndarray): The first image channel.
        channel2 (numpy.ndarray): The second image channel.
        channel1_offset (int): The vertical offset for the first channel.
        channel2_offset (int): The vertical offset for the second channel.
        channel1_shift (int): The horizontal offset for the first channel.
        channel2_shift (int): The horizontal offset for the second channel.
        channel1_rotate (float): The rotation angle in degrees for the first channel.
        channel2_rotate (float): The rotation angle in degrees for the second channel.
        
    Returns:
        numpy.ndarray: The combined two-channel image.
    """
    # Helper function to apply offsets
    def apply_offsets(channel, vert_offset, horiz_offset):
        # Apply vertical offset
        if vert_offset > 0:
            channel = np.vstack((channel[vert_offset:], np.zeros((vert_offset, channel.shape[1]))))
        elif vert_offset < 0:
            channel = np.vstack((np.zeros((-vert_offset, channel.shape[1])), channel[:vert_offset]))
        
        # Apply horizontal offset
        if horiz_offset > 0:
            channel = np.hstack((np.zeros((channel.shape[0], horiz_offset)), channel[:, :-horiz_offset]))
        elif horiz_offset < 0:
            channel = np.hstack((channel[:, -horiz_offset:], np.zeros((channel.shape[0], -horiz_offset))))
        
        return channel

    # Rotate channels
    channel1_rotated = rotate(channel1, channel1_rotate, reshape=False, mode='constant')
    channel2_rotated = rotate(channel2, channel2_rotate, reshape=False, mode='constant')
    
    # Apply offsets to the rotated channels
    channel1_shifted = apply_offsets(channel1_rotated, channel1_offset, channel1_shift)
    channel2_shifted = apply_offsets(channel2_rotated, channel2_offset, channel2_shift)
    
    # Normalize the shifted channels
    channel1_norm = channel1_shifted / channel1_shifted.max() if channel1_shifted.max() > 0 else channel1_shifted
    channel2_norm = channel2_shifted / channel2_shifted.max() if channel2_shifted.max() > 0 else channel2_shifted
    
    # Combine the two normalized channels into a single image
    combined_image = np.dstack((channel1_norm, channel2_norm, np.zeros_like(channel1_norm)))
    
    return combined_image


def combine_rgb_channels(red_channel, green_channel, blue_channel, 
                         red_offset, green_offset, blue_offset, 
                         red_shift, green_shift, blue_shift,
                         red_rotate, green_rotate, blue_rotate):
    """
    Combines separate red, green, and blue channels into a single color image after rotating and applying 
    vertical and horizontal offsets to each channel. Normalizes the channels to the same scale before combining.
    
    Parameters:
        red_channel (numpy.ndarray): The red channel of the image.
        green_channel (numpy.ndarray): The green channel of the image.
        blue_channel (numpy.ndarray): The blue channel of the image.
        red_offset (int): The vertical offset for the red channel.
        green_offset (int): The vertical offset for the green channel.
        blue_offset (int): The vertical offset for the blue channel.
        red_shift (int): The horizontal offset for the red channel.
        green_shift (int): The horizontal offset for the green channel.
        blue_shift (int): The horizontal offset for the blue channel.
        red_rotate (float): The rotation angle in degrees for the red channel.
        green_rotate (float): The rotation angle in degrees for the green channel.
        blue_rotate (float): The rotation angle in degrees for the blue channel.
        
    Returns:
        numpy.ndarray: The combined color image.
    """
    # Helper function to apply offsets
    def apply_offsets(channel, vert_offset, horiz_offset):
        # Apply vertical offset
        if vert_offset > 0:
            channel = np.vstack((channel[vert_offset:], np.zeros((vert_offset, channel.shape[1]))))
        elif vert_offset < 0:
            channel = np.vstack((np.zeros((-vert_offset, channel.shape[1])), channel[:vert_offset]))
        
        # Apply horizontal offset
        if horiz_offset > 0:
            channel = np.hstack((np.zeros((channel.shape[0], horiz_offset)), channel[:, :-horiz_offset]))
        elif horiz_offset < 0:
            channel = np.hstack((channel[:, -horiz_offset:], np.zeros((channel.shape[0], -horiz_offset))))
        
        return channel

    # Rotate channels
    red_rotated = rotate(red_channel, red_rotate, reshape=False, mode='constant')
    green_rotated = rotate(green_channel, green_rotate, reshape=False, mode='constant')
    blue_rotated = rotate(blue_channel, blue_rotate, reshape=False, mode='constant')
    
    # Apply offsets to the rotated channels
    red_shifted = apply_offsets(red_rotated, red_offset, red_shift)
    green_shifted = apply_offsets(green_rotated, green_offset, green_shift)
    blue_shifted = apply_offsets(blue_rotated, blue_offset, blue_shift)
    
    # Normalize the shifted channels
    red_norm = red_shifted / red_shifted.max() if red_shifted.max() > 0 else red_shifted
    green_norm = green_shifted / green_shifted.max() if green_shifted.max() > 0 else green_shifted
    blue_norm = blue_shifted / blue_shifted.max() if blue_shifted.max() > 0 else blue_shifted
    
    # Stack the R, G, and B channels to make a color image
    color_image = np.dstack((red_norm, green_norm, blue_norm))
    
    return color_image



def apply_np_roll_to_bands(channel, roll_values):
    """
    Applies np.roll to each band in a channel with a specified roll value.
    
    Parameters:
        channel (numpy.ndarray): The image channel containing the bands.
        roll_values (list): A list of integers specifying the roll amount for each band.
        
    Returns:
        numpy.ndarray: The channel with adjusted bands.
    """
    # Check if the number of roll_values provided matches the number of bands in the channel
    number_of_bands = 16
    if len(roll_values) != number_of_bands:
        raise ValueError("The length of roll_values must match the number of bands in the channel.")
    
    # Apply np.roll to each band
    band_height = 128
    adjusted_channel = np.copy(channel)
    for i in range(number_of_bands):
        start_index = i * band_height
        end_index = start_index + band_height
        adjusted_channel[start_index:end_index] = np.roll(
            channel[start_index:end_index], roll_values[i], axis=1
        )
        
    return adjusted_channel

