"""
Configurations file defining the options for the model. The options specified 
here denotes the number of channels in the generator and the label organization.
The generator is constructed using these options.
"""

# Architecture Details
channel_size = 3
img_dim = 256
latent_dim = 2048
encoder_channels = [32, 64, 128, 256, 512, 512, 512, 1024, 2048]
decoder_channels = [2048, 1024, 512, 512, 512, 256, 128, 64, 32]

# Label Organization
## The tags and attributes are labeled with this index order.
tags = [
    {
        "name": "Bangs", # Tag 0
        "attributes": ["with", "without"] # Attributes 0, 1
    },
    {
        "name": "Eyeglasses", # Tag 1
        "attributes": ["with", "without"] # Attributes 0, 1
    },
    {
        "name": "Hair_Color", # Tag 2
        "attributes": ["black", "brown", "blond"] # Attributes 0, 1, 2
    },
    {
        "name": "Young", # Tag 3
        "attributes": ["with", "without"] # Attributes 0, 1
    },
    {
        "name": "Male", # Tag 4
        "attributes": ["with", "without"] # Attributes 0, 1
    },
    {
        "name": "Smiling", # Tag 5
        "attributes": ["with", "without"] # Attributes 0, 1
    },
]